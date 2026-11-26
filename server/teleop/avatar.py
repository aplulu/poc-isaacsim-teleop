"""The operator's VRM avatar in the sim, driven by the headset and controller poses (3-point IK).

Frames
------
* model: the avatar's own skeleton space (glTF, Y-up), what the SkelAnimation is written in.
* body : x forward, y left, z up, origin on the floor under the hips, in model units (unscaled).
         ``meta["model_to_body"]`` (C) maps model vectors to body vectors.
* world: the sim. ``world = O + s * Rz(yaw) @ body`` is the transform written to ``/World/Avatar``.

Solver (per frame)
------------------
1. body yaw follows the head (and the hands), the scale follows the user's standing eye height,
2. hips are placed under the neck (they drop / move back when the user crouches or leans),
3. spine bends so the neck reaches the head, neck/head take the headset orientation,
4. arms: two-bone IK to the wrist under each controller (they may stretch a little to reach),
   hands take the controller orientation, fingers curl with trigger (index) / grip (others),
5. legs: two-bone IK to planted feet that step when the body moves away from them.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdSkel, Vt

from . import config as cfg
from .perf import PERF
from .mathutil import matrix_from_rotvec, matrix_to_quat_wxyz, quat_xyzw_to_matrix, rot_z, rotvec_from_matrix

log = logging.getLogger("teleop.avatar")

FINGERS = ("Thumb", "Index", "Middle", "Ring", "Little")
SEGMENTS = ("Proximal", "Intermediate", "Distal")


def _cross(a, b) -> np.ndarray:
    """3-vector cross product (np.cross has a large per-call overhead)."""
    a0, a1, a2 = float(a[0]), float(a[1]), float(a[2])
    b0, b1, b2 = float(b[0]), float(b[1]), float(b[2])
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


def _unit(v: np.ndarray) -> np.ndarray:
    n = math.sqrt(float(v[0]) ** 2 + float(v[1]) ** 2 + float(v[2]) ** 2)
    return v / n if n > 1e-9 else v


def _basis(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Orthonormal basis with first axis along a and second axis towards b."""
    x = _unit(a)
    y = _unit(b - x * float(np.dot(b, x)))
    return np.stack([x, y, _cross(x, y)], 1)


def _swing(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest rotation taking direction a to direction b."""
    a, b = _unit(a), _unit(b)
    c = _cross(a, b)
    s, d = np.linalg.norm(c), float(np.dot(a, b))
    if s < 1e-9:
        if d > 0:
            return np.eye(3)
        p = _unit(_cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0]))
        return matrix_from_rotvec(p * math.pi)
    return matrix_from_rotvec(c / s * math.atan2(s, d))


def _slerp_rot(R: np.ndarray, t: float) -> np.ndarray:
    return matrix_from_rotvec(rotvec_from_matrix(R) * t)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class Avatar:
    def __init__(self, stage, meta_path: Path, prim_path: str = cfg.AVATAR_PRIM) -> None:
        self.meta = m = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        self.stage = stage
        self.prim_path = prim_path
        self.C = np.array(m["model_to_body"], dtype=float)
        self.parents = np.array(m["parents"], dtype=int)
        self.rest_local = np.array(m["rest_local"], dtype=float)  # (N,4,4) model frame
        self.anim_joints = list(m["anim_joints"])
        self.h = dict(m["humanoid"])
        self.n = len(self.parents)
        self.G0 = self._fk_all(self.rest_local)
        self.p0 = {b: self.C @ self.G0[j][:3, 3] for b, j in self.h.items()}  # rest positions (body)
        self.eye0 = np.array(m["eye_body"], dtype=float)
        self._init_springs(m.get("springs", []), m.get("colliders", []))
        self.bodies = [b for b in m.get("bodies", []) if b["bone"] in self.h]
        self._W: np.ndarray | None = None  # model -> world of the last solved frame
        # animation prim / root xform
        prim = stage.GetPrimAtPath(prim_path)
        self.prim = prim
        self.anim = UsdSkel.Animation(stage.GetPrimAtPath(f"{prim_path}/Root/Skeleton/Anim"))
        xf = UsdGeom.Xformable(prim)
        ops = xf.GetOrderedXformOps()
        self.root_op = ops[0] if ops and ops[0].GetOpType() == UsdGeom.XformOp.TypeTransform else None
        if self.root_op is None:
            xf.ClearXformOpOrder()
            self.root_op = xf.AddTransformOp()
        self.imageable = UsdGeom.Imageable(prim)
        self.visible = True
        # state
        self.yaw: float | None = None
        self.scale = float(cfg.AVATAR_SCALE) if cfg.AVATAR_SCALE else 1.0
        self.auto_scale = not cfg.AVATAR_SCALE
        self.user_eye: float | None = None
        self.feet: dict[str, dict] = {}
        self.last_world: dict[str, np.ndarray] = {}
        self.ok = True
        self._local = self.rest_local.copy()
        self._G = self.G0.copy()
        self._valid = np.zeros(self.n, dtype=bool)
        kids: list[list[int]] = [[] for _ in range(self.n)]
        for j in range(self.n):
            if self.parents[j] >= 0:
                kids[self.parents[j]].append(j)
        self._desc = []
        for j in range(self.n):
            out, stack = [], [j]
            while stack:
                k = stack.pop()
                out.append(k)
                stack.extend(kids[k])
            self._desc.append(np.array(out, dtype=int))
        self._scale_hint: float | None = None
        chain = [b for b in ("spine", "chest", "upperChest") if b in self.h]
        self.spine_chain = chain
        self.arm_len = {s: (self._dist(f"{s}UpperArm", f"{s}LowerArm"), self._dist(f"{s}LowerArm", f"{s}Hand")) for s in ("left", "right")}
        self.leg_len = {s: (self._dist(f"{s}UpperLeg", f"{s}LowerLeg"), self._dist(f"{s}LowerLeg", f"{s}Foot")) for s in ("left", "right")}
        self.ik = None  # Warp IK (needs the spring solver's buffers); None: the numpy reference path
        if cfg.AVATAR_IK_WARP and self.spring_solver is not None:
            try:
                from .avatar_ik import IkSolver

                self.ik = IkSolver(self, self.spring_solver.local)
                log.info("avatar IK: Warp kernel")
            except Exception:  # noqa: BLE001 - fall back to numpy
                log.exception("Warp avatar IK unavailable; using numpy")
                self.ik = None
        log.info("avatar %s: %d joints, eye height %.3f (model units)", m.get("source"), self.n, self.eye0[2])

    # ---------------------------------------------------------------------------------------------
    def _dist(self, a: str, b: str) -> float:
        return float(np.linalg.norm(self.p0[b] - self.p0[a]))

    def _fk_all(self, local: np.ndarray) -> np.ndarray:
        G = np.empty_like(local)
        for j in range(len(local)):  # joints are stored parents-first
            p = self.parents[j]
            G[j] = local[j] if p < 0 else G[p] @ local[j]
        return G

    def _refresh(self, j: int) -> np.ndarray:
        """Global (model) transform of joint j from the current locals (cached until a local above changes)."""
        if self._valid[j]:
            return self._G[j]
        path = []
        k = j
        while k >= 0 and not self._valid[k]:
            path.append(k)
            k = self.parents[k]
        G = self._G[k] if k >= 0 else np.eye(4)
        for k in reversed(path):
            G = G @ self._local[k]
            self._G[k] = G
            self._valid[k] = True
        return G

    def _touch(self, j: int) -> None:
        self._valid[self._desc[j]] = False

    def _pos(self, bone: str) -> np.ndarray:
        return self.C @ self._refresh(self.h[bone])[:3, 3]

    def _rest_rot(self, bone: str) -> np.ndarray:
        return self.C @ self.G0[self.h[bone]][:3, :3] @ self.C.T

    def _set_rot(self, bone: str, R_body: np.ndarray) -> None:
        """Set the global (body frame) rotation of a bone."""
        j = self.h[bone]
        p = self.parents[j]
        Gp = self._refresh(p) if p >= 0 else np.eye(4)
        Rm = self.C.T @ R_body @ self.C
        Rp = Gp[:3, :3]
        # exact inverse, not Rp^T: rest transforms carry tiny scales, and with the transpose the error compounded
        # along each chain (the hands ended up ~5% sheared, fingertips ~28%)
        self._local[j][:3, :3] = np.linalg.solve(Rp, Rm)
        self._touch(j)

    def _set_delta(self, bone: str, D: np.ndarray) -> None:
        """Rotate a bone so that its global rotation is D @ rest."""
        self._set_rot(bone, D @ self._rest_rot(bone))

    def _set_pos(self, bone: str, p_body: np.ndarray) -> None:
        j = self.h[bone]
        p = self.parents[j]
        Gp = self._refresh(p) if p >= 0 else np.eye(4)
        pm = self.C.T @ p_body
        self._local[j][:3, 3] = np.linalg.inv(Gp)[:3, :3] @ (pm - Gp[:3, 3])
        self._touch(j)

    def _set_len(self, bone: str, factor: float) -> None:
        """Stretch the bone from its parent (rest offset times factor)."""
        j = self.h[bone]
        self._local[j][:3, 3] = self.rest_local[j][:3, 3] * factor
        self._touch(j)

    # ---------------------------------------------------------------------------------------------
    def set_visible(self, visible: bool) -> None:
        if visible != self.visible:
            self.visible = visible
            (self.imageable.MakeVisible if visible else self.imageable.MakeInvisible)()

    def update(self, head_w: np.ndarray | None, hands_w: dict[str, dict], dt: float, scale: float | None = None) -> None:
        """head_w: 4x4 world pose of the headset (camera axes: x right, y up, -z forward).
        hands_w: side -> {"T": 4x4 world grip pose, "trigger": float, "grip": float}."""
        if head_w is None or not self.ok:
            return
        try:
            self._scale_hint = scale
            self._solve(head_w, hands_w, dt)
        except Exception:  # noqa: BLE001 - never take the sim loop down
            log.exception("avatar IK failed; avatar disabled")
            self.ok = False

    def _solve(self, head_w: np.ndarray, hands_w: dict[str, dict], dt: float) -> None:
        self._t_solve = time.perf_counter()
        dt = min(max(dt, 1e-3), 0.1)
        Hw = head_w[:3, :3]
        E_w = head_w[:3, 3]
        # head axes as (forward, left, up)
        fwd, left, up = -Hw[:, 2], -Hw[:, 0], Hw[:, 1]
        Hw_body_axes = np.stack([fwd, left, up], 1)

        # 1. scale from the user's standing eye height
        if self.auto_scale and self._scale_hint is not None:
            self.scale = float(np.clip(self._scale_hint, 0.4, 3.0))
        elif self.auto_scale:
            z = float(E_w[2])
            if self.user_eye is None:
                self.user_eye = max(z, 1.2)
            elif z > self.user_eye:
                self.user_eye += (z - self.user_eye) * min(1.0, dt * 3.0)
            else:
                self.user_eye -= 0.002 * dt
            self.user_eye = float(np.clip(self.user_eye, 1.0, 2.1))
            self.scale = float(np.clip(self.user_eye / max(self.eye0[2], 0.3), 0.4, 3.0))
        s = self.scale

        # body yaw: head heading, pulled towards the hands, never more than 50 deg away from the head
        d = fwd - up * fwd[2]
        yaw_h = math.atan2(d[1], d[0])
        target = yaw_h
        if "left" in hands_w and "right" in hands_w:
            mid = (hands_w["left"]["T"][:3, 3] + hands_w["right"]["T"][:3, 3]) / 2 - E_w
            if math.hypot(mid[0], mid[1]) > 0.15:
                target = yaw_h + 0.4 * _wrap(math.atan2(mid[1], mid[0]) - yaw_h)
        if self.yaw is None:
            self.yaw = target
        self.yaw += _wrap(target - self.yaw) * min(1.0, dt * 3.0)
        off = _wrap(self.yaw - yaw_h)
        lim = math.radians(50)
        if abs(off) > lim:
            self.yaw = yaw_h + math.copysign(lim, off)
        Rb = rot_z(self.yaw)

        # 2. hips under the neck
        H_b = Rb.T @ Hw_body_axes  # head rotation relative to the body frame
        head0, neck0, hips0 = self.p0["head"], self.p0["neck"], self.p0["hips"]
        Hd_w = E_w - s * Rb @ H_b @ (self.eye0 - head0)
        R_half = _slerp_rot(H_b, 0.5)
        N_w = Hd_w - s * Rb @ R_half @ (head0 - neck0)
        down = max(0.0, s * neck0[2] - float(N_w[2]))
        hips_w = N_w - s * Rb @ np.array([neck0[0] - hips0[0], neck0[1] - hips0[1], 0.0])
        hips_w = hips_w - Rb @ np.array([0.5 * down, 0.0, 0.0])
        hips_z = s * hips0[2] - 0.55 * down
        O = np.array([hips_w[0], hips_w[1], 0.0]) - s * Rb @ np.array([hips0[0], hips0[1], 0.0])

        def to_body(p):
            return Rb.T @ (np.asarray(p, dtype=float) - O) / s

        N_b = to_body(N_w)
        twist = _wrap(yaw_h - self.yaw) * 0.5
        if self.ik is not None:
            self._solve_warp(hands_w, to_body, Rb, s, O, hips_z, N_b, twist, H_b, dt)
        else:
            self._solve_np(hands_w, to_body, Rb, s, O, hips_z, N_b, twist, H_b, dt)

        t_ik = time.perf_counter()
        PERF.add("av.ik", (t_ik - self._t_solve) * 1e3)
        # 6. hair / skirt / ribbons (spring bones)
        W = np.eye(4)
        W[:3, :3] = rot_z(self.yaw) @ self.C * s
        W[:3, 3] = O
        self._simulate_springs(W, s, dt)
        t_sp = time.perf_counter()
        PERF.add("av.springs", (t_sp - t_ik) * 1e3)

        # write
        self._W = W
        self._write(O, s)
        PERF.add("av.write", (time.perf_counter() - t_sp) * 1e3)

    def _solve_warp(self, hands_w, to_body, Rb, s, O, hips_z, N_b, twist, H_b, dt) -> None:
        """Skeleton-independent targets here, the skeleton pass in the Warp kernel (avatar_ik.py)."""
        ik = self.ik
        hips0 = self.p0["hips"]
        ik.fv_np[0] = (hips0[0], hips0[1], hips_z / s)
        ik.fv_np[1] = N_b
        ik.fm_np[0] = H_b
        ik.ff_np[0] = twist
        ik.ff_np[7] = cfg.AVATAR_MAX_STRETCH
        for si, (side, sign) in enumerate((("left", 1.0), ("right", -1.0))):
            hand = hands_w.get(side)
            if hand is not None:
                self.last_world[side] = hand["T"]
            if ik.has_arm[si]:
                Wt, hand_D = self._hand_target(side, sign, hand["T"] if hand is not None else None, Rb, to_body)
                ik.fi_np[si] = Wt is not None
                if Wt is not None:
                    ik.fv_np[2 + si] = Wt
                    ik.fm_np[1 + si] = hand_D
                hand = hand or {}
                ik.ff_np[1 + 2 * si] = float(hand.get("trigger", 0.0) or 0.0)
                ik.ff_np[2 + 2 * si] = float(hand.get("grip", 0.0) or 0.0)
            if ik.has_leg[si]:
                F, foot_yaw = self._foot_target(side, O, Rb, s, to_body, dt)
                ik.fv_np[4 + si] = F
                ik.ff_np[5 + si] = foot_yaw
        ik.run()

    def _solve_np(self, hands_w, to_body, Rb, s, O, hips_z, N_b, twist, H_b, dt) -> None:
        """Reference path: the same IK bone by bone in numpy."""
        head0, neck0, hips0 = self.p0["head"], self.p0["neck"], self.p0["hips"]
        self._local = self.rest_local.copy()
        self._valid[:] = False
        self._set_pos("hips", np.array([hips0[0], hips0[1], hips_z / s]))

        # 3. spine -> neck target, neck/head orientation
        hips_b = self._pos("hips")
        D = _swing(neck0 - hips0, N_b - hips_b) @ rot_z(twist)
        k = len(self.spine_chain)
        for i, b in enumerate(self.spine_chain):
            self._set_delta(b, _slerp_rot(D, (i + 1) / k))
        if k:
            last = self.spine_chain[-1]
            lp = self._pos(last)
            corr = _swing(self._pos("neck") - lp, N_b - lp)
            R_last = self.C @ self._refresh(self.h[last])[:3, :3] @ self.C.T
            self._set_rot(last, corr @ R_last)
        chest_D = self.C @ self._refresh(self.h[self.spine_chain[-1]] if k else self.h["hips"])[:3, :3] @ self.C.T
        chest_D = chest_D @ (self._rest_rot(self.spine_chain[-1] if k else "hips")).T
        self._set_delta("neck", chest_D @ _slerp_rot(chest_D.T @ H_b, 0.5))
        self._set_delta("head", H_b)

        # 4. arms
        for side, sign in (("left", 1.0), ("right", -1.0)):
            hand = hands_w.get(side)
            if hand is not None:
                self.last_world[side] = hand["T"]
            T = hand["T"] if hand is not None else None
            self._solve_arm(side, sign, T, hand or {}, to_body, Rb, s)

        # 5. legs
        for side, sign in (("left", 1.0), ("right", -1.0)):
            self._solve_leg(side, sign, O, Rb, s, to_body, dt)

    # ---------------------------------------------------------------------------------------------
    def _hand_target(self, side, sign, T_w, Rb, to_body) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Wrist target (body frame) and the hand's rotation delta for a controller pose; (None, None) without."""
        if T_w is None:
            return None, None
        ha_b, lo_b = f"{side}Hand", f"{side}LowerArm"
        f0 = _unit(self.p0[ha_b] - self.p0[lo_b])
        Rg = T_w[:3, :3]
        off = np.array(cfg.AVATAR_WRIST_OFFSET) * np.array([-sign, 1.0, 1.0])
        W = to_body(T_w[:3, 3] + Rg @ off)
        a = Rb.T @ (Rg @ np.array([0.0, 0.0, -1.0]))
        palm = Rb.T @ (Rg @ np.array([sign, 0.0, 0.0]))  # left palm faces +X(grip), right palm -X
        pitch = matrix_from_rotvec(_unit(_cross(a, palm)) * math.radians(cfg.AVATAR_HAND_PITCH_DEG))
        a = pitch @ a
        a0 = _unit(self.p0.get(f"{side}MiddleProximal", self.p0[ha_b] + f0 * 0.05) - self.p0[ha_b])
        return W, _basis(a, palm) @ _basis(a0, np.array([0.0, 0.0, -1.0])).T

    def _solve_arm(self, side, sign, T_w, hand, to_body, Rb, s) -> None:
        up_b, lo_b, ha_b = f"{side}UpperArm", f"{side}LowerArm", f"{side}Hand"
        if not all(b in self.h for b in (up_b, lo_b, ha_b)):
            return
        u0 = _unit(self.p0[lo_b] - self.p0[up_b])
        f0 = _unit(self.p0[ha_b] - self.p0[lo_b])
        bend = np.array([1.0, 0.0, 0.0])  # the forearm folds forward
        n0 = _unit(_cross(u0, bend))
        S = self._pos(up_b)
        l1, l2 = self.arm_len[side]
        W, hand_D = self._hand_target(side, sign, T_w, Rb, to_body)
        if W is None:
            # no controller: let the arm hang down
            W = S + np.array([0.02, sign * 0.04, -(l1 + l2) * 0.98])
        dvec = W - S
        dist = float(np.linalg.norm(dvec))
        stretch = 1.0
        if dist > (l1 + l2) * 0.999:
            stretch = min(dist / ((l1 + l2) * 0.999), cfg.AVATAR_MAX_STRETCH)
        L1, L2 = l1 * stretch, l2 * stretch
        dist = float(np.clip(dist, 1e-4, (L1 + L2) * 0.999))
        u = _unit(dvec)
        pole = _unit(np.array([-0.35, sign * 0.5, -1.0]))
        v = _unit(pole - u * float(np.dot(pole, u)))
        if np.linalg.norm(v) < 1e-6:
            v = _unit(_cross(u, [0.0, 0.0, 1.0]))
        cos_a = (L1 * L1 + dist * dist - L2 * L2) / (2 * L1 * dist)
        a_ang = math.acos(float(np.clip(cos_a, -1.0, 1.0)))
        E = S + L1 * (math.cos(a_ang) * u + math.sin(a_ang) * v)
        Wc = S + u * dist
        n = _unit(_cross(E - S, Wc - E))
        if np.linalg.norm(n) < 1e-6:
            n = n0
        up_dir = _unit(E - S)
        lo_dir = _unit(Wc - E)
        D_up = np.stack([up_dir, n, _cross(up_dir, n)], 1) @ np.stack([u0, n0, _cross(u0, n0)], 1).T
        self._set_delta(up_b, D_up)
        if stretch > 1.0:
            self._set_len(lo_b, stretch)
            self._set_len(ha_b, stretch)
        D_lo = np.stack([lo_dir, n, _cross(lo_dir, n)], 1) @ np.stack([f0, n0, _cross(f0, n0)], 1).T
        if hand_D is not None:
            # forearm takes half of the wrist twist (about the forearm axis) to avoid a candy-wrapper wrist
            palm_now = D_lo @ np.array([0.0, 0.0, -1.0])
            palm_tgt = hand_D @ np.array([0.0, 0.0, -1.0])
            p1 = _unit(palm_now - lo_dir * float(np.dot(palm_now, lo_dir)))
            p2 = _unit(palm_tgt - lo_dir * float(np.dot(palm_tgt, lo_dir)))
            ang = math.atan2(float(np.dot(_cross(p1, p2), lo_dir)), float(np.dot(p1, p2)))
            D_lo = matrix_from_rotvec(lo_dir * ang * 0.5) @ D_lo
        self._set_delta(lo_b, D_lo)
        if hand_D is None:
            hand_D = D_lo
        self._set_delta(ha_b, hand_D)
        # fingers
        trig = float(hand.get("trigger", 0.0) or 0.0)
        grip = float(hand.get("grip", 0.0) or 0.0)
        if cfg.AVATAR_FINGERS_BATCHED:
            self._solve_fingers(side, hand_D, trig, grip)
            return
        palm0 = np.array([0.0, 0.0, -1.0])
        for f in FINGERS:
            if f == "Thumb":
                curls = (0.25 + 0.25 * grip, 0.25 + 0.2 * grip, 0.2 + 0.2 * grip)
            else:
                v_ = trig if f == "Index" else grip
                curls = (0.2 + 1.2 * v_, 0.25 + 1.3 * v_, 0.15 + 0.9 * v_)
            total = 0.0
            names = [f"{side}{f}{seg}" for seg in SEGMENTS]
            for i, (b, c) in enumerate(zip(names, curls)):
                if b not in self.h:
                    continue
                if i < 2 and names[i + 1] in self.p0:
                    d0 = _unit(self.p0[names[i + 1]] - self.p0[b])
                elif i > 0 and names[i - 1] in self.p0:
                    d0 = _unit(self.p0[b] - self.p0[names[i - 1]])
                else:
                    d0 = a0_dir(self, side)
                axis = _unit(_cross(d0, palm0)) if f != "Thumb" else _unit(_cross(d0, palm0 + np.array([0.0, sign * 0.6, 0.0])))
                total += c
                self._set_delta(b, hand_D @ matrix_from_rotvec(axis * total))

    def _init_fingers(self) -> None:
        """Per side: the finger segments in parent-before-child order with their curl axis and coefficients."""
        palm0 = np.array([0.0, 0.0, -1.0])
        self._fingers = {}
        for side, sign in (("left", 1.0), ("right", -1.0)):
            joints, axes, base, gain, src, finger, rest = [], [], [], [], [], [], []
            for fi, f in enumerate(FINGERS):
                if f == "Thumb":
                    curls = ((0.25, 0.25, 1), (0.25, 0.2, 1), (0.2, 0.2, 1))
                else:
                    sv = 0 if f == "Index" else 1  # 0 = trigger, 1 = grip
                    curls = ((0.2, 1.2, sv), (0.25, 1.3, sv), (0.15, 0.9, sv))
                names = [f"{side}{f}{seg}" for seg in SEGMENTS]
                for i, (b, (c0, c1, sv)) in enumerate(zip(names, curls)):
                    if b not in self.h:
                        continue
                    if i < 2 and names[i + 1] in self.p0:
                        d0 = _unit(self.p0[names[i + 1]] - self.p0[b])
                    elif i > 0 and names[i - 1] in self.p0:
                        d0 = _unit(self.p0[b] - self.p0[names[i - 1]])
                    else:
                        d0 = a0_dir(self, side)
                    pv = palm0 if f != "Thumb" else palm0 + np.array([0.0, sign * 0.6, 0.0])
                    joints.append(self.h[b])
                    axes.append(_unit(_cross(d0, pv)))
                    base.append(c0)
                    gain.append(c1)
                    src.append(sv)
                    finger.append(fi)
                    rest.append(self._rest_rot(b))
            if not joints:
                continue
            n = len(joints)
            finger = np.array(finger)
            cum = np.tril(np.ones((n, n))) * (finger[:, None] == finger[None, :])  # sum over earlier segments
            pidx = np.array([joints.index(self.parents[j]) if self.parents[j] in joints else -1 for j in joints])
            desc = np.unique(np.concatenate([self._desc[j] for j in joints]))
            self._fingers[side] = {
                "joints": np.array(joints), "axes": np.array(axes), "base": np.array(base), "gain": np.array(gain),
                "src": np.array(src), "finger": finger, "cum": cum, "rest": np.array(rest), "pidx": pidx, "desc": desc,
                "outer_parents": sorted({int(self.parents[j]) for j in joints if self.parents[j] not in joints}),
            }

    def _solve_fingers(self, side: str, hand_D: np.ndarray, trig: float, grip: float) -> None:
        """Batched finger curl: global rotation of each segment = hand_D @ R(axis * cumulative curl) @ rest."""
        fd = getattr(self, "_fingers", None)
        if fd is None:
            self._init_fingers()
            fd = self._fingers
        f = fd.get(side)
        if f is None:
            return
        v = np.where(f["src"] == 0, trig, grip)
        total = f["cum"] @ (f["base"] + f["gain"] * v)
        k = f["axes"]
        a = total[:, None, None]
        K = np.zeros((len(k), 3, 3))
        K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
        K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
        R = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)
        Rm = self.C.T @ (hand_D @ R @ f["rest"]) @ self.C  # model-space globals
        joints, pidx = f["joints"], f["pidx"]
        direct = pidx >= 0
        # segments whose parent is another segment: one batched product
        if direct.any():
            jd = joints[direct]
            self._local[jd, :3, :3] = np.linalg.solve(Rm[pidx[direct]], Rm[direct])
        self._valid[f["desc"]] = False
        # segments under a non-finger joint (the hand, or an extra bone between segments): parent from FK,
        # in order, so that an extra bone below an already written segment sees its new rotation
        for i in np.nonzero(~direct)[0]:
            j = joints[i]
            Rp = self._refresh(int(self.parents[j]))[:3, :3]
            self._local[j, :3, :3] = np.linalg.solve(Rp, Rm[i])
            self._valid[self._desc[j]] = False

    def _solve_leg(self, side, sign, O, Rb, s, to_body, dt) -> None:
        up_b, lo_b, ft_b = f"{side}UpperLeg", f"{side}LowerLeg", f"{side}Foot"
        if not all(b in self.h for b in (up_b, lo_b, ft_b)):
            return
        F, foot_yaw = self._foot_target(side, O, Rb, s, to_body, dt)
        S = self._pos(up_b)
        l1, l2 = self.leg_len[side]
        u0 = _unit(self.p0[lo_b] - self.p0[up_b])
        f0 = _unit(self.p0[ft_b] - self.p0[lo_b])
        n0 = _unit(_cross(u0, np.array([-1.0, 0.0, 0.0])))  # the shin folds backwards
        dvec = F - S
        dist = float(np.clip(np.linalg.norm(dvec), 1e-4, (l1 + l2) * 0.999))
        u = _unit(dvec)
        pole = _unit(np.array([1.0, sign * 0.15, 0.0]))
        v = _unit(pole - u * float(np.dot(pole, u)))
        cos_a = (l1 * l1 + dist * dist - l2 * l2) / (2 * l1 * dist)
        ang = math.acos(float(np.clip(cos_a, -1.0, 1.0)))
        K = S + l1 * (math.cos(ang) * u + math.sin(ang) * v)
        Fc = S + u * dist
        n = _unit(_cross(K - S, Fc - K))
        if np.linalg.norm(n) < 1e-6:
            n = n0
        a1, a2 = _unit(K - S), _unit(Fc - K)
        self._set_delta(up_b, np.stack([a1, n, _cross(a1, n)], 1) @ np.stack([u0, n0, _cross(u0, n0)], 1).T)
        self._set_delta(lo_b, np.stack([a2, n, _cross(a2, n)], 1) @ np.stack([f0, n0, _cross(f0, n0)], 1).T)
        self._set_delta(ft_b, rot_z(foot_yaw))

    def _foot_target(self, side, O, Rb, s, to_body, dt) -> tuple[np.ndarray, float]:
        """Planted foot (body frame) and its yaw relative to the body; steps when the body moves away / turns."""
        ft_b = f"{side}Foot"
        # ideal foot spot (world) under the hips, at rest spacing
        ideal = O + s * Rb @ np.array([self.p0[ft_b][0], self.p0[ft_b][1], 0.0])
        ideal[2] = s * self.p0[ft_b][2]
        f = self.feet.get(side)
        if f is None:
            f = self.feet[side] = {"p": ideal.copy(), "yaw": self.yaw, "from": None, "t": 0.0}
        far = np.linalg.norm((f["p"] - ideal)[:2]) > cfg.AVATAR_STEP_DIST
        turned = abs(_wrap(f["yaw"] - self.yaw)) > math.radians(35)
        other = self.feet.get("right" if side == "left" else "left")
        if f["from"] is None and (far or turned) and not (other and other["from"] is not None):
            f["from"] = (f["p"].copy(), f["yaw"])
            f["t"] = 0.0
        if f["from"] is not None:
            f["t"] = min(1.0, f["t"] + dt / 0.25)
            p_from, y_from = f["from"]
            a = f["t"]
            f["p"] = p_from + (ideal - p_from) * a
            f["p"][2] = ideal[2] + math.sin(math.pi * a) * 0.06 * s
            f["yaw"] = y_from + _wrap(self.yaw - y_from) * a
            if a >= 1.0:
                f["from"] = None
        return to_body(f["p"]), _wrap(f["yaw"] - self.yaw)

    # ---------------------------------------------------------------------------------------------
    # spring bones (VRM SpringBone: verlet tails, stiffness towards the rest direction, gravity, drag,
    # sphere colliders), simulated in world space so that the avatar's own motion swings them
    def _init_springs(self, springs: list[dict], colliders: list[dict]) -> None:
        keep = []
        for e in springs:
            j, c = e["joint"], e["child"]
            if c >= 0:
                axis = self.rest_local[c][:3, 3].copy()
            else:  # leaf: virtual tail 7 cm along the parent -> joint direction (in the joint's frame)
                p = self.parents[j]
                d = self.G0[j][:3, 3] - (self.G0[p][:3, 3] if p >= 0 else 0)
                axis = self.G0[j][:3, :3].T @ (_unit(d) * 0.07)
            length = float(np.linalg.norm(axis))
            if length > 1e-5:
                keep.append((e, axis / length, length))
        n = len(keep)
        self.sp_joint = np.array([e["joint"] for e, _, _ in keep], dtype=int)
        # rest direction of the tail in the parent's frame: rest_rot @ axis
        self.sp_dir = np.array([self.rest_local[e["joint"]][:3, :3] @ ax for e, ax, _ in keep]).reshape(n, 3)
        self.sp_rest_rot = np.array([self.rest_local[e["joint"]][:3, :3] for e, _, _ in keep]).reshape(n, 3, 3)
        self.sp_len = np.array([ln for _, _, ln in keep])
        self.sp_stiff = np.array([e["stiffness"] for e, _, _ in keep])
        self.sp_grav = np.array([np.asarray(e["gravityDir"], dtype=float) * e["gravityPower"] for e, _, _ in keep]).reshape(n, 3)
        self.sp_drag = np.array([e["drag"] for e, _, _ in keep])
        self.sp_hit = np.array([e["hitRadius"] for e, _, _ in keep])
        self.col_joint = np.array([c["joint"] for c in colliders], dtype=int)
        self.col_off = np.array([[*c["offset"], 1.0] for c in colliders]).reshape(len(colliders), 4)
        self.col_rad = np.array([c["radius"] for c in colliders])
        self.sp_colmask = np.zeros((n, len(colliders)), dtype=bool)
        for k, (e, _, _) in enumerate(keep):
            for ci in e["colliders"]:
                if ci < len(colliders):
                    self.sp_colmask[k, ci] = True
        # FK levels (joints of equal depth are computed together) and the springs of each level
        depth = np.zeros(self.n, dtype=int)
        for j in range(self.n):
            depth[j] = depth[self.parents[j]] + 1 if self.parents[j] >= 0 else 0
        slot = {int(j): k for k, j in enumerate(self.sp_joint)}
        self.levels = []
        for d in range(int(depth.max()) + 1):
            js = np.nonzero(depth == d)[0]
            sk = np.array([slot[int(j)] for j in js if int(j) in slot], dtype=int)
            self.levels.append((js, self.parents[js], sk))
        self._tails = None  # (n, 2, 3) current / previous tail, world
        self._last_root = None
        self._springs_init = True  # next Warp step restarts the tails from the rest pose
        self.spring_solver = None
        if n and cfg.AVATAR_SPRINGS_WARP:
            try:
                from .avatar_springs import SpringSolver

                self.spring_solver = SpringSolver(self.parents, self.sp_joint, {
                    "dir": self.sp_dir, "rest_rot": self.sp_rest_rot, "len": self.sp_len, "stiff": self.sp_stiff,
                    "grav": self.sp_grav, "drag": self.sp_drag, "hit": self.sp_hit, "colmask": self.sp_colmask,
                    "col_rad": self.col_rad,
                }, self.n, anim_joints=np.array(self.anim_joints), col_joint=self.col_joint, col_off=self.col_off)
                log.info("spring bones: Warp kernel (%d springs, %d chains)", n, self.spring_solver.n_threads)
            except Exception:  # noqa: BLE001 - fall back to numpy
                log.exception("Warp spring bones unavailable; using numpy")
                self.spring_solver = None

    def _fk_levels(self, local: np.ndarray, springs=None) -> np.ndarray:
        """Vectorized FK by depth; ``springs(level_spring_slots, G)`` may rewrite locals level by level."""
        G = np.empty_like(local)
        eye = np.eye(4)
        for js, ps, sk in self.levels:
            if (ps < 0).all():
                G[js] = local[js]
            else:
                Gp = np.where((ps >= 0)[:, None, None], G[np.maximum(ps, 0)], eye)
                G[js] = Gp @ local[js]
            if springs is not None and len(sk):
                springs(sk, G)
        return G

    def _simulate_springs(self, W: np.ndarray, s: float, dt: float) -> None:
        if len(self.sp_joint) == 0:
            return
        if self.spring_solver is None:
            self._simulate_springs_np(W, s, dt)
            return
        Wt = W[:3, 3]
        if self._last_root is not None and np.linalg.norm(Wt - self._last_root) > 0.5:
            self._springs_init = True  # teleported (recenter / view switch)
        self._last_root = Wt.copy()
        # FK + colliders + springs + SkelAnimation values in Warp; _write takes the values from the solver
        sv = self.spring_solver
        if self.ik is not None:
            # the Warp IK already wrote the locals into the solver; mirror the results for body_poses & co.
            sv.step(None, W, s, dt, self._springs_init)
            self._local[:] = sv.local_np
            self._G[:] = sv.G_np
            self._valid[:] = True
        else:
            sv.step(self._local, W, s, dt, self._springs_init)
        self._springs_init = False

    def _simulate_springs_np(self, W: np.ndarray, s: float, dt: float) -> None:
        Rw = W[:3, :3] / s
        Wl, Wt = W[:3, :3], W[:3, 3]
        if self._last_root is not None and np.linalg.norm(Wt - self._last_root) > 0.5:
            self._tails = None  # teleported (recenter / view switch)
        self._last_root = Wt.copy()
        # colliders sit on humanoid bones: their poses after the IK
        cols = None
        if len(self.col_joint):
            Gc = np.stack([self._refresh(int(j)) for j in self.col_joint])
            cols = (Gc @ self.col_off[:, :, None])[:, :3, 0] @ Wl.T + Wt
        col_r = self.col_rad * s
        init = self._tails is None
        if init:
            self._tails = np.zeros((len(self.sp_joint), 2, 3))
        tails = self._tails
        local = self._local

        def step(sk: np.ndarray, G: np.ndarray) -> None:
            js = self.sp_joint[sk]
            Gp = G[self.parents[js]]
            pos = G[js][:, :3, 3] @ Wl.T + Wt
            Rp_w = Rw @ Gp[:, :3, :3]  # (k,3,3)
            rest_dir = np.einsum("kij,kj->ki", Rp_w, self.sp_dir[sk])
            L = (self.sp_len[sk] * s)[:, None]
            cur, prev = tails[sk, 0], tails[sk, 1]
            if init:
                cur = prev = pos + rest_dir * L
            nxt = (
                cur
                + (cur - prev) * (1.0 - self.sp_drag[sk])[:, None]
                + rest_dir * (self.sp_stiff[sk] * dt * s)[:, None]
                + (self.sp_grav[sk] @ Rw.T) * (dt * s)
            )
            v = nxt - pos
            nxt = pos + v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9) * L
            if cols is not None:
                hit = (self.sp_hit[sk] * s)[:, None] + col_r[None, :]  # (k,c)
                for ci in range(len(col_r)):
                    m = self.sp_colmask[sk, ci]
                    if not m.any():
                        continue
                    d = nxt - cols[ci]
                    dist = np.linalg.norm(d, axis=1)
                    inside = m & (dist < hit[:, ci])
                    if inside.any():
                        pushed = cols[ci] + d[inside] / np.maximum(dist[inside], 1e-9)[:, None] * hit[inside, ci][:, None]
                        w = pushed - pos[inside]
                        nxt[inside] = pos[inside] + w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-9) * L[inside]
            tails[sk, 1] = cur
            tails[sk, 0] = nxt
            # swing rest_dir -> (nxt - pos), batched Rodrigues
            a = rest_dir / np.maximum(np.linalg.norm(rest_dir, axis=1, keepdims=True), 1e-9)
            b = nxt - pos
            b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
            ax = np.stack([a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1], a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2], a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]], 1)
            sn = np.linalg.norm(ax, axis=1)
            cs = np.einsum("ki,ki->k", a, b)
            ang = np.arctan2(sn, cs)
            k_ = ax / np.maximum(sn, 1e-12)[:, None]
            K = np.zeros((len(sk), 3, 3))
            K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k_[:, 2], k_[:, 1], -k_[:, 0]
            K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = k_[:, 2], -k_[:, 1], k_[:, 0]
            Rs = np.eye(3) + np.sin(ang)[:, None, None] * K + (1 - np.cos(ang))[:, None, None] * (K @ K)
            R_w = Rs @ Rp_w @ self.sp_rest_rot[sk]
            local[js, :3, :3] = np.linalg.solve(Gp[:, :3, :3], Rw.T @ R_w)
            G[js] = Gp @ local[js]

        self._fk_levels(local, step)

    # ---------------------------------------------------------------------------------------------
    def _write(self, O: np.ndarray, s: float) -> None:
        sv = self.spring_solver
        if sv is not None:
            q, t = sv.rot_np, sv.trans_np  # computed by the Warp step from the final locals
        else:
            L = self._local[self.anim_joints]
            q = _batch_quat_xyzw(L[:, :3, :3]).astype(np.float32)
            t = np.ascontiguousarray(L[:, :3, 3].astype(np.float32))
        try:
            rots = Vt.QuatfArray.FromNumpy(np.ascontiguousarray(q))
        except Exception:  # noqa: BLE001 - Vt without FromNumpy for quaternions
            rots = Vt.QuatfArray([Gf.Quatf(float(w), float(x), float(y), float(z)) for x, y, z, w in q])
        M = np.eye(4)
        M[:3, :3] = rot_z(self.yaw) * s
        M[:3, 3] = O
        self.anim.GetRotationsAttr().Set(rots)
        self.anim.GetTranslationsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(t)))
        self.root_op.Set(Gf.Matrix4d(*[float(v) for v in M.T.reshape(-1)]))

    def body_poses(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """World poses of the collision bodies: positions (n,3), quaternions xyzw (n,4) with the capsule axis
        along local +Z, radii (n,), cylinder half heights (n,). None before the first solved frame."""
        if self._W is None or not self.bodies:
            return None
        W, s = self._W, self.scale
        Rw = W[:3, :3] / s
        if getattr(self, "_body_arr", None) is None:
            self._body_arr = (
                np.array([self.h[b["bone"]] for b in self.bodies]),
                np.array([b["center"] for b in self.bodies], dtype=float),
                np.array([b["axis"] for b in self.bodies], dtype=float),
                np.array([b["radius"] for b in self.bodies], dtype=float),
                np.array([b["half"] for b in self.bodies], dtype=float),
            )
        bones, center, axis, radius, halfh = self._body_arr
        G = np.stack([self._refresh(int(j)) for j in bones])
        c = np.einsum("kij,kj->ki", G[:, :3, :3], center) + G[:, :3, 3]
        pos = c @ W[:3, :3].T + W[:3, 3]
        ax = np.einsum("kij,kj->ki", G[:, :3, :3], axis) @ Rw.T
        ax /= np.maximum(np.linalg.norm(ax, axis=1, keepdims=True), 1e-12)
        # shortest rotation +Z -> ax as a quaternion (x, y, z, w); antiparallel -> 180 deg about X
        w = 1.0 + ax[:, 2]
        quat = np.stack([-ax[:, 1], ax[:, 0], np.zeros(len(ax)), w], 1)
        flip = w < 1e-9
        quat[flip] = (1.0, 0.0, 0.0, 0.0)
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        return pos, quat, radius * s, halfh * s

    def status(self) -> dict:
        return {
            "source": self.meta.get("source"),
            "shown": self.visible,
            "scale": round(self.scale, 3),
            "autoScale": self.auto_scale,
            "ok": self.ok,
        }


def _batch_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """(n,3,3) rotation matrices -> (n,4) quaternions (x, y, z, w), the Gf.Quatf memory order.

    Shepperd's method (divide by the largest component): well conditioned also for bones whose matrices carry a
    slight scale, where taking every component from the diagonal was off by ~2 deg. The matrices are first replaced
    by their polar rotation (two Newton steps), as in the Warp path."""
    for _ in range(2):
        R = (R + np.transpose(np.linalg.inv(R), (0, 2, 1))) * 0.5
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    q = np.empty((len(R), 4))
    c = np.argmax(np.stack([tr, m00, m11, m22], 1), axis=1)
    k = c == 0
    if k.any():
        r = np.sqrt(np.maximum(1 + tr[k], 1e-12)) * 2
        q[k] = np.stack([(R[k, 2, 1] - R[k, 1, 2]) / r, (R[k, 0, 2] - R[k, 2, 0]) / r, (R[k, 1, 0] - R[k, 0, 1]) / r, r / 4], 1)
    k = c == 1
    if k.any():
        r = np.sqrt(np.maximum(1 + m00[k] - m11[k] - m22[k], 1e-12)) * 2
        q[k] = np.stack([r / 4, (R[k, 0, 1] + R[k, 1, 0]) / r, (R[k, 0, 2] + R[k, 2, 0]) / r, (R[k, 2, 1] - R[k, 1, 2]) / r], 1)
    k = c == 2
    if k.any():
        r = np.sqrt(np.maximum(1 - m00[k] + m11[k] - m22[k], 1e-12)) * 2
        q[k] = np.stack([(R[k, 0, 1] + R[k, 1, 0]) / r, r / 4, (R[k, 1, 2] + R[k, 2, 1]) / r, (R[k, 0, 2] - R[k, 2, 0]) / r], 1)
    k = c == 3
    if k.any():
        r = np.sqrt(np.maximum(1 - m00[k] - m11[k] + m22[k], 1e-12)) * 2
        q[k] = np.stack([(R[k, 0, 2] + R[k, 2, 0]) / r, (R[k, 1, 2] + R[k, 2, 1]) / r, r / 4, (R[k, 1, 0] - R[k, 0, 1]) / r], 1)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def a0_dir(av: Avatar, side: str) -> np.ndarray:
    return _unit(av.p0[f"{side}Hand"] - av.p0[f"{side}LowerArm"])


def hands_from_sample(sample: dict | None, to_world) -> dict[str, dict]:
    """XR sample -> {side: {"T": world grip pose, "trigger", "grip"}} (to_world maps an XR pose to 4x4)."""
    out = {}
    if not sample:
        return out
    for side in ("left", "right"):
        h = sample.get(side)
        if h and "p" in h and "q" in h:
            out[side] = {"T": to_world(h["p"], h["q"]), "trigger": h.get("trigger", 0.0), "grip": h.get("grip", 0.0)}
    return out


def xr_pose_matrix(p, q) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = quat_xyzw_to_matrix(q)
    T[:3, 3] = p
    return T
