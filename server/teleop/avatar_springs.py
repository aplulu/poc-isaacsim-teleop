"""VRM spring bones (hair, skirt, ribbons) as one NVIDIA Warp kernel on the CPU.

The numpy version processed the springs depth level by level (a dozen levels, ~1000 small numpy calls per
frame: ~10 ms, p95 25 ms). Here every independent spring subtree is one kernel thread that walks its joints in
parent-before-child order, so the per-call overhead disappears. Same algorithm as ``Avatar._simulate_springs_np``:
verlet tail, stiffness towards the rest direction, gravity, drag, sphere colliders, then the joint is swung so
that its tail points at the simulated position.
"""

from __future__ import annotations

import sys

import numpy as np
import warp as wp

DEVICE = "cpu"


@wp.func
def _rot(m: wp.mat44) -> wp.mat33:
    return wp.mat33(m[0, 0], m[0, 1], m[0, 2], m[1, 0], m[1, 1], m[1, 2], m[2, 0], m[2, 1], m[2, 2])


@wp.func
def _compose(r: wp.mat33, t: wp.vec3) -> wp.mat44:
    return wp.mat44(
        r[0, 0], r[0, 1], r[0, 2], t[0],
        r[1, 0], r[1, 1], r[1, 2], t[1],
        r[2, 0], r[2, 1], r[2, 2], t[2],
        0.0, 0.0, 0.0, 1.0,
    )


@wp.func
def _safe_unit(v: wp.vec3) -> wp.vec3:
    n = wp.length(v)
    return v / wp.max(n, 1.0e-9)


@wp.kernel
def spring_kernel(
    order: wp.array(dtype=wp.int32),
    starts: wp.array(dtype=wp.int32),
    ends: wp.array(dtype=wp.int32),
    parents: wp.array(dtype=wp.int32),
    slot: wp.array(dtype=wp.int32),
    local: wp.array(dtype=wp.mat44),  # in/out: spring joints get their new rotation
    G: wp.array(dtype=wp.mat44),  # in/out: model-space globals (non-spring parents filled by the caller)
    W: wp.mat44,  # model -> world (includes the avatar scale)
    Rw: wp.mat33,  # rotation part of W without the scale
    s: float,
    dt: float,
    sp_dir: wp.array(dtype=wp.vec3),
    sp_rest_rot: wp.array(dtype=wp.mat33),
    sp_len: wp.array(dtype=float),
    sp_stiff: wp.array(dtype=float),
    sp_grav: wp.array(dtype=wp.vec3),
    sp_drag: wp.array(dtype=float),
    sp_hit: wp.array(dtype=float),
    cols: wp.array(dtype=wp.vec3),
    col_rad: wp.array(dtype=float),
    colmask: wp.array(dtype=wp.int32),
    ncol: int,
    tail_cur: wp.array(dtype=wp.vec3),
    tail_prev: wp.array(dtype=wp.vec3),
    init: int,
):
    t = wp.tid()
    for i in range(starts[t], ends[t]):
        j = order[i]
        Gp = G[parents[j]]
        L = local[j]
        k = slot[j]
        if k >= 0:
            Gj = Gp * L
            pos = wp.transform_point(W, wp.vec3(Gj[0, 3], Gj[1, 3], Gj[2, 3]))
            Rp_w = Rw * _rot(Gp)
            rest_dir = Rp_w * sp_dir[k]
            Lk = sp_len[k] * s
            cur = tail_cur[k]
            prev = tail_prev[k]
            if init != 0:
                cur = pos + rest_dir * Lk
                prev = cur
            nxt = cur + (cur - prev) * (1.0 - sp_drag[k]) + rest_dir * (sp_stiff[k] * dt * s) + (Rw * sp_grav[k]) * (dt * s)
            nxt = pos + _safe_unit(nxt - pos) * Lk
            for c in range(ncol):
                if colmask[k * ncol + c] != 0:
                    hit = (sp_hit[k] + col_rad[c]) * s
                    d = nxt - cols[c]
                    dist = wp.length(d)
                    if dist < hit:
                        pushed = cols[c] + d / wp.max(dist, 1.0e-9) * hit
                        nxt = pos + _safe_unit(pushed - pos) * Lk
            tail_prev[k] = cur
            tail_cur[k] = nxt
            q = wp.quat_between_vectors(_safe_unit(rest_dir), _safe_unit(nxt - pos))
            R_w = wp.quat_to_matrix(q) * Rp_w * sp_rest_rot[k]
            R_l = wp.inverse(_rot(Gp)) * (wp.transpose(Rw) * R_w)
            L = _compose(R_l, wp.vec3(L[0, 3], L[1, 3], L[2, 3]))
            local[j] = L
        G[j] = Gp * L


@wp.kernel
def fk_kernel(order: wp.array(dtype=wp.int32), n: int, parents: wp.array(dtype=wp.int32),
              local: wp.array(dtype=wp.mat44), G: wp.array(dtype=wp.mat44)):
    # one thread, parent-before-child order (a few hundred 4x4 products: microseconds natively)
    for i in range(n):
        j = order[i]
        p = parents[j]
        if p < 0:
            G[j] = local[j]
        else:
            G[j] = G[p] * local[j]


@wp.kernel
def anim_kernel(anim: wp.array(dtype=wp.int32), local: wp.array(dtype=wp.mat44),
                rot: wp.array(dtype=wp.vec4), trans: wp.array(dtype=wp.vec3)):
    # SkelAnimation values: local rotation as a quaternion (x, y, z, w = Gf.Quatf memory order), translation
    t = wp.tid()
    m = local[anim[t]]
    # the rotation block carries a slight scale (bones compensate their parents' rest scale): take its polar
    # rotation (two Newton steps R <- (R + R^-T) / 2) before converting
    r = _rot(m)
    r = (r + wp.transpose(wp.inverse(r))) * 0.5
    r = (r + wp.transpose(wp.inverse(r))) * 0.5
    q = wp.normalize(wp.quat_from_matrix(r))
    rot[t] = wp.vec4(q[0], q[1], q[2], q[3])
    trans[t] = wp.vec3(m[0, 3], m[1, 3], m[2, 3])


class SpringSolver:
    """Static data on the Warp CPU device + reusable per-frame buffers (numpy views, no copies)."""

    def __init__(self, parents: np.ndarray, sp_joint: np.ndarray, params: dict, n_joints: int,
                 anim_joints: np.ndarray | None = None, col_joint: np.ndarray | None = None,
                 col_off: np.ndarray | None = None) -> None:
        wp.init()
        n_sp = len(sp_joint)
        slot = np.full(n_joints, -1, dtype=np.int32)
        slot[sp_joint] = np.arange(n_sp, dtype=np.int32)
        kids: list[list[int]] = [[] for _ in range(n_joints)]
        for j in range(n_joints):
            if parents[j] >= 0:
                kids[parents[j]].append(j)
        # roots: springs whose ancestors are not springs; each thread walks the whole subtree under its root
        is_sp = slot >= 0
        under = np.zeros(n_joints, dtype=bool)  # strictly below a spring
        order_parent_first = []
        stack = [j for j in range(n_joints) if parents[j] < 0]
        while stack:
            j = stack.pop()
            order_parent_first.append(j)
            stack.extend(kids[j])
        for j in order_parent_first:
            p = parents[j]
            under[j] = p >= 0 and (under[p] or is_sp[p])
        roots = [j for j in order_parent_first if is_sp[j] and not under[j]]
        order, starts, ends = [], [], []
        for r in roots:
            starts.append(len(order))
            st = [r]
            while st:  # depth-first, parent before child
                j = st.pop()
                order.append(j)
                st.extend(reversed(kids[j]))
            ends.append(len(order))
        self.subtree_joints = np.array(order, dtype=int)
        self.n_threads = len(roots)
        f = np.float32
        self.order = wp.array(np.array(order, dtype=np.int32), dtype=wp.int32, device=DEVICE)
        self.starts = wp.array(np.array(starts, dtype=np.int32), dtype=wp.int32, device=DEVICE)
        self.ends = wp.array(np.array(ends, dtype=np.int32), dtype=wp.int32, device=DEVICE)
        self.parents = wp.array(parents.astype(np.int32), dtype=wp.int32, device=DEVICE)
        self.slot = wp.array(slot, dtype=wp.int32, device=DEVICE)
        self.sp_dir = wp.array(params["dir"].astype(f), dtype=wp.vec3, device=DEVICE)
        self.sp_rest_rot = wp.array(params["rest_rot"].astype(f), dtype=wp.mat33, device=DEVICE)
        self.sp_len = wp.array(params["len"].astype(f), dtype=float, device=DEVICE)
        self.sp_stiff = wp.array(params["stiff"].astype(f), dtype=float, device=DEVICE)
        self.sp_grav = wp.array(params["grav"].astype(f), dtype=wp.vec3, device=DEVICE)
        self.sp_drag = wp.array(params["drag"].astype(f), dtype=float, device=DEVICE)
        self.sp_hit = wp.array(params["hit"].astype(f), dtype=float, device=DEVICE)
        ncol = params["colmask"].shape[1]
        self.ncol = int(ncol)
        self.colmask = wp.array(params["colmask"].astype(np.int32).reshape(-1), dtype=wp.int32, device=DEVICE)
        self.col_rad = wp.array(params["col_rad"].astype(f), dtype=float, device=DEVICE)
        self.cols = wp.zeros(max(ncol, 1), dtype=wp.vec3, device=DEVICE)
        self.local = wp.zeros(n_joints, dtype=wp.mat44, device=DEVICE)
        self.G = wp.zeros(n_joints, dtype=wp.mat44, device=DEVICE)
        self.tail_cur = wp.zeros(n_sp, dtype=wp.vec3, device=DEVICE)
        self.tail_prev = wp.zeros(n_sp, dtype=wp.vec3, device=DEVICE)
        # zero-copy numpy views of the per-frame buffers
        self.local_np = self.local.numpy().reshape(n_joints, 4, 4)
        self.G_np = self.G.numpy().reshape(n_joints, 4, 4)
        self.cols_np = self.cols.numpy().reshape(-1, 3)
        # FK of everything outside the spring subtrees (their parents), parent first
        in_sub = np.zeros(n_joints, dtype=bool)
        in_sub[self.subtree_joints] = True
        fk = [j for j in order_parent_first if not in_sub[j]]
        self.fk_order = wp.array(np.array(fk, dtype=np.int32), dtype=wp.int32, device=DEVICE)
        self.n_fk = len(fk)
        anim = np.arange(n_joints) if anim_joints is None else np.asarray(anim_joints)
        self.anim = wp.array(anim.astype(np.int32), dtype=wp.int32, device=DEVICE)
        self.rot = wp.zeros(len(anim), dtype=wp.vec4, device=DEVICE)
        self.trans = wp.zeros(len(anim), dtype=wp.vec3, device=DEVICE)
        self.rot_np = self.rot.numpy().reshape(-1, 4)
        self.trans_np = self.trans.numpy().reshape(-1, 3)
        self.col_joint = None if col_joint is None or not len(col_joint) else np.asarray(col_joint)
        self.col_off = None if col_off is None else np.asarray(col_off, dtype=np.float64)
        wp.load_module(module=sys.modules[__name__], device=DEVICE)  # no stall at the first step

    def tails(self) -> tuple[np.ndarray, np.ndarray]:
        return self.tail_cur.numpy().reshape(-1, 3), self.tail_prev.numpy().reshape(-1, 3)

    def step(self, local: np.ndarray | None, W: np.ndarray, s: float, dt: float, init: bool) -> None:
        """Whole skeleton after the IK: FK, collider positions, springs, SkelAnimation values (``rot_np`` /
        ``trans_np``). ``local`` (n,4,4) float64 is updated in place for the spring joints; None: the locals are
        already in ``self.local`` (written by the Warp IK)."""
        if local is not None:
            self.local_np[:] = local
        wp.launch(fk_kernel, dim=1, inputs=[self.fk_order, self.n_fk, self.parents, self.local, self.G], device=DEVICE)
        ncol = 0
        if self.col_joint is not None:
            Gc = self.G_np[self.col_joint].astype(np.float64)
            self.cols_np[: len(Gc)] = (Gc @ self.col_off[:, :, None])[:, :3, 0] @ W[:3, :3].T + W[:3, 3]
            ncol = self.ncol
        Rw = W[:3, :3] / s
        wp.launch(
            spring_kernel,
            dim=self.n_threads,
            inputs=[
                self.order, self.starts, self.ends, self.parents, self.slot, self.local, self.G,
                wp.mat44(*W.astype(np.float32).reshape(-1)), wp.mat33(*Rw.astype(np.float32).reshape(-1)),
                float(s), float(dt), self.sp_dir, self.sp_rest_rot, self.sp_len, self.sp_stiff, self.sp_grav,
                self.sp_drag, self.sp_hit, self.cols, self.col_rad, self.colmask,
                ncol, self.tail_cur, self.tail_prev, int(init),
            ],
            device=DEVICE,
        )
        wp.launch(anim_kernel, dim=len(self.rot_np), inputs=[self.anim, self.local, self.rot, self.trans], device=DEVICE)
        if local is not None:
            sub = self.subtree_joints
            local[sub] = self.local_np[sub]

    def run(self, local: np.ndarray, G: np.ndarray, W: np.ndarray, s: float, dt: float,
            cols: np.ndarray | None, init: bool) -> None:
        """local / G: (n,4,4) float64 model-space (G valid for every joint outside the spring subtrees).
        Updates ``local`` (spring rotations) and ``G`` (spring subtrees) in place."""
        self.local_np[:] = local
        self.G_np[:] = G
        if cols is not None and len(cols):
            self.cols_np[: len(cols)] = cols
        Rw = W[:3, :3] / s
        wp.launch(
            spring_kernel,
            dim=self.n_threads,
            inputs=[
                self.order, self.starts, self.ends, self.parents, self.slot, self.local, self.G,
                wp.mat44(*W.astype(np.float32).reshape(-1)), wp.mat33(*Rw.astype(np.float32).reshape(-1)),
                float(s), float(dt), self.sp_dir, self.sp_rest_rot, self.sp_len, self.sp_stiff, self.sp_grav,
                self.sp_drag, self.sp_hit, self.cols, self.col_rad, self.colmask,
                self.ncol if cols is not None else 0, self.tail_cur, self.tail_prev, int(init),
            ],
            device=DEVICE,
        )
        sub = self.subtree_joints
        local[sub] = self.local_np[sub]
        G[sub] = self.G_np[sub]
