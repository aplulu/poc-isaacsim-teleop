"""Free-viewpoint ("immersive") mode: a stereo camera rig in Isaac Sim that follows the user's head.

Coordinate frames
-----------------
* XR space (WebXR local-floor): x right, y up, z towards the user (Y-up). A pose in this space is what the
  headset reports. Its camera convention (looks down -Z, +Y up) is the same as a USD camera's.
* Sim world: Z-up. The XR origin (the floor point the user calibrated) is placed in the sim world by an
  "anchor" = (position on the floor, yaw). ``C`` converts XR axes to the anchor's local Z-up axes:
  XR -z (forward) -> +x, XR x (right) -> -y, XR y (up) -> +z.

The head camera pose in the sim world is ``Rz(yaw) @ C @ R_head`` at ``anchor + Rz(yaw) @ C @ p_head``. Every
rendered frame is tagged with the *XR-space* head pose that was used, so the web app can show the frame at
exactly that pose (world-locked) and rendering latency does not make the image swim with head rotation.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import omni.replicator.core as rep
from pxr import Gf, Sdf, UsdGeom

from . import config as cfg
from .mathutil import quat_xyzw_to_matrix, rot_z

if TYPE_CHECKING:
    from .sim import SimController

HEAD_CAMERA = f"{cfg.CAMERAS_ROOT}/Head"
C_XR = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # XR axes -> anchor-local Z-up axes
MOVE_SPEED = 0.8  # m/s (thumbstick locomotion)
SNAP_TURN_DEG = 30.0


class ImmersiveView:
    def __init__(self, sim: SimController) -> None:
        self.sim = sim
        self.stage = sim.stage
        scfg = sim.scfg
        self.fov = float(scfg.immersive_fov)
        self.size = (scfg.immersive_size, scfg.immersive_size)
        # default stance: across the table from the robot, facing it (and the work area)
        wx, wy = cfg.WORK_AREA_CENTER_XY
        self.default_pos = np.array([float(sim.table_max[0]) + 0.45, wy, 0.0])
        self.default_yaw = math.pi  # facing -X (towards the robot)
        self.anchor_pos = self.default_pos.copy()
        self.anchor_yaw = self.default_yaw
        self.head_xr: tuple[list, list] | None = None  # latest head pose (p, q xyzw) in XR space
        self.used_pose: dict | None = None  # head pose used for the frame being rendered
        self.pose_ts: float | None = None  # web app clock (performance.now) of the XR sample that pose came from
        self.pose_applied = 0.0  # perf_counter when the pose was applied
        self.pose_rx: float | None = None  # server wall clock [ms] when that sample arrived
        # poses applied in the last Kit updates (newest last): the grabbed image belongs to an older one
        self._pose_hist: deque = deque(maxlen=8)
        self._turn_armed = True
        self._create_rig()
        self.render_products = {}
        self.annotators = {}
        for eye in ("L", "R"):
            rp = rep.create.render_product(f"{HEAD_CAMERA}/Stereo{eye}", self.size, name=f"Head_{eye}")
            annot = rep.AnnotatorRegistry.get_annotator("rgb", device=getattr(sim, "annot_device", "cpu"))
            annot.attach(rp)
            self.render_products["H" + eye] = rp
            self.annotators["H" + eye] = annot

    # -- rig ------------------------------------------------------------------------------------------
    def _create_rig(self) -> None:
        aperture = 20.955
        focal = aperture / (2 * math.tan(math.radians(self.fov) / 2))
        head = UsdGeom.Camera.Define(self.stage, HEAD_CAMERA)
        self._head_op = UsdGeom.Xformable(head).AddTransformOp()
        self._head_op.Set(Gf.Matrix4d(1))
        self._eye_ops = []
        for eye, sign in (("L", -1.0), ("R", 1.0)):
            cam = UsdGeom.Camera.Define(self.stage, f"{HEAD_CAMERA}/Stereo{eye}")
            cam.CreateFocalLengthAttr(focal)
            cam.CreateHorizontalApertureAttr(aperture)
            cam.CreateVerticalApertureAttr(aperture * self.size[1] / self.size[0])
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.02, 100.0))
            op = UsdGeom.Xformable(cam).AddTranslateOp()
            op.Set(Gf.Vec3d(sign * self.sim.ipd / 2, 0, 0))
            self._eye_ops.append((op, sign))
        head.GetPrim().CreateAttribute("teleop:fov", Sdf.ValueTypeNames.Double).Set(self.fov)

    def set_ipd(self, ipd: float) -> None:
        for op, sign in self._eye_ops:
            op.Set(Gf.Vec3d(sign * ipd / 2, 0, 0))

    # -- mapping --------------------------------------------------------------------------------------
    @property
    def R_anchor(self) -> np.ndarray:
        return rot_z(self.anchor_yaw)

    def xr_to_world(self) -> np.ndarray:
        """3x3 matrix mapping XR vectors to world vectors (exact: what you see is where it moves)."""
        return self.R_anchor @ C_XR

    def head_world(self, p_xr, q_xr) -> np.ndarray:
        M = self.xr_to_world()
        T = np.eye(4)
        T[:3, :3] = M @ quat_xyzw_to_matrix(q_xr)
        T[:3, 3] = self.anchor_pos + M @ np.asarray(p_xr, dtype=float)
        return T

    # -- per frame ------------------------------------------------------------------------------------
    def update(self, sample: dict | None, dt: float, control_hand: str) -> None:
        if sample:
            head = sample.get("head")
            if head and "p" in head and "q" in head:
                self.head_xr = (head["p"], head["q"])
                ts = sample.get("ts")
                self.pose_ts = float(ts) if isinstance(ts, (int, float)) else None
                self.pose_rx = sample.get("_rx")
            self._locomotion(sample, dt, control_hand)
        if self.head_xr is None:
            return
        T = self.head_world(*self.head_xr)
        # USD (row-vector) matrix: rows = camera axes and translation
        m = Gf.Matrix4d(*[float(v) for v in T.T.reshape(-1)])
        self._head_op.Set(m)
        self.used_pose = {"p": list(map(float, self.head_xr[0])), "q": list(map(float, self.head_xr[1]))}
        self.pose_applied = time.perf_counter()
        self._pose_hist.append((self.used_pose, self.pose_ts, self.pose_rx, self.pose_applied))

    def rendered_pose(self) -> tuple[dict, float | None, float | None, float] | None:
        """(head pose, sample ts, server receive ms, applied perf_counter) of the image the annotator returns now."""
        if not self._pose_hist:
            return None
        k = min(cfg.IMMERSIVE_RENDER_LAG_FRAMES, len(self._pose_hist) - 1)
        return self._pose_hist[-1 - k]

    def _locomotion(self, sample: dict, dt: float, control_hand: str) -> None:
        other = "left" if control_hand == "right" else "right"
        move = (sample.get(other) or {}).get("axes")
        if move and len(move) >= 2 and self.head_xr is not None:
            x, y = float(move[0]), float(move[1])
            if abs(x) > 0.15 or abs(y) > 0.15:
                # move in the direction the user is looking (horizontal), stick forward = -y
                fwd = self.xr_to_world() @ quat_xyzw_to_matrix(self.head_xr[1]) @ np.array([0.0, 0.0, -1.0])
                fwd[2] = 0
                fwd /= np.linalg.norm(fwd) or 1.0
                right = np.array([fwd[1], -fwd[0], 0.0])
                self.anchor_pos = self.anchor_pos + (fwd * -y + right * x) * MOVE_SPEED * min(dt, 0.1)
        turn = (sample.get(control_hand) or {}).get("axes")
        if turn and len(turn) >= 1:
            x = float(turn[0])
            if self._turn_armed and abs(x) > 0.75:
                self._snap_turn(-math.copysign(math.radians(SNAP_TURN_DEG), x))
                self._turn_armed = False
            elif abs(x) < 0.3:
                self._turn_armed = True

    def _snap_turn(self, angle: float) -> None:
        """Rotate the anchor about the user's current head position (so the user does not swing around)."""
        if self.head_xr is None:
            self.anchor_yaw += angle
            return
        head = self.head_world(*self.head_xr)[:3, 3]
        R = rot_z(angle)
        self.anchor_pos = head + R @ (self.anchor_pos - head)
        self.anchor_pos[2] = 0.0
        self.anchor_yaw += angle

    def recenter(self) -> dict:
        """Put the user's current head at the default stance, looking at the work area."""
        if self.head_xr is None:
            self.anchor_pos, self.anchor_yaw = self.default_pos.copy(), self.default_yaw
            return self.status()
        p, q = self.head_xr
        fwd_xr = quat_xyzw_to_matrix(q) @ np.array([0.0, 0.0, -1.0])
        fwd_local = C_XR @ fwd_xr
        head_yaw_local = math.atan2(fwd_local[1], fwd_local[0])
        self.anchor_yaw = self.default_yaw - head_yaw_local
        head_local = C_XR @ np.asarray(p, dtype=float)
        head_local[2] = 0.0
        self.anchor_pos = self.default_pos - self.R_anchor @ head_local
        self.anchor_pos[2] = 0.0
        return self.status()

    def status(self) -> dict:
        return {
            "fov": self.fov,
            "size": list(self.size),
            "anchor": [round(float(v), 3) for v in self.anchor_pos[:2]] + [round(math.degrees(self.anchor_yaw), 1)],
        }
