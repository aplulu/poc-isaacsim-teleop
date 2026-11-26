"""Simulation-side controller. Runs on the Kit main thread only.

* Robot: FANUC CRX-10iA/L articulation driven by a damped-least-squares IK on the PhysX Jacobian.
* Teleop: a "clutch" scheme. While the grip button is held, the controller's motion relative to the moment
  the grip was pressed is applied to the tool target (positions are mapped from the WebXR frame (Y-up) into
  the Isaac Sim world (Z-up) using the yaw of the active view camera, so "forward" means "into the screen").
* Gripper: attach-style (the closest prop within reach is rigidly carried by the tool while closed).
* Cameras: render product + RGB annotator -> frames published to WebRTC video tracks.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from . import config as cfg
from .bridge import CommandQueue, FrameHub, StateHub, TeleopInput
from .perf import PERF
from .mathutil import (
    gf_to_np,
    inv_tf,
    make_tf,
    matrix_from_rotvec,
    matrix_to_quat_wxyz,
    quat_wxyz_to_matrix,
    quat_xyzw_to_matrix,
    rot_z,
    rotvec_from_matrix,
    skew,
)
from .randomization import DomainRandomizer

log = logging.getLogger("teleop.sim")

CAMERA_PATHS = {
    "operator": f"{cfg.CAMERAS_ROOT}/Operator",
    "front": f"{cfg.CAMERAS_ROOT}/Front",
    "top": f"{cfg.CAMERAS_ROOT}/Top",
    "dataset": f"{cfg.CAMERAS_ROOT}/Dataset",
    "wrist": cfg.WRIST_CAMERA,
}

# WebXR (x right, y up, z back) -> operator frame columns are filled per camera yaw
GRIP_ON, GRIP_OFF = 0.6, 0.3


class Prop:
    def __init__(self, name: str, root: str, body: str, category: str) -> None:
        self.name = name
        self.root = root
        self.body = body
        self.category = category
        self.center_local = np.zeros(3)  # bbox center in the body frame
        self.bottom_offset = 0.0  # body origin height above the bbox bottom (upright)
        self.radius = 0.05  # horizontal bounding radius
        self.half_height = 0.05
        self.default_mass = 0.1
        self.default_quat = np.array([1.0, 0, 0, 0])
        self.home_xy = np.zeros(2)  # bounding-box center on the table when its set is active
        self.fixed = False
        self.size = np.zeros(3)


class SimController:
    def __init__(
        self,
        server_cfg: cfg.ServerConfig,
        frames: FrameHub,
        state: StateHub,
        teleop: TeleopInput,
        commands: CommandQueue,
    ) -> None:
        self.scfg = server_cfg
        self.frames = frames
        self.state = state
        self.teleop = teleop
        self.commands = commands
        self.stage: Usd.Stage = omni.usd.get_context().get_stage()
        self.timeline = omni.timeline.get_timeline_interface()
        self.ready = False
        self.camera = "operator"
        # rotationMode: "yaw" (gripper keeps its tilt, only turns about the vertical axis), "full", "off"
        # mapping: how hand motion is turned into arm motion in the screen view
        #   "camera"   - relative to the camera currently shown (re-evaluated every time the clutch engages)
        #   "operator" - always relative to the operator camera, whichever camera is shown
        #   "robot"    - pushing the hand forward moves the gripper along the robot's +X (away from its base)
        # In the free-viewpoint view the mapping is always exact (VR space == sim world).
        self.settings = {
            "positionScale": 1.0,
            "rotationMode": "yaw",
            "smoothing": 0.5,
            "hand": "right",
            "mapping": "camera",
            # grasp assist: once the fingers have closed on an object, carry it rigidly with the tool
            "graspAssist": True,
            "handMode": "robot",
        }
        self._assist: dict | None = None  # pending/active grasp assist state
        self.view_mode = "screen"  # "screen" (video on a virtual screen) | "immersive" (head-tracked stereo)
        self.object_set = cfg.DEFAULT_OBJECT_SET  # graspable object set on the table
        self.immersive = None
        self.avatar = None  # operator avatar (teleop.avatar.Avatar), loaded from a VRM in setup()
        # collision: "on" / "off" (the avatar and its colliders exist only in the free viewpoint)
        self.avatar_settings = {"visible": True, "hideWhileRecording": True, "collision": "on"}
        self.avatar_body_view: RigidPrim | None = None
        # free viewpoint: grab props with your own hands (grip button). handMode "robot" = the control hand
        # drives the robot and the other hand grabs, "grab" = both hands grab (robot teleop paused)
        self._hand_hold: dict[str, dict | None] = {"left": None, "right": None}
        self._hand_release_t: dict[str, float] = {}
        self._hand_prev: dict[str, tuple[np.ndarray, float]] = {}
        self._hand_hold_v: dict[str, np.ndarray] = {}
        self._hand_grip_down: dict[str, bool] = {}
        self._avatar_body_state = {"active": False, "scale": 0.0, "shapes": []}
        self.pos_filter = OneEuroFilter()
        self.T_link_wrist = np.eye(4)
        # robot
        self.robot: Articulation | None = None
        self.home = np.radians(np.array(cfg.HOME_JOINTS_DEG, dtype=float))
        self.q_cmd: np.ndarray | None = None
        self.limits: np.ndarray | None = None
        self.link_index = 0
        self.T_link_tcp = np.eye(4)
        self.target_T: np.ndarray | None = None
        self.resync_target = True
        # clutch state
        self.clutch = False
        self.clutch_ctrl0: tuple[np.ndarray, np.ndarray] | None = None
        self.clutch_target0: np.ndarray | None = None
        self.clutch_M = np.eye(3)
        # gripper
        self.gripper_closed = False
        self.gripper_cmd_closed = False  # last request coming from the UI/trigger
        self._trigger_was_down = False
        self.held: Prop | None = None
        self.held_offset = np.eye(4)
        self.prev_tcp: np.ndarray | None = None
        self.tcp_vel = np.zeros(3)
        self.finger_ops = []
        self.gripper_type = "virtual"
        self.arm_idx: np.ndarray | None = None  # DOF indices of J1..J6
        self.grip_idx: np.ndarray | None = None  # DOF index of the Robotiq finger_joint
        self.gripper_pos = 0.0  # 0 = open .. 1 = closed
        self.rec = None  # RecordingManager
        # props
        self.props: list[Prop] = []
        self.prop_view: RigidPrim | None = None
        self.default_poses: tuple[np.ndarray, np.ndarray] | None = None
        # table
        table = self.stage.GetPrimAtPath(cfg.TABLE_PRIM)
        self.table_top = float(table.GetAttribute("teleop:tableTop").Get() or 0.85)
        tmin = table.GetAttribute("teleop:tableMin").Get()
        tmax = table.GetAttribute("teleop:tableMax").Get()
        self.table_min = np.array(tmin if tmin is not None else (0.2, -0.9))
        self.table_max = np.array(tmax if tmax is not None else (0.9, 0.9))
        # rendering
        # eye -> render product / annotator. "M" = mono stream, "L"/"R" = stereo pair (side by side)
        self.render_products: dict[str, object] = {}
        self.annotators: dict[str, object] = {}
        self.stereo = bool(server_cfg.stereo)
        self.ipd = float(server_cfg.stereo_ipd)
        self._rig_ops: list[tuple[object, float]] = []
        self._rp_state: dict[str, bool] = {}
        self._stream_warmup = 0
        self.viewport_mode = "auto"  # Isaac Sim's viewport: "auto" (paused while streaming) | "on" | "off"
        self._viewport_on: bool | None = None
        self.kit_tracer = None  # carb profiler interface when main.py runs with --kit-trace
        self.kit_trace_done = False
        self.perf_probe: dict | None = None  # {"mode": "physics_off" | "render_off", "left": frames}
        self._physx_sub = None
        self.last_frame_t = 0.0
        self.last_state_t = 0.0
        self.fps_counter = [0, time.monotonic(), 0.0]
        self.randomizer: DomainRandomizer | None = None
        self.camera_base_tf: dict[str, Gf.Matrix4d] = {}

    # ==============================================================================================
    # setup
    # ==============================================================================================
    def setup(self) -> None:
        """Called once after the stage is loaded and before the timeline starts playing."""
        stage = self.stage
        log.info("physics: %d Hz (was %.0f Hz in the scene)", self.scfg.physics_hz, self._set_physics_hz(self.scfg.physics_hz))
        # -- props: find the rigid body of every prop ------------------------------------------------
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        root = stage.GetPrimAtPath(cfg.PROPS_ROOT)
        for child in root.GetChildren() if root else []:
            body = next((p for p in Usd.PrimRange(child) if p.HasAPI(UsdPhysics.RigidBodyAPI)), None)
            if body is None:
                continue
            cat_attr = child.GetAttribute("teleop:category")
            prop = Prop(child.GetName(), str(child.GetPath()), str(body.GetPath()), cat_attr.Get() if cat_attr else "")
            T_body = gf_to_np(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            rng = bbox.ComputeWorldBound(child).ComputeAlignedRange()
            c = np.array(rng.GetMidpoint())
            size = np.array(rng.GetSize())
            Rb = T_body[:3, :3] / np.linalg.norm(T_body[:3, :3], axis=0)
            prop.center_local = Rb.T @ (c - T_body[:3, 3])
            prop.bottom_offset = float(T_body[2, 3] - rng.GetMin()[2])
            prop.radius = float(0.5 * math.hypot(size[0], size[1]))
            prop.half_height = float(size[2] / 2)
            prop.default_quat = matrix_to_quat_wxyz(Rb)
            prop.size = size
            home = child.GetAttribute("teleop:homeXY").Get() if child.GetAttribute("teleop:homeXY") else None
            prop.home_xy = np.array(home if home is not None else c[:2], dtype=float)
            fixed = child.GetAttribute("teleop:fixed")
            prop.fixed = bool(fixed.Get()) if fixed and fixed.Get() is not None else False
            self.props.append(prop)
        log.info("props: %s", [p.name for p in self.props])
        if self.props:
            self.prop_view = RigidPrim([p.body for p in self.props])

        # -- robot -----------------------------------------------------------------------------------
        self.robot = Articulation(cfg.ROBOT_PRIM)
        link = stage.GetPrimAtPath(f"{cfg.ROBOT_PRIM}/{cfg.EE_LINK}")
        ee = stage.GetPrimAtPath(f"{cfg.ROBOT_PRIM}/{cfg.EE_LINK}/flange/ee_link")
        T_link = gf_to_np(UsdGeom.Xformable(link).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        T_ee = gf_to_np(UsdGeom.Xformable(ee).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        world = stage.GetPrimAtPath(cfg.WORLD)
        gt = world.GetAttribute("teleop:gripperType")
        self.gripper_type = gt.Get() if gt and gt.Get() else "virtual"
        tcp_attr = world.GetAttribute("teleop:tcpOffset")
        tcp_offset = tuple(tcp_attr.Get()) if tcp_attr and tcp_attr.Get() is not None else cfg.TOOL_TCP_OFFSET
        self.T_link_tcp = inv_tf(T_link) @ T_ee @ make_tf(np.eye(3), tcp_offset)
        log.info("gripper: %s, tcp offset %s", self.gripper_type, tcp_offset)
        wrist = stage.GetPrimAtPath(cfg.WRIST_CAMERA)
        if wrist:
            T_w = gf_to_np(UsdGeom.Xformable(wrist).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            self.T_link_wrist = inv_tf(T_link) @ T_w
        for side in ("L", "R"):
            p = stage.GetPrimAtPath(f"{cfg.TOOL_PRIM}/Fingers/Finger{side}")
            ops = UsdGeom.Xformable(p).GetOrderedXformOps() if p else []
            if ops:
                self.finger_ops.append((ops[0], 1.0 if side == "L" else -1.0))

        # -- cameras / render products ---------------------------------------------------------------
        for name, path in CAMERA_PATHS.items():
            prim = stage.GetPrimAtPath(path)
            if prim and name != "wrist":
                ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
                if ops:
                    self.camera_base_tf[name] = ops[0].Get()
        # Both the mono and the stereo render products exist; only the ones in use are rendered, so the
        # web app can switch between mono and stereo at runtime.
        self._create_stereo_rigs()
        # H.264 straight from the GPU (gpu_stream.py): the stream annotators then keep their images on the GPU
        self.gpu_stream = None
        self.annot_device = "cpu"
        if self.scfg.video_codec in ("auto", "nvenc"):
            from . import gpu_stream

            if gpu_stream.available():
                try:
                    self.gpu_stream = gpu_stream.GpuH264Stream(self.frames.publish_packet, self.scfg.video_fps)
                    self.gpu_stream.bitrate = getattr(self, "_bitrate", self.scfg.video_bitrate)
                    self.frames.on_need_key = self.gpu_stream.request_keyframe
                    self.frames.gpu_h264 = True
                    self.annot_device = "cuda"
                    log.info("video: GPU capture -> NVENC enabled")
                except Exception:  # noqa: BLE001
                    log.exception("GPU stream encoder failed to start; using the CPU path")
                    self.gpu_stream = None
        sizes = {
            "M": (self.scfg.video_width, self.scfg.video_height),
            "L": (self.scfg.stereo_width, self.scfg.stereo_height),
            "R": (self.scfg.stereo_width, self.scfg.stereo_height),
        }
        for eye, size in sizes.items():
            rp = rep.create.render_product(self._camera_path(self.camera, eye), size, name=f"Teleop_{eye}")
            annot = rep.AnnotatorRegistry.get_annotator("rgb", device=self.annot_device)
            annot.attach(rp)
            self.render_products[eye] = rp
            self.annotators[eye] = annot
            self._rp_state[eye] = True
        from .immersive import ImmersiveView

        self.immersive = ImmersiveView(self)
        for eye, rp in self.immersive.render_products.items():
            self.render_products[eye] = rp
            self.annotators[eye] = self.immersive.annotators[eye]
            self._rp_state[eye] = True
        self._apply_stream_rps(False)
        self._bitrate = self.scfg.video_bitrate
        self._bitrate_manual: int | None = None  # None: scaled with the frame size
        self._apply_bitrate()
        self._setup_avatar()

        self.randomizer = DomainRandomizer(self)
        if self.scfg.record:
            from .recorder import RecordingManager

            self.rec = RecordingManager(self, self.scfg)

    def _create_stereo_rigs(self) -> None:
        """Left/right child cameras of every camera, offset by +-IPD/2 along the camera's x (image right)."""
        half = self.ipd / 2
        for path in CAMERA_PATHS.values():
            src = UsdGeom.Camera(self.stage.GetPrimAtPath(path))
            if not src:
                continue
            for eye, dx in (("L", -half), ("R", half)):
                cam = UsdGeom.Camera.Define(self.stage, f"{path}/Stereo{eye}")
                for attr in ("focalLength", "horizontalAperture", "verticalAperture", "clippingRange"):
                    v = src.GetPrim().GetAttribute(attr).Get()
                    if v is not None:
                        cam.GetPrim().CreateAttribute(attr, src.GetPrim().GetAttribute(attr).GetTypeName()).Set(v)
                xf = UsdGeom.Xformable(cam)
                xf.ClearXformOpOrder()
                op = xf.AddTranslateOp()
                op.Set(Gf.Vec3d(dx, 0, 0))
                self._rig_ops.append((op, -1.0 if eye == "L" else 1.0))

    # ==============================================================================================
    # operator avatar (VRM)
    # ==============================================================================================
    def _setup_avatar(self) -> None:
        import os

        src, source = cfg.avatar_file(self.scfg.avatar)
        if src is None:
            (log.warning if "not found" in source else log.info)("avatar: %s", source)
            return
        log.info("avatar: %s (%s)", src, source)
        try:
            from . import vrm
            from .avatar import Avatar

            stem = os.path.splitext(os.path.basename(src))[0]
            usd = vrm.convert(src, os.path.join(cfg.AVATAR_CONVERTED_DIR, stem.replace(" ", "_")))
            prim = self.stage.DefinePrim(cfg.AVATAR_PRIM, "Xform")
            prim.GetReferences().AddReference(str(usd).replace("\\", "/"))
            meta = str(usd)[: -len(".usdc")] + ".avatar.json"
            self.avatar = Avatar(self.stage, meta)
            self._apply_avatar_clip()
            self._create_avatar_bodies()
        except Exception:  # noqa: BLE001 - the avatar is optional
            log.exception("could not load the avatar %s", src)
            self.avatar = None

    def _create_avatar_bodies(self) -> None:
        """Kinematic capsules / spheres that follow the avatar's bones (they push props and block the robot).

        Created before the simulation starts (adding rigid bodies later would resync PhysX). While collisions
        are off the bodies are parked far away, like inactive props."""
        from pxr import PhysxSchema

        av = self.avatar
        if av is None or not av.bodies:
            return
        root = UsdGeom.Scope.Define(self.stage, cfg.AVATAR_BODIES_ROOT)
        paths, shapes = [], []
        for k, b in enumerate(av.bodies):
            path = f"{root.GetPath()}/{b['name']}"
            xf = UsdGeom.Xform.Define(self.stage, path)
            xf.AddTranslateOp().Set(Gf.Vec3d(*self._avatar_park_pos(k)))
            xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
            prim = xf.GetPrim()
            rb = UsdPhysics.RigidBodyAPI.Apply(prim)
            rb.CreateKinematicEnabledAttr(True)
            px = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            px.CreateSleepThresholdAttr(0.0)
            if b["kind"] == "capsule":
                g = UsdGeom.Capsule.Define(self.stage, f"{path}/Shape")
                g.CreateAxisAttr(UsdGeom.Tokens.z)
                g.CreateRadiusAttr(0.05)
                g.CreateHeightAttr(0.1)
            else:
                g = UsdGeom.Sphere.Define(self.stage, f"{path}/Shape")
                g.CreateRadiusAttr(0.05)
            UsdPhysics.CollisionAPI.Apply(g.GetPrim())
            g.CreatePurposeAttr(UsdGeom.Tokens.guide)  # never rendered (cameras / RTX LiDAR see the skinned mesh)
            paths.append(path)
            shapes.append(g)
        self.avatar_body_view = RigidPrim(paths)
        self._avatar_body_state["shapes"] = shapes
        log.info("avatar: %d collision bodies", len(paths))

    @staticmethod
    def _avatar_park_pos(k: int) -> tuple[float, float, float]:
        return (cfg.AVATAR_PARKING[0], cfg.AVATAR_PARKING[1] + 0.6 * k, cfg.AVATAR_PARKING[2])

    def _avatar_collision_active(self) -> bool:
        return self.avatar_settings.get("collision", "on") == "on"

    def _drive_avatar_bodies(self, shown: bool) -> None:
        av, view = self.avatar, self.avatar_body_view
        if av is None or view is None:
            return
        st = self._avatar_body_state
        poses = av.body_poses() if shown and self._avatar_collision_active() else None
        active = poses is not None
        n = len(av.bodies)
        parked_now = set()
        if active:
            pos, quat, rad, half = poses
            # hands that can grab are "ghost" hands: their colliders would push away what they reach for
            for side in self._grab_sides():
                for k, b in enumerate(av.bodies):
                    if b["name"] in (f"{side}Hand", f"{side}LowerArm"):
                        parked_now.add(k)
            for k in parked_now:
                pos[k] = self._avatar_park_pos(k)
                quat[k] = (0.0, 0.0, 0.0, 1.0)
            # resize the shapes only when the avatar scale changed noticeably (USD edits rebuild the shapes)
            if abs(av.scale - st["scale"]) > 0.03 * max(st["scale"], 1e-3):
                for k, g in enumerate(st["shapes"]):
                    g.GetRadiusAttr().Set(float(rad[k]))
                    if isinstance(g, UsdGeom.Capsule):
                        g.GetHeightAttr().Set(float(2 * half[k]))
                st["scale"] = av.scale
        else:
            pos = np.array([self._avatar_park_pos(k) for k in range(n)])
            quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
        try:
            if active != st["active"]:
                # jump without sweeping through whatever is in between
                view.set_world_poses(pos, quat[:, [3, 0, 1, 2]])
                st["active"] = active
            jump = sorted(parked_now ^ st.get("parked", set())) if active else []
            if jump:
                view.set_world_poses(pos[jump], quat[jump][:, [3, 0, 1, 2]], indices=jump)
            st["parked"] = parked_now
            import warp as wp

            rbv = view._physics_rigid_body_view  # noqa: SLF001 - kinematic targets are not wrapped yet
            dev = st.get("device")
            if dev is None:
                dev = st["device"] = rbv.get_transforms().device
            tgt = wp.array(np.hstack([pos, quat]).astype(np.float32), dtype=wp.float32, device=dev)
            idx = st.get("idx")
            if idx is None or len(idx) != n:
                idx = st["idx"] = wp.array(np.arange(n, dtype=np.int32), dtype=wp.int32, device=dev)
            rbv.set_kinematic_targets(tgt, idx)
        except Exception:  # noqa: BLE001 - never take the loop down for the colliders
            log.exception("avatar colliders disabled")
            self.avatar_body_view = None

    def _apply_avatar_clip(self) -> None:
        """First-person cameras clip the inside of the avatar's head while it is shown."""
        near = cfg.AVATAR_NEAR_CLIP if self.avatar is not None and self.avatar.visible else 0.02
        paths = [f"{cfg.CAMERAS_ROOT}/Head/Stereo{e}" for e in "LR"]
        for p in paths:
            cam = UsdGeom.Camera(self.stage.GetPrimAtPath(p))
            if cam:
                far = (cam.GetClippingRangeAttr().Get() or Gf.Vec2f(0.01, 100.0))[1]
                cam.GetClippingRangeAttr().Set(Gf.Vec2f(near, far))

    def set_avatar(self, payload: dict) -> dict:
        if self.avatar is None:
            raise RuntimeError(
                f"no avatar loaded (set {cfg.AVATAR_ENV} to a .vrm file or put one at assets/avatars/avatar.vrm, then restart the server)"
            )
        for k in ("visible", "hideWhileRecording"):
            if k in payload:
                self.avatar_settings[k] = bool(payload[k])
        if "collision" in payload:
            mode = str(payload["collision"])
            if mode == "auto":
                mode = "on"
            if mode not in ("on", "off"):
                raise ValueError("collision must be on or off")
            self.avatar_settings["collision"] = mode
        if "autoScale" in payload:
            self.avatar.auto_scale = bool(payload["autoScale"])
            if not self.avatar.auto_scale:
                self.avatar.scale = float(payload.get("scale", 1.0))
        return {**self.avatar.status(), **self.avatar_settings}

    def _operator_camera_pose(self) -> tuple[np.ndarray, float]:
        cam = self.stage.GetPrimAtPath(CAMERA_PATHS["operator"])
        T = gf_to_np(UsdGeom.Xformable(cam).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        fwd = -T[:3, 2]
        return T[:3, 3], math.atan2(fwd[1], fwd[0])

    def _update_avatar(self, dt: float) -> None:
        """The avatar exists only in the free viewpoint, where the sim space is the user's own space."""
        av = self.avatar
        if av is None:
            return
        recording = self.rec is not None and self.rec.state != "idle"
        imm = self.view_mode == "immersive" and self.immersive is not None and self.immersive.head_xr is not None
        show = (
            imm
            and self.avatar_settings["visible"]
            and not (self.avatar_settings["hideWhileRecording"] and recording)
        )
        if show != av.visible:
            av.set_visible(show)
            self._apply_avatar_clip()
        if not show:
            self._drive_avatar_bodies(False)
            return
        from .avatar import hands_from_sample

        _owner, sample = self.teleop.latest()
        to_world = self.immersive.head_world
        av.update(to_world(*self.immersive.head_xr), hands_from_sample(sample, to_world), dt)
        t_b = time.perf_counter()
        self._drive_avatar_bodies(True)
        PERF.add("av.bodies", (time.perf_counter() - t_b) * 1e3)

    # ==============================================================================================
    # stream resolution (render products are recreated at the new size)
    # ==============================================================================================
    def resolution_status(self) -> dict:
        sc = self.scfg
        return {
            "screen": [sc.video_width, sc.video_height],
            "stereo": [sc.stereo_width, sc.stereo_height],
            "immersive": int(self.immersive.size[0]) if self.immersive else sc.immersive_size,
            "bitrate": int(self._bitrate),
            "bitrateAuto": self._bitrate_manual is None,
            "dlss": self.render_quality(),
        }

    # DLSS: the RTX renderer draws at a lower internal resolution and upscales (Isaac Sim's default: Performance)
    DLSS_MODES = {"performance": 0, "balanced": 1, "quality": 2, "auto": 3}

    def render_quality(self) -> str:
        import carb.settings

        s = carb.settings.get_settings()
        if s.get("/rtx/post/aa/op") != 3:
            return "native"  # no DLSS: full resolution (TAA)
        mode = s.get("/rtx/post/dlss/execMode")
        return next((k for k, v in self.DLSS_MODES.items() if v == mode), str(mode))

    def set_render_quality(self, mode: str) -> str:
        import carb.settings

        s = carb.settings.get_settings()
        if mode == "native":
            s.set("/rtx/post/aa/op", 1)
        elif mode in self.DLSS_MODES:
            s.set("/rtx/post/aa/op", 3)
            s.set("/rtx/post/dlss/execMode", self.DLSS_MODES[mode])
        else:
            raise ValueError("dlss must be performance / balanced / quality / auto / native")
        log.info("render: %s (aa op %s, dlss mode %s)", mode, s.get("/rtx/post/aa/op"), s.get("/rtx/post/dlss/execMode"))
        return self.render_quality()

    def set_resolution(self, payload: dict) -> dict:
        sc = self.scfg

        def wh(v, lo, hi):
            w, h = int(v[0]), int(v[1])
            if not (lo[0] <= w <= hi[0] and lo[1] <= h <= hi[1]) or w % 2 or h % 2:
                raise ValueError(f"resolution {w}x{h} out of range ({lo[0]}x{lo[1]} .. {hi[0]}x{hi[1]}, even)")
            return w, h

        if "screen" in payload:
            sc.video_width, sc.video_height = wh(payload["screen"], (320, 180), (2560, 1440))
            self._recreate_rp("M", (sc.video_width, sc.video_height))
        if "stereo" in payload:
            sc.stereo_width, sc.stereo_height = wh(payload["stereo"], (320, 180), (1920, 1080))
            for eye in ("L", "R"):
                self._recreate_rp(eye, (sc.stereo_width, sc.stereo_height))
        if "bitrate" in payload:
            b = payload["bitrate"]
            if b in (None, "auto"):
                self._bitrate_manual = None
            else:
                b = int(b)
                if not 1_000_000 <= b <= 100_000_000:
                    raise ValueError("bitrate must be 1..100 Mbit/s")
                self._bitrate_manual = b
        if "dlss" in payload:
            self.set_render_quality(str(payload["dlss"]))
        if "immersive" in payload and self.immersive is not None:
            n = int(payload["immersive"])
            # 2048 per eye: the side-by-side stream is 4096 wide, the H.264 (NVENC) limit
            if not 1024 <= n <= 2048 or n % 2:
                raise ValueError("immersive size must be 1024..2048 (even)")
            sc.immersive_size = n
            self.immersive.size = (n, n)
            for eye in ("HL", "HR"):
                self._recreate_rp(eye, (n, n))
        self._apply_bitrate()
        self._stream_warmup = 3
        return self.resolution_status()

    def _recreate_rp(self, eye: str, size: tuple[int, int]) -> None:
        if eye in ("HL", "HR"):
            cam = f"{cfg.CAMERAS_ROOT}/Head/Stereo{eye[1]}"
        else:
            cam = self._camera_path(self.camera, eye)
        old = self.render_products.get(eye)
        annot = self.annotators[eye]
        try:
            annot.detach()
        except Exception:  # noqa: BLE001
            pass
        self._rp_serial = getattr(self, "_rp_serial", 0) + 1
        rp = rep.create.render_product(cam, size, name=f"Stream_{eye}_{self._rp_serial}")
        annot.attach(rp)
        self.render_products[eye] = rp
        if eye in ("HL", "HR") and self.immersive is not None:
            self.immersive.render_products[eye] = rp
        self._rp_state[eye] = True  # new render products start enabled; _apply_stream_rps disables if idle
        if old is not None:
            try:
                old.destroy()
            except Exception:  # noqa: BLE001
                log.warning("could not destroy the old %s render product", eye)
        log.info("stream %s: %dx%d", eye, size[0], size[1])

    def _apply_bitrate(self) -> None:
        """Scale the encoder bitrate with the largest streamed frame (base: --bitrate at 1280x720)."""
        from .webrtc import configure_bitrate

        sc = self.scfg
        px = max(sc.video_width * sc.video_height, 2 * sc.stereo_width * sc.stereo_height, 2 * sc.immersive_size**2)
        self._bitrate = int(min(60_000_000, max(sc.video_bitrate, sc.video_bitrate * px / (1280 * 720))))
        if getattr(self, "_bitrate_manual", None) is not None:
            self._bitrate = self._bitrate_manual  # set from the web app (upper limit of the VBR encoder)
        configure_bitrate(self._bitrate)
        if getattr(self, "gpu_stream", None) is not None:
            self.gpu_stream.bitrate = self._bitrate

    def set_stereo(self, enabled: bool | None = None, ipd: float | None = None) -> dict:
        if ipd is not None:
            ipd = float(ipd)
            if not 0.045 <= ipd <= 0.085:
                raise ValueError("ipd must be between 45 and 85 mm")
            self.ipd = ipd
            for op, sign in self._rig_ops:
                op.Set(Gf.Vec3d(sign * ipd / 2, 0, 0))
            if self.immersive is not None:
                self.immersive.set_ipd(ipd)
        if enabled is not None and bool(enabled) != self.stereo:
            self.stereo = bool(enabled)
            self._stream_warmup = 3
        return {"stereo": self.stereo, "ipd": round(self.ipd, 4)}

    def _active_eyes(self) -> list[str]:
        if self.view_mode == "immersive":
            return ["HL", "HR"]
        return ["L", "R"] if self.stereo else ["M"]

    def set_view(self, mode: str | None = None, recenter: bool = False) -> dict:
        if mode is not None:
            if mode not in ("screen", "immersive"):
                raise ValueError(f"unknown view mode: {mode}")
            if mode != self.view_mode:
                self.view_mode = mode
                self.clutch = False  # the hand/arm mapping changes with the view
                self._stream_warmup = 3
        if recenter and self.immersive is not None:
            self.immersive.recenter()
        return {"view": self.view_mode, **(self.immersive.status() if self.immersive else {})}

    def _apply_viewport(self, idle: bool) -> None:
        """Render Isaac Sim's own viewport only when it is useful: it costs as much as one stream camera."""
        on = {"on": True, "off": False}.get(self.viewport_mode, idle)
        if on == self._viewport_on:
            return
        self._viewport_on = on
        try:
            from omni.kit.viewport.utility import get_active_viewport

            vp = get_active_viewport()
            if vp is not None:
                vp.updates_enabled = on
                log.info("Isaac Sim viewport %s", "rendering" if on else "paused (the stream is being watched)"
                         if self.viewport_mode == "auto" else "paused")
        except Exception as exc:  # noqa: BLE001 - only an optimization
            log.warning("could not toggle the viewport: %s", exc)

    def _apply_stream_rps(self, streaming: bool) -> None:
        """Render only the render products that are currently streamed."""
        active = set(self._active_eyes()) if streaming else set()
        for eye, rp in self.render_products.items():
            on = eye in active
            if self._rp_state.get(eye) != on:
                rp.hydra_texture.set_updates_enabled(on)
                self._rp_state[eye] = on
                if on:
                    self._stream_warmup = 3  # the first frames after re-enabling may still be stale

    def _camera_path(self, name: str, eye: str) -> str:
        base = CAMERA_PATHS[name]
        return base if eye == "M" else f"{base}/Stereo{eye}"

    def set_camera(self, name: str) -> None:
        if name not in CAMERA_PATHS:
            raise ValueError(f"unknown camera: {name}")
        self.camera = name
        # Switch through the HydraTexture: retargeting the render product's USD "camera" relationship
        # leaves the renderer with a blown-out (white) image.
        for eye in ("M", "L", "R"):
            self.render_products[eye].hydra_texture.set_camera_path(self._camera_path(name, eye))

    def camera_hfov_deg(self) -> float:
        cam = UsdGeom.Camera(self.stage.GetPrimAtPath(CAMERA_PATHS[self.camera]))
        f = cam.GetFocalLengthAttr().Get() or 14.0
        ap = cam.GetHorizontalApertureAttr().Get() or 20.955
        return math.degrees(2 * math.atan(ap / (2 * f)))

    def _on_physics_ready(self) -> None:
        try:
            import omni.physx

            # physics substeps per rendered frame (--physics-hz substeps over the wall-clock frame time) and the time of
            # each step, measured from the first pre-step callback to the last post-step callback
            px = omni.physx.get_physx_interface()
            t_pre = [0.0]

            def pre(_dt):
                t_pre[0] = time.perf_counter()

            def post(_dt):
                PERF.count("physics.steps")
                PERF.add("physics.step", (time.perf_counter() - t_pre[0]) * 1e3)

            self._physx_sub = (
                px.subscribe_physics_on_step_events(pre, True, -1000000),
                px.subscribe_physics_on_step_events(post, False, 1000000),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("physics step counter unavailable: %s", exc)
        robot = self.robot
        names = list(robot.dof_names)
        log.info("robot dofs: %s", names)
        self.arm_idx = np.array([names.index(f"J{i}") for i in range(1, 7)], dtype=np.int32)
        if self.gripper_type == "robotiq_2f85":
            if cfg.GRIPPER_JOINT not in names:
                log.error("finger_joint is not part of the robot articulation - using the virtual gripper")
                self.gripper_type = "virtual"
            else:
                self.grip_idx = np.array([names.index(cfg.GRIPPER_JOINT)], dtype=np.int32)
        self.link_index = int(robot.get_link_indices(cfg.EE_LINK).numpy()[0])
        lim = robot.get_dof_limits()
        lo, hi = (np.asarray(x.numpy())[0] for x in lim)
        self.limits = np.stack([lo, hi], axis=1)[self.arm_idx]
        # Gravity off on the robot links (an industrial arm holds position; keeps IK tracking crisp)
        robot.set_link_enabled_gravities(False)
        stiff = np.array([20000, 20000, 15000, 8000, 8000, 5000], dtype=float)
        robot.set_dof_gains(stiff, stiff * 0.08, dof_indices=self.arm_idx)
        # Torque limits: when the gripper is pushed into an object or the table the arm stalls instead of
        # crushing through it (the stiff position drives would otherwise squeeze objects out at high speed).
        robot.set_dof_max_efforts(np.array(cfg.ARM_MAX_EFFORT, dtype=float), dof_indices=self.arm_idx)
        if self.prop_view is not None:
            log.info("prop rigid body view valid: %s", self.prop_view.is_physics_tensor_entity_valid())
            # Teleporting a sleeping body does not publish its new pose to the renderer -> never sleep
            self.prop_view.set_sleep_thresholds(np.zeros((len(self.props), 1)))
            pos, quat = (np.asarray(x.numpy()) for x in self.prop_view.get_world_poses())
            log.info("prop poses (first 3): %s", np.round(pos[:3], 3).tolist())
            self.default_poses = (pos.copy(), quat.copy())
            masses = np.asarray(self.prop_view.get_masses().numpy()).reshape(-1)
            for p, m in zip(self.props, masses):
                p.default_mass = float(m)

        self.ready = True
        self.reset()

    def _log_finger_geometry(self) -> None:
        """Where the finger pads really are relative to the TCP frame (diagnostics / TCP calibration)."""
        try:
            names = list(self.robot.link_names)
            data = np.asarray(self.robot._physics_articulation_view.get_link_transforms().numpy()).reshape(-1, 7)  # noqa: SLF001
            T_tcp = self.tcp_transform()
            inv = inv_tf(T_tcp)
            for n in names:
                if "finger" in n or "knuckle" in n:
                    p = inv @ np.append(data[names.index(n)][:3], 1.0)
                    log.info("link %-28s in TCP frame: %s", n, np.round(p[:3], 4).tolist())
        except Exception as exc:  # noqa: BLE001
            log.warning("finger geometry: %s", exc)

    # ==============================================================================================
    # per-frame
    # ==============================================================================================
    def pre_update(self, dt: float) -> None:
        if not self.ready:
            if self.timeline.is_playing() and self.robot is not None:
                try:
                    if self.robot.is_physics_tensor_entity_valid():
                        self._on_physics_ready()
                except AssertionError:
                    pass
            self._run_commands(allow_sim=False)
            return
        self._run_commands(allow_sim=True)
        self._tick_perf_probe()
        if self.randomizer:
            self.randomizer.tick(dt)
        T_tcp = self.tcp_transform()
        if self.prev_tcp is not None and dt > 0:
            self.tcp_vel = 0.7 * self.tcp_vel + 0.3 * (T_tcp[:3, 3] - self.prev_tcp) / dt
        self.prev_tcp = T_tcp[:3, 3].copy()
        if self.resync_target or self.target_T is None:
            self.target_T = T_tcp.copy()
            self.resync_target = False
        if self.view_mode == "immersive" and self.immersive is not None:
            _owner, sample = self.teleop.latest()
            self.immersive.update(sample, dt, self.settings.get("hand", "right"))
        t_av = time.perf_counter()
        self._update_avatar(dt)
        if self.avatar is not None and self.avatar.visible:
            PERF.add("sim.avatar", (time.perf_counter() - t_av) * 1e3)
        if not (self.view_mode == "immersive" and self.settings.get("handMode") == "grab"):
            self._process_teleop(T_tcp)
        elif self.clutch:
            self.clutch = False
        self._update_hand_grab(dt)
        self._solve_ik(T_tcp, dt)
        self._update_gripper(T_tcp)
        if self.rec:
            self.rec.on_pre_update()

    def post_update(self, dt: float) -> None:
        now = time.monotonic()
        c = self.fps_counter
        c[0] += 1
        if now - c[1] >= 1.0:
            c[2] = c[0] / (now - c[1])
            c[0], c[1] = 0, now
        want = self.frames.has_subscribers or self.frames.snapshot_wanted
        if self.perf_probe and self.perf_probe["mode"] == "render_off":
            want = False
        # stop rendering the stream cameras while nobody is watching
        self._apply_stream_rps(want)
        self._apply_viewport(not self.frames.has_subscribers)
        if self._stream_warmup > 0:
            self._stream_warmup -= 1
        elif want and now - self.last_frame_t >= 1.0 / self.scfg.video_fps * 0.95:
            meta = {"view": self.view_mode, "spub": time.time() * 1000}  # wall clock at publish
            rp = self.immersive.rendered_pose() if self.view_mode == "immersive" and self.immersive else None
            applied = None
            if rp is not None:
                # XR-space head pose this image was rendered with -> the web app shows it world-locked
                head, ts, rx, applied = rp
                meta = {"view": "immersive", "head": head, "ipd": self.ipd, "fov": self.immersive.fov,
                        "ts": ts, "srx": rx, "spub": time.time() * 1000}
            sent = False
            if self.gpu_stream is not None and self.frames.has_kind("h264"):
                sent = self._submit_gpu(meta)
            raw_peers = self.frames.has_kind("raw")
            if raw_peers or self.frames.snapshot_wanted:
                frame = self._grab_frame()
                if frame is not None:
                    if raw_peers:
                        self.frames.publish(frame, meta)
                    else:
                        self.frames.set_latest(frame)  # /api/snapshot only
                    self.frames.snapshot_wanted = False
                    sent = sent or raw_peers
            if sent:
                self.last_frame_t = now
                PERF.count("frames.published")
                if applied is not None:
                    PERF.add("pose.to_publish", (time.perf_counter() - applied) * 1e3)
        if self.rec and self.ready:
            self.rec.on_post_update()
        if now - self.last_state_t >= 1.0 / self.scfg.state_hz:
            self.last_state_t = now
            self.state.set(self._snapshot())

    def _submit_gpu(self, meta: dict) -> bool:
        """Hand the eyes (Warp arrays on the GPU) to the shared NVENC encoder."""
        t0 = time.perf_counter()
        eyes = []
        for eye in self._active_eyes():
            data = self.annotators[eye].get_data(device="cuda")
            if data is None or not hasattr(data, "shape") or len(data.shape) != 3 or data.shape[0] == 0:
                return False
            eyes.append(data)
        PERF.add("grab.read", (time.perf_counter() - t0) * 1e3)
        return self.gpu_stream.submit(eyes, meta)

    def _grab_frame(self) -> np.ndarray | None:
        """The streamed image (eyes side by side) as RGBA uint8. The annotator returns a view of its own buffer:
        one row-wise copy per eye into a new array is the only work done here; dropping alpha / converting to YUV
        happens in the encoder's thread (aiortc / PyAV)."""
        datas = []
        t0 = time.perf_counter()
        for eye in self._active_eyes():
            data = self.annotators[eye].get_data(device="cpu")
            if data is None or not hasattr(data, "shape") or data.size == 0 or data.ndim != 3:
                return None
            datas.append(data)
        t1 = time.perf_counter()
        h, c = datas[0].shape[0], datas[0].shape[2]
        if any(d.shape[0] != h or d.shape[2] != c for d in datas) or c not in (3, 4):
            return None  # eyes mid-resize
        img = np.empty((h, sum(d.shape[1] for d in datas), c), dtype=np.uint8)
        x = 0
        for d in datas:
            w = d.shape[1]
            img[:, x:x + w] = d
            x += w
        PERF.add("grab.read", (t1 - t0) * 1e3)
        PERF.add("grab.concat", (time.perf_counter() - t1) * 1e3)
        return img

    # ==============================================================================================
    # robot kinematics
    # ==============================================================================================
    def _link_tf(self) -> np.ndarray:
        view = self.robot._physics_articulation_view  # noqa: SLF001 - link poses are not wrapped yet
        data = np.asarray(view.get_link_transforms().numpy()).reshape(-1, 7)
        d = data[self.link_index]
        return make_tf(quat_xyzw_to_matrix(d[3:7]), d[:3])

    def tcp_transform(self) -> np.ndarray:
        return self._link_tf() @ self.T_link_tcp

    def all_dof_positions(self) -> np.ndarray:
        return np.asarray(self.robot.get_dof_positions().numpy()).reshape(-1)

    def joint_positions(self) -> np.ndarray:
        """Arm joints J1..J6 (rad)."""
        return self.all_dof_positions()[self.arm_idx]

    def _set_arm_targets(self, q: np.ndarray) -> None:
        self.robot.set_dof_position_targets(q, dof_indices=self.arm_idx)

    def _solve_ik(self, T_tcp: np.ndarray, dt: float) -> None:
        q = self.joint_positions()
        if self.q_cmd is None:
            self.q_cmd = q.copy()
        T_goal = self.target_T
        e_p = T_goal[:3, 3] - T_tcp[:3, 3]
        e_r = rotvec_from_matrix(T_goal[:3, :3] @ T_tcp[:3, :3].T)
        n_p, n_r = np.linalg.norm(e_p), np.linalg.norm(e_r)
        if n_p < 5e-4 and n_r < 2e-3:
            self._set_arm_targets(self.q_cmd)
            return
        e_p = e_p * min(1.0, 0.06 / max(n_p, 1e-9))
        e_r = e_r * min(1.0, 0.12 / max(n_r, 1e-9))
        J = np.asarray(self.robot.get_jacobian_matrices().numpy())[0][self.link_index - 1][:, self.arm_idx]
        r = T_tcp[:3, 3] - self._link_tf()[:3, 3]
        Jv = J[:3] - skew(r) @ J[3:]
        Jw = J[3:]
        Jt = np.vstack([Jv, Jw * 0.5])  # weight rotation lower than translation
        err = np.concatenate([e_p, e_r * 0.5])
        lam = 0.03
        dq = Jt.T @ np.linalg.solve(Jt @ Jt.T + (lam**2) * np.eye(6), err)
        max_step = 3.0 * max(dt, 1 / 120)  # rad per frame (~3 rad/s)
        dq = np.clip(dq, -max_step, max_step)
        q_new = np.clip(self.q_cmd + dq, q - 0.25, q + 0.25)
        if self.limits is not None:
            q_new = np.clip(q_new, self.limits[:, 0] + 1e-3, self.limits[:, 1] - 1e-3)
        self.q_cmd = q_new
        self._set_arm_targets(self.q_cmd)

    # ==============================================================================================
    # teleoperation
    # ==============================================================================================
    def _camera_world_tf(self, name: str | None = None) -> np.ndarray:
        """World pose of a camera (default: the one shown). USD camera convention: looks down -Z, +Y up."""
        name = name or self.camera
        if name == "wrist":
            return self._link_tf() @ self.T_link_wrist  # moves with the arm (USD is not updated by physics)
        prim = self.stage.GetPrimAtPath(CAMERA_PATHS[name])
        return gf_to_np(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))

    def _xr_to_world(self) -> np.ndarray:
        """3x3 matrix mapping WebXR vectors (x right, y up, z towards the user) to world vectors.

        The video screen sits straight ahead of the user, so the hand must move the gripper the way the
        image moves: hand right -> image right, hand forward -> into the image.
        * wrist camera: fully view aligned (forward = viewing direction, up = image up)
        * fixed cameras: yaw only (up stays world-up so lowering the hand always lowers the gripper);
          for a camera looking straight down, "forward" is the image's up direction.
        """
        if self.view_mode == "immersive" and self.immersive is not None:
            return self.immersive.xr_to_world()
        mapping = self.settings.get("mapping", "camera")
        up = np.array([0, 0, 1.0])
        if mapping == "robot":
            fwd = np.array([1.0, 0.0, 0.0])  # the robot's forward (+X of its base)
            return np.stack([np.cross(fwd, up), up, -fwd], axis=1)
        name = "operator" if mapping == "operator" else self.camera
        T = self._camera_world_tf(name)
        R = T[:3, :3] / np.linalg.norm(T[:3, :3], axis=0)
        cam_right, cam_up, cam_back = R[:, 0], R[:, 1], R[:, 2]
        if name == "wrist":
            return np.stack([cam_right, cam_up, cam_back], axis=1)
        fwd = -cam_back.copy()
        fwd[2] = 0
        if np.linalg.norm(fwd) < 0.3:  # looking (almost) straight down
            fwd = cam_up.copy()
            fwd[2] = 0
        fwd /= np.linalg.norm(fwd) or 1.0
        up = np.array([0, 0, 1.0])
        right = np.cross(fwd, up)
        return np.stack([right, up, -fwd], axis=1)

    def _process_teleop(self, T_tcp: np.ndarray) -> None:
        _owner, sample = self.teleop.latest()
        hand = None
        if sample:
            hand = sample.get(self.settings.get("hand", "right")) or None
        if not hand or "p" not in hand or "q" not in hand:
            if self.clutch:
                self.clutch = False
            return
        p = self.pos_filter(np.asarray(hand["p"], dtype=float), float(sample.get("ts", 0.0)) / 1000.0,
                            float(self.settings.get("smoothing", 0.5)))
        R = quat_xyzw_to_matrix(hand["q"])
        grip = float(hand.get("grip", 0.0))
        trigger = float(hand.get("trigger", 0.0))

        if not self.clutch and grip > GRIP_ON:
            self.clutch = True
            self.clutch_ctrl0 = (p.copy(), R.copy())
            self.clutch_target0 = self.target_T.copy()
            self.clutch_M = self._xr_to_world()
        elif self.clutch and grip < GRIP_OFF:
            self.clutch = False
            # Hold the arm where it currently is (avoid drifting towards an unreachable target)
            self.target_T = T_tcp.copy() if np.linalg.norm(self.target_T[:3, 3] - T_tcp[:3, 3]) > 0.05 else self.target_T

        if self.clutch:
            p0, R0 = self.clutch_ctrl0
            M = self.clutch_M
            scale = float(self.settings.get("positionScale", 1.0))
            T = self.clutch_target0.copy()
            T[:3, 3] = self.clutch_target0[:3, 3] + scale * (M @ (p - p0))
            mode = self.settings.get("rotationMode", "yaw")
            if mode != "off":
                dR = M @ (R @ R0.T) @ M.T
                if mode == "yaw":
                    dR = _twist_about_z(dR)
                T[:3, :3] = dR @ self.clutch_target0[:3, :3]
            T[:3, 3] = self._clamp_workspace(T[:3, 3])
            self.target_T = T

        # trigger drives the gripper while held (with hysteresis). A UI "close" command is only
        # overridden once the trigger has actually been pressed.
        if trigger > GRIP_ON:
            self._trigger_was_down = True
            self.gripper_cmd_closed = True
        elif trigger < GRIP_OFF and self.gripper_cmd_closed and self._trigger_was_down:
            self.gripper_cmd_closed = False
            self._trigger_was_down = False

    # ==============================================================================================
    # grabbing with your own hands (free viewpoint)
    # ==============================================================================================
    def _grab_sides(self) -> list[str]:
        if self.view_mode != "immersive" or self.immersive is None:
            return []
        if self.settings.get("handMode") == "grab":
            return ["left", "right"]
        return ["left" if self.settings.get("hand", "right") == "right" else "right"]

    def _update_hand_grab(self, dt: float) -> None:
        if self.prop_view is None:
            return
        sides = self._grab_sides()
        _owner, sample = self.teleop.latest()
        now = time.monotonic()
        for side in ("left", "right"):
            hold = self._hand_hold.get(side)
            h = (sample or {}).get(side) if side in sides else None
            if not h or "p" not in h or "q" not in h:
                if hold is not None:
                    self._hand_drop(side, np.zeros(3))
                self._hand_prev.pop(side, None)
                continue
            T = self.immersive.head_world(h["p"], h["q"])
            prev = self._hand_prev.get(side)
            v = np.zeros(3)
            if prev is not None and now - prev[1] > 1e-3:
                v = (T[:3, 3] - prev[0]) / (now - prev[1])
                v = 0.5 * v + 0.5 * self._hand_hold_v.get(side, v)
            self._hand_hold_v[side] = v
            self._hand_prev[side] = (T[:3, 3].copy(), now)
            grip = float(h.get("grip", 0.0) or 0.0)
            if hold is None:
                if grip > GRIP_ON and not self._hand_grip_down.get(side, False):
                    self._hand_try_grab(side, T)
                self._hand_grip_down[side] = grip > GRIP_ON
                continue
            if grip < GRIP_OFF:
                self._hand_grip_down[side] = False
                self._hand_drop(side, v)
                continue
            i = self.props.index(hold["prop"])
            T_obj = T @ hold["offset"]
            self.prop_view.set_world_poses(
                T_obj[:3, 3].reshape(1, 3), matrix_to_quat_wxyz(T_obj[:3, :3]).reshape(1, 4), indices=[i]
            )
            self.prop_view.set_velocities(np.clip(v, -3, 3).reshape(1, 3), np.zeros((1, 3)), indices=[i])

    def _hand_try_grab(self, side: str, T_hand: np.ndarray) -> None:
        held_by_other = {hh["prop"].name for hh in self._hand_hold.values() if hh}
        pos, quat = self._prop_poses()
        p_hand = T_hand[:3, 3]
        best, best_d = None, 1e9
        for i, prop in enumerate(self.props):
            if not self.is_active(prop) or prop.fixed or prop.name in held_by_other:
                continue
            center = pos[i] + quat_wxyz_to_matrix(quat[i]) @ prop.center_local
            d = float(np.linalg.norm(center - p_hand)) - min(prop.radius, 0.06)
            if d < cfg.HAND_GRAB_DISTANCE and d < best_d:
                best, best_d = i, d
        if best is None:
            return
        prop = self.props[best]
        if self._assist and self._assist.get("prop") is prop:
            self._assist = None  # taken out of the robot's gripper
            self.held = None
        T_obj = make_tf(quat_wxyz_to_matrix(quat[best]), pos[best])
        self._hand_hold[side] = {"prop": prop, "offset": inv_tf(T_hand) @ T_obj}
        self.prop_view.set_enabled_gravities(np.array([[False]]), indices=[best])
        log.info("hand grab: %s hand took %s", side, prop.name)

    def _hand_drop(self, side: str, v: np.ndarray) -> None:
        hold = self._hand_hold.get(side)
        self._hand_hold[side] = None
        if hold is None or self.prop_view is None:
            return
        i = self.props.index(hold["prop"])
        self.prop_view.set_enabled_gravities(np.array([[True]]), indices=[i])
        self.prop_view.set_velocities(np.clip(v, -3, 3).reshape(1, 3), np.zeros((1, 3)), indices=[i])
        self._hand_release_t[side] = time.monotonic()
        log.info("hand grab: %s hand released %s", side, hold["prop"].name)

    def _hand_drop_all(self) -> None:
        for side in ("left", "right"):
            if self._hand_hold.get(side) is not None:
                self._hand_drop(side, np.zeros(3))

    def _input_info(self) -> dict:
        info = self.teleop.info()
        sample = info.pop("sample", None) or {}
        hand = sample.get(self.settings.get("hand", "right")) or {}
        return {
            "age": info.get("age"),
            "hand": bool(hand),
            "grip": round(float(hand.get("grip", 0.0)), 2) if hand else None,
            "trigger": round(float(hand.get("trigger", 0.0)), 2) if hand else None,
        }

    def _clamp_workspace(self, p: np.ndarray) -> np.ndarray:
        base = np.array(self.stage.GetPrimAtPath(cfg.WORLD).GetAttribute("teleop:robotBase").Get() or (0, 0, 0.85))
        rel = p - base
        reach = 1.30
        n = np.linalg.norm(rel)
        if n > reach:
            rel *= reach / n
        out = base + rel
        # fingertips are ~2.5 cm below the TCP and sweep down while closing: keep them clear of the table
        out[2] = max(out[2], self.table_top + 0.03)
        return out

    # ==============================================================================================
    # gripper
    # ==============================================================================================
    def _update_gripper(self, T_tcp: np.ndarray) -> None:
        if self.gripper_type == "robotiq_2f85":
            self._update_robotiq(T_tcp)
            return
        want = self.gripper_cmd_closed
        if want and not self.gripper_closed:
            self.gripper_closed = True
            self._try_grasp(T_tcp)
        elif not want and self.gripper_closed:
            self.gripper_closed = False
            self._release()
        if self.held is not None:
            T_obj = T_tcp @ self.held_offset
            i = self.props.index(self.held)
            self.prop_view.set_world_poses(
                T_obj[:3, 3].reshape(1, 3), matrix_to_quat_wxyz(T_obj[:3, :3]).reshape(1, 4), indices=[i]
            )
            self.prop_view.set_velocities(np.zeros((1, 3)), np.zeros((1, 3)), indices=[i])
        # visual fingers
        if self.held is not None:
            gap = min(0.045, max(0.012, self.held.radius * 0.7))
        else:
            gap = 0.010 if self.gripper_closed else 0.045
        for op, sign in self.finger_ops:
            op.Set(Gf.Vec3d(0, sign * gap, 0))

    def _update_robotiq(self, T_tcp: np.ndarray) -> None:
        """Physical parallel gripper: drive finger_joint, infer what is held from the finger stall."""
        target = cfg.GRIPPER_CLOSED if self.gripper_cmd_closed else cfg.GRIPPER_OPEN
        a = self._assist
        if self.gripper_cmd_closed and a and a.get("attached"):
            # carried by the grasp assist: keep squeezing gently instead of driving the fingers into the part
            target = min(cfg.GRIPPER_CLOSED, a["finger_at_attach"] + 0.04)
        self.robot.set_dof_position_targets(np.array([target]), dof_indices=self.grip_idx)
        pos = float(self.all_dof_positions()[self.grip_idx[0]])
        self.gripper_pos = float(np.clip(pos / cfg.GRIPPER_CLOSED, 0.0, 1.0))
        self.gripper_closed = self.gripper_cmd_closed
        if self.settings.get("graspAssist", True):
            self._grasp_assist(T_tcp, pos)
            return
        holding = None
        if self.gripper_cmd_closed and self.prop_view is not None:
            # closed and an object sits between the finger pads (fingertip pinches may close fully)
            centers = self.prop_centers()
            name, c = min(centers.items(), key=lambda kv: np.linalg.norm(kv[1] - T_tcp[:3, 3]))
            if np.linalg.norm(c - T_tcp[:3, 3]) < (0.08 if self.gripper_pos < 0.97 else 0.06):
                holding = next(p for p in self.props if p.name == name)
        self.held = holding

    # -- object sets --------------------------------------------------------------------------------
    def is_active(self, prop: Prop) -> bool:
        return prop.category == "item" or prop.category == self.object_set

    def active_prop_names(self) -> list[str]:
        return [p.name for p in self.props if self.is_active(p)]

    def available_object_sets(self) -> list[dict]:
        # every set is listed; a set without objects (e.g. no plush models added yet) has count 0
        return [{"id": k, "label": v, "count": sum(p.category == k for p in self.props)} for k, v in cfg.OBJECT_SETS.items()]

    def set_object_set(self, name: str) -> dict:
        if name not in cfg.OBJECT_SETS:
            raise ValueError(f"unknown object set: {name}")
        self._hand_drop_all()
        if not any(p.category == name for p in self.props):
            raise ValueError(f"no objects in set '{cfg.OBJECT_SETS[name]}' (add models to assets/props/{name}/)")
        self.object_set = name
        self.reset()
        return {"objectSet": name, "objects": self.active_prop_names()}

    def _place_object_set(self) -> None:
        """Active set + items at their home positions on the table, every other prop parked far away."""
        n = len(self.props)
        pos = np.zeros((n, 3))
        quat = np.zeros((n, 4))
        gravity = np.zeros((n, 1), dtype=bool)
        parked = 0
        for i, prop in enumerate(self.props):
            R = quat_wxyz_to_matrix(prop.default_quat)
            c = R @ prop.center_local
            if self.is_active(prop):
                xy = prop.home_xy
                pos[i] = [xy[0] - c[0], xy[1] - c[1], self.table_top + prop.bottom_offset + 0.003]
                gravity[i] = True
            else:
                px, py, pz = cfg.PARKING_ORIGIN
                pos[i] = [px - c[0], py + 0.5 * parked - c[1], pz + prop.bottom_offset]
                parked += 1
            quat[i] = matrix_to_quat_wxyz(R)
        self.prop_view.set_world_poses(pos, quat)
        dyn = [i for i, p in enumerate(self.props) if not p.fixed]  # kinematic bodies take no velocities
        self.prop_view.set_velocities(np.zeros((len(dyn), 3)), np.zeros((len(dyn), 3)), indices=dyn)
        self.prop_view.set_enabled_gravities(gravity)

    def _grasp_object_candidate(self, T_tcp: np.ndarray) -> Prop | None:
        """Prop whose center lies between the finger pads (TCP frame: x approach, y opening, z pad width)."""
        if self.prop_view is None:
            return None
        inv = inv_tf(T_tcp)
        best, best_d = None, 1e9
        for name, c in self.prop_centers().items():
            local = (inv @ np.append(c, 1.0))[:3]
            prop = next(p for p in self.props if p.name == name)
            if not self.is_active(prop) or prop.fixed:
                continue
            if abs(local[1]) < 0.045 and abs(local[2]) < 0.04 and -0.05 < local[0] < 0.045:
                d = float(np.linalg.norm(local))
                if d < best_d:
                    best, best_d = prop, d
        return best

    def _grasp_assist(self, T_tcp: np.ndarray, finger_pos: float) -> None:
        """Physical fingers + "sticky" grasp: the 2F-85 pads tilt under load (mimic joints) and squeeze small,
        round or flat parts out. Once the fingers stop on an object between the pads, that object is carried
        rigidly with the tool until the gripper opens."""
        now = time.monotonic()
        a = self._assist
        if not self.gripper_cmd_closed:
            if a and a.get("attached"):
                i = self.props.index(a["prop"])
                v = np.clip(self.tcp_vel, -2.0, 2.0)
                self.prop_view.set_velocities(v.reshape(1, 3), np.zeros((1, 3)), indices=[i])
                self.prop_view.set_enabled_gravities(np.array([[True]]), indices=[i])
                log.info("grasp assist: released %s", a["prop"].name)
            self._assist = None
            self.held = None
            return
        if a is None:
            prop = self._grasp_object_candidate(T_tcp)
            self._assist = a = {"prop": prop, "t0": now, "last": finger_pos, "still": 0, "attached": False}
            if prop is None:
                self.held = None
                return
        prop = a["prop"]
        if prop is None:
            self.held = None
            return
        i = self.props.index(prop)
        if not a["attached"]:
            # wait until the fingers have moved and then stopped on the object (or a timeout)
            moving = abs(finger_pos - a["last"]) >= 0.002
            a["moved"] = a.get("moved", False) or moving
            a["still"] = 0 if moving else a["still"] + 1
            a["last"] = finger_pos
            if (a["moved"] and a["still"] >= 3) or now - a["t0"] > 1.0:
                pos, quat = self._prop_poses()
                T_obj = make_tf(quat_wxyz_to_matrix(quat[i]), pos[i])
                local = (inv_tf(T_tcp) @ np.append(self.prop_centers()[prop.name], 1.0))[:3]
                if abs(local[1]) < 0.06 and abs(local[2]) < 0.05 and -0.06 < local[0] < 0.06:
                    a["offset"] = inv_tf(T_tcp) @ T_obj
                    a["attached"] = True
                    a["finger_at_attach"] = finger_pos
                    self.prop_view.set_enabled_gravities(np.array([[False]]), indices=[i])
                    log.info("grasp assist: attached %s (fingers %.2f)", prop.name, finger_pos / cfg.GRIPPER_CLOSED)
                else:
                    a["prop"] = None  # the object escaped before the fingers settled
                    self.held = None
                    return
            else:
                self.held = None
                return
        T_obj = T_tcp @ a["offset"]
        self.prop_view.set_world_poses(
            T_obj[:3, 3].reshape(1, 3), matrix_to_quat_wxyz(T_obj[:3, :3]).reshape(1, 4), indices=[i]
        )
        v = np.clip(self.tcp_vel, -2.0, 2.0)
        self.prop_view.set_velocities(v.reshape(1, 3), np.zeros((1, 3)), indices=[i])
        self.held = prop

    def prop_centers(self) -> dict[str, np.ndarray]:
        """World-space bounding-box centers of all props."""
        pos, quat = self._prop_poses()
        return {p.name: pos[i] + quat_wxyz_to_matrix(quat[i]) @ p.center_local for i, p in enumerate(self.props)}

    def _prop_poses(self) -> tuple[np.ndarray, np.ndarray]:
        pos, quat = self.prop_view.get_world_poses()
        return np.asarray(pos.numpy()), np.asarray(quat.numpy())

    def _try_grasp(self, T_tcp: np.ndarray) -> None:
        if self.prop_view is None:
            return
        pos, quat = self._prop_poses()
        best, best_d = None, 1e9
        tcp = T_tcp[:3, 3]
        for i, prop in enumerate(self.props):
            R = quat_wxyz_to_matrix(quat[i])
            center = pos[i] + R @ prop.center_local
            d = np.linalg.norm(center - tcp) - max(prop.radius, prop.half_height)
            if d < 0.035 and d < best_d:
                best, best_d = i, d
        if best is None:
            return
        prop = self.props[best]
        T_obj = make_tf(quat_wxyz_to_matrix(quat[best]), pos[best])
        self.held = prop
        self.held_offset = inv_tf(T_tcp) @ T_obj
        log.info("grasped %s", prop.name)

    def _release(self) -> None:
        if self.gripper_type != "virtual":
            self.held = None
            return
        if self.held is None:
            return
        i = self.props.index(self.held)
        v = np.clip(self.tcp_vel, -2.0, 2.0)
        self.prop_view.set_velocities(v.reshape(1, 3), np.zeros((1, 3)), indices=[i])
        log.info("released %s", self.held.name)
        self.held = None

    # ==============================================================================================
    # commands
    # ==============================================================================================
    def _run_commands(self, allow_sim: bool) -> None:
        for cmd in self.commands.drain():
            if cmd.future.set_running_or_notify_cancel() is False:
                continue
            try:
                if not allow_sim and cmd.name not in ("get_dr_config", "set_dr_config", "set_teleop_settings"):
                    raise RuntimeError("Simulation is not ready yet")
                cmd.future.set_result(self._dispatch(cmd.name, cmd.payload))
            except Exception as exc:  # noqa: BLE001
                log.warning("command %s failed: %s", cmd.name, exc)
                cmd.future.set_exception(exc)

    def _set_physics_hz(self, hz: float) -> float:
        """Change the PhysX step rate; returns the previous rate."""
        from pxr import PhysxSchema

        prim = next((p for p in self.stage.Traverse() if p.IsA(UsdPhysics.Scene)), None)
        api = PhysxSchema.PhysxSceneAPI.Apply(prim)
        attr = api.GetTimeStepsPerSecondAttr()
        old = float(attr.Get() or 60.0)
        attr.Set(int(hz))
        return old

    def _tick_perf_probe(self) -> None:
        p = self.perf_probe
        if p is None:
            return
        if p["left"] <= 0:
            if p["mode"] == "physics_off":
                self.timeline.play()
            if p["mode"].startswith("physics_hz_"):
                self._set_physics_hz(p["old_hz"])
            if p["mode"] == "kit_trace" and self.kit_tracer is not None:
                self.kit_tracer.set_capture_mask(0)
                self.kit_trace_done = True
            self.perf_probe = None
            PERF.tag = None
            return
        if PERF.tag is None:
            if p["mode"] == "physics_off":
                self.timeline.pause()
            if p["mode"].startswith("physics_hz_"):
                p["old_hz"] = self._set_physics_hz(float(p["mode"].rsplit("_", 1)[1]))
            if p["mode"] == "kit_trace" and self.kit_tracer is not None:
                self.kit_tracer.set_capture_mask(0xFFFFFFFFFFFFFFFF)
            PERF.tag = p["mode"]
        p["left"] -= 1

    def _dispatch(self, name: str, payload: dict):
        if name == "perf_probe":
            # diagnostics: run some frames with physics paused or the stream cameras off (main-loop timings
            # are recorded as loop.kit@<mode>)
            mode = payload.get("mode")
            if mode not in ("physics_off", "render_off", "physics_hz_30", "physics_hz_45", "physics_hz_120", "kit_trace"):
                raise ValueError("unknown probe mode")
            if self.perf_probe is None:
                self.perf_probe = {"mode": mode, "left": int(min(max(payload.get("frames", 30), 5), 300))}
            return {"probe": mode}
        if name == "reset":
            self.reset(
                randomize=payload.get("randomize"),
                restore_appearance=bool(payload.get("restoreAppearance", False)),
            )
            return {"reset": True}
        if name == "randomize":
            return self.randomizer.randomize(payload.get("targets"))
        if name == "get_dr_config":
            return self.randomizer.config
        if name == "set_dr_config":
            return self.randomizer.update_config(payload.get("config", payload))
        if name == "set_camera":
            self.set_camera(str(payload.get("camera", "operator")))
            return {"camera": self.camera, "hfov": self.camera_hfov_deg()}
        if name == "set_gripper":
            self.gripper_cmd_closed = bool(payload.get("closed", not self.gripper_cmd_closed))
            return {"closed": self.gripper_cmd_closed}
        if name == "go_home":
            self._go_home()
            return {"home": True}
        if name == "move_tcp":
            # Desktop jogging / scripted moves: absolute ("p") or relative ("delta") TCP position in meters
            if self.target_T is None:
                raise RuntimeError("robot not ready")
            T = self.target_T.copy()
            if "p" in payload:
                T[:3, 3] = np.asarray(payload["p"], dtype=float)[:3]
            if "delta" in payload:
                T[:3, 3] += np.clip(np.asarray(payload["delta"], dtype=float)[:3], -0.5, 0.5)
            T[:3, 3] = self._clamp_workspace(T[:3, 3])
            self.clutch = False
            self.target_T = T
            return {"target": [round(float(v), 4) for v in T[:3, 3]]}
        if name == "set_teleop_settings":
            if "rotation" in payload and "rotationMode" not in payload:  # legacy boolean
                payload["rotationMode"] = "full" if payload["rotation"] else "off"
            for k in ("positionScale", "rotationMode", "smoothing", "hand", "mapping", "graspAssist", "handMode"):
                if k in payload:
                    self.settings[k] = payload[k]
            if self.settings["mapping"] not in ("camera", "operator", "robot"):
                self.settings["mapping"] = "camera"
            if self.settings["rotationMode"] not in ("yaw", "full", "off"):
                self.settings["rotationMode"] = "yaw"
            self.settings["smoothing"] = float(min(0.95, max(0.0, float(self.settings["smoothing"]))))
            self.settings["graspAssist"] = bool(self.settings["graspAssist"])
            self.settings["positionScale"] = float(min(3.0, max(0.1, float(self.settings["positionScale"]))))
            if self.settings.get("handMode") not in ("robot", "grab"):
                self.settings["handMode"] = "robot"
            if self.settings["hand"] not in ("left", "right"):
                self.settings["hand"] = "right"
            return dict(self.settings)
        if name.startswith("rec_") or name == "task_templates":
            if self.rec is None:
                raise RuntimeError("recording is disabled (server started with --no-record)")
            return self.rec.command(name, payload)
        if name == "set_object_set":
            return self.set_object_set(str(payload.get("set", "")))
        if name == "set_view":
            return self.set_view(payload.get("mode"), bool(payload.get("recenter", False)))
        if name == "set_stereo":
            return self.set_stereo(payload.get("enabled"), payload.get("ipd"))
        if name == "set_resolution":
            return self.set_resolution(payload)
        if name == "set_avatar":
            return self.set_avatar(payload)
        if name == "set_mirror":
            return self.randomizer.set_mirror(bool(payload.get("visible", True)))
        if name == "pause":
            self.timeline.pause()
            return {"paused": True}
        if name == "resume":
            self.timeline.play()
            return {"paused": False}
        raise ValueError(f"unknown command {name}")

    def _go_home(self) -> None:
        self.robot.set_dof_positions(self.home, dof_indices=self.arm_idx)
        self.robot.set_dof_velocities(np.zeros_like(self.home), dof_indices=self.arm_idx)
        self._set_arm_targets(self.home)
        self.q_cmd = self.home.copy()
        self.clutch = False
        self.resync_target = True

    def reset(self, randomize: bool | None = None, restore_appearance: bool = False) -> None:
        self._hand_drop_all()
        self._release()
        if self._assist and self._assist.get("attached") and self.prop_view is not None:
            i = self.props.index(self._assist["prop"])
            self.prop_view.set_enabled_gravities(np.array([[True]]), indices=[i])
        self._assist = None
        self.gripper_closed = False
        self.gripper_cmd_closed = False
        self._trigger_was_down = False
        self._go_home()
        if self.grip_idx is not None:
            # open the gripper instantly (all gripper DOFs incl. mimic joints back to 0)
            n = len(self.robot.dof_names)
            arm = set(self.arm_idx.tolist())
            g_idx = np.array([i for i in range(n) if i not in arm], dtype=np.int32)
            if len(g_idx):
                self.robot.set_dof_positions(np.zeros(len(g_idx)), dof_indices=g_idx)
                self.robot.set_dof_velocities(np.zeros(len(g_idx)), dof_indices=g_idx)
            self.robot.set_dof_position_targets(np.array([cfg.GRIPPER_OPEN]), dof_indices=self.grip_idx)
        if self.prop_view is not None:
            self._place_object_set()
        if self.randomizer:
            if restore_appearance:
                self.randomizer.restore()
            do_rand = self.randomizer.config.get("randomizeOnReset", False) if randomize is None else randomize
            if do_rand:
                self.randomizer.randomize(None)

    # ==============================================================================================
    # state
    # ==============================================================================================
    def _snapshot(self) -> dict:
        st = {
            "ready": self.ready,
            "playing": self.timeline.is_playing(),
            "simTime": self.timeline.get_current_time(),
            "fps": round(self.fps_counter[2], 1),
            "camera": self.camera,
            "cameras": list(CAMERA_PATHS),
            "settings": dict(self.settings),
            "clutch": self.clutch,
            "input": self._input_info(),
            "gripper": {
                "closed": self.gripper_closed,
                "holding": self.held.name if self.held else None,
                "position": round(self.gripper_pos, 3),
                "type": self.gripper_type,
            },
            "mirror": self.randomizer.mirror_visible if self.randomizer else True,
            "recording": self.rec.status() if self.rec else None,
            "stereo": self.stereo,
            "tableTop": round(self.table_top, 4),
            "ipd": round(self.ipd, 4),
            "view": self.view_mode,
            "objectSet": self.object_set,
            "objectSets": self.available_object_sets(),
            "immersive": self.immersive.status() if self.immersive else None,
            "handGrab": {k: (v["prop"].name if v else None) for k, v in self._hand_hold.items()},
            "resolution": self.resolution_status(),
            "perf": PERF.summary(),
            "avatar": {
                **self.avatar.status(),
                **self.avatar_settings,
                "collisionActive": self._avatar_body_state["active"],
                "bodies": len(self.avatar.bodies) if self.avatar_body_view is not None else 0,
            }
            if self.avatar
            else None,
        }
        if self.ready:
            try:
                q = self.joint_positions()
                T = self.tcp_transform()
                st["joints"] = [round(math.degrees(v), 2) for v in q]
                st["jointNames"] = list(self.robot.dof_names)[:6]
                st["dofs"] = {n: round(math.degrees(float(v)), 1) for n, v in zip(self.robot.dof_names, self.all_dof_positions())}
                st["tcp"] = {
                    "p": [round(float(v), 4) for v in T[:3, 3]],
                    "q": [round(float(v), 4) for v in matrix_to_quat_wxyz(T[:3, :3])],
                }
                if self.target_T is not None:
                    st["targetError"] = round(float(np.linalg.norm(self.target_T[:3, 3] - T[:3, 3])), 4)
                Tc = self._camera_world_tf()
                st["cameraAxes"] = np.round(Tc[:3, :3] / np.linalg.norm(Tc[:3, :3], axis=0), 4).tolist()
                if self.prop_view is not None:
                    # bounding-box centers (world), not body origins (those differ per asset)
                    st["props"] = [
                        {"name": n, "p": [round(float(v), 3) for v in c]}
                        for n, c in self.prop_centers().items()
                        if n in set(self.active_prop_names())
                    ]
            except Exception as exc:  # noqa: BLE001
                st["error"] = str(exc)
        try:
            st["hfov"] = round(self.camera_hfov_deg(), 2)
        except Exception:  # noqa: BLE001
            pass
        return st


def _twist_about_z(dR: np.ndarray) -> np.ndarray:
    """Keep only the rotation about the world Z axis (swing-twist decomposition)."""
    q = matrix_to_quat_wxyz(dR)
    w, z = q[0], q[3]
    n = math.hypot(w, z)
    if n < 1e-9:
        return np.eye(3)
    return quat_wxyz_to_matrix((w / n, 0.0, 0.0, z / n))


class OneEuroFilter:
    """One-euro filter for the controller position: strong smoothing when the hand is (almost) still,
    little lag when it moves fast. ``amount`` 0 disables it, 1 is very smooth."""

    def __init__(self) -> None:
        self.x: np.ndarray | None = None
        self.dx = np.zeros(3)
        self.t: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: np.ndarray, t: float, amount: float) -> np.ndarray:
        if amount <= 0 or self.x is None or self.t is None or t <= self.t or t - self.t > 0.5:
            self.x, self.t, self.dx = x.copy(), t, np.zeros(3)
            return x
        dt = t - self.t
        self.t = t
        dx = (x - self.x) / dt
        self.dx = self.dx + self._alpha(1.0, dt) * (dx - self.dx)
        min_cutoff = 0.4 + (1.0 - amount) * 6.0  # Hz
        cutoff = min_cutoff + 0.8 * float(np.linalg.norm(self.dx)) / max(amount, 0.05)
        self.x = self.x + self._alpha(cutoff, dt) * (x - self.x)
        return self.x.copy()


__all__ = ["SimController", "CAMERA_PATHS", "matrix_from_rotvec"]
