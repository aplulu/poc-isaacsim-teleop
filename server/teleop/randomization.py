"""Domain randomization controlled from the web app.

Randomizers:
    objects    - scatter props on the table (non-overlapping, random yaw)
    lighting   - dome / key light intensity, colour temperature and key light direction
    materials  - random colour / roughness / metallic for the table and the floor
    camera     - small position / rotation jitter of the active fixed camera
    physics    - prop mass scale and friction

Every USD edit is authored in the *session layer* so ``restore()`` can drop it and get the original look back.
"""

from __future__ import annotations

import colorsys
import copy
import logging
import math
import random
from typing import TYPE_CHECKING

import numpy as np
from pxr import Gf, Sdf, Usd, UsdLux, UsdPhysics, UsdShade

from . import config as cfg
from .mathutil import matrix_to_quat_wxyz, quat_wxyz_to_matrix, rot_z

if TYPE_CHECKING:
    from .sim import SimController

log = logging.getLogger("teleop.dr")

# "objects" last: it teleports bodies, everything else is plain USD authoring
TARGETS = ("lighting", "materials", "camera", "mirror", "physics", "objects")

DEFAULT_CONFIG: dict = {
    "randomizeOnReset": False,
    "autoInterval": 0.0,
    "seed": None,
    "objects": {"enabled": True, "randomYaw": True, "minSpacing": 0.02},
    "lighting": {
        "enabled": True,
        "intensityScaleMin": 0.5,
        "intensityScaleMax": 1.4,
        "colorTempMin": 3000.0,
        "colorTempMax": 8000.0,
        "randomDirection": True,
    },
    "materials": {"enabled": False, "table": True, "floor": True},
    "camera": {"enabled": False, "positionJitter": 0.04, "rotationJitterDeg": 3.0},
    "mirror": {"enabled": False, "probability": 0.5},
    "physics": {
        "enabled": False,
        "massScaleMin": 0.5,
        "massScaleMax": 2.0,
        "frictionMin": 0.3,
        "frictionMax": 1.2,
    },
}

LIGHT_DEFAULTS = {"Dome": 800.0, "Key": 2500.0, "Fill": 30000.0}


def _merge(dst: dict, src: dict, schema: dict) -> None:
    for k, v in src.items():
        if k not in schema:
            continue
        ref = schema[k]
        if isinstance(ref, dict):
            if isinstance(v, dict):
                _merge(dst[k], v, ref)
        elif isinstance(ref, bool):
            dst[k] = bool(v)
        elif isinstance(ref, (int, float)) and not isinstance(ref, bool):
            try:
                dst[k] = float(v)
            except (TypeError, ValueError):
                pass
        elif ref is None:  # seed
            dst[k] = None if v in (None, "") else int(v)


class DomainRandomizer:
    def __init__(self, sim: SimController) -> None:
        self.sim = sim
        self.stage: Usd.Stage = sim.stage
        self.config = copy.deepcopy(DEFAULT_CONFIG)
        self.rng = random.Random()
        self._timer = 0.0
        self._authored: set[tuple[str, str]] = set()
        self.count = 0
        self._deferred_objects = 0
        self.mirror_visible = True

    # -- config -----------------------------------------------------------------------------------
    def update_config(self, new: dict) -> dict:
        if not isinstance(new, dict):
            raise ValueError("config object expected")
        _merge(self.config, new, DEFAULT_CONFIG)
        c = self.config
        c["autoInterval"] = max(0.0, min(600.0, c["autoInterval"]))
        lt = c["lighting"]
        lt["intensityScaleMin"], lt["intensityScaleMax"] = sorted(
            (max(0.0, lt["intensityScaleMin"]), min(10.0, lt["intensityScaleMax"]))
        )
        lt["colorTempMin"], lt["colorTempMax"] = sorted(
            (max(1000.0, lt["colorTempMin"]), min(15000.0, lt["colorTempMax"]))
        )
        ph = c["physics"]
        ph["massScaleMin"], ph["massScaleMax"] = sorted((max(0.05, ph["massScaleMin"]), min(20.0, ph["massScaleMax"])))
        ph["frictionMin"], ph["frictionMax"] = sorted((max(0.0, ph["frictionMin"]), min(3.0, ph["frictionMax"])))
        cam = c["camera"]
        cam["positionJitter"] = max(0.0, min(0.3, cam["positionJitter"]))
        cam["rotationJitterDeg"] = max(0.0, min(20.0, cam["rotationJitterDeg"]))
        c["mirror"]["probability"] = max(0.0, min(1.0, c["mirror"]["probability"]))
        if c["seed"] is not None:
            self.rng.seed(c["seed"])
        return self.config

    def tick(self, dt: float) -> None:
        if self._deferred_objects > 0:
            self._deferred_objects -= 1
            if self._deferred_objects == 0:
                self._randomize_objects()
        interval = self.config.get("autoInterval", 0.0)
        if interval <= 0:
            self._timer = 0.0
            return
        self._timer += dt
        if self._timer >= interval:
            self._timer = 0.0
            self.randomize(None)

    # -- helpers ----------------------------------------------------------------------------------
    def _set(self, prim: Usd.Prim, attr_name: str, value, type_name=None) -> None:
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            attr = prim.GetAttribute(attr_name)
            if not attr:
                attr = prim.CreateAttribute(attr_name, type_name or Sdf.ValueTypeNames.Float)
            attr.Set(value)
        self._authored.add((str(prim.GetPath()), attr_name))

    def set_mirror(self, visible: bool) -> dict:
        prim = self.stage.GetPrimAtPath(cfg.MIRROR_PRIM)
        if not prim:
            raise RuntimeError("scene has no mirror (rebuild the scene)")
        # visibility only: the collider stays, and no prim resync happens (keeps PhysX state intact)
        self._set(prim, "visibility", "inherited" if visible else "invisible", Sdf.ValueTypeNames.Token)
        self.mirror_visible = visible
        return {"mirror": visible}

    def _randomize_mirror(self) -> None:
        self.set_mirror(self.rng.random() < self.config["mirror"]["probability"])

    def restore(self) -> None:
        layer = self.stage.GetSessionLayer()
        with Sdf.ChangeBlock():
            for path, name in self._authored:
                spec = layer.GetPrimAtPath(path)
                if spec and name in spec.properties:
                    spec.RemoveProperty(spec.properties[name])
        self._authored.clear()
        self.mirror_visible = True
        sim = self.sim
        if sim.prop_view is not None:
            sim.prop_view.set_masses(np.array([p.default_mass for p in sim.props]).reshape(-1, 1))
        log.info("appearance / physics restored")

    # -- main entry -------------------------------------------------------------------------------
    def randomize(self, targets: list[str] | None) -> dict:
        if targets:
            todo = [t for t in targets if t in TARGETS]
        else:
            todo = [t for t in TARGETS if self.config[t]["enabled"]]
        applied = []
        usd_edits = any(t in ("materials", "physics", "mirror") for t in todo)
        for t in todo:
            try:
                if t == "objects" and usd_edits:
                    # Material / physics-material edits make PhysX re-read body poses from USD on the next
                    # step, which would undo the teleport. Place the objects a few frames later instead.
                    self._deferred_objects = 4
                else:
                    getattr(self, f"_randomize_{t}")()
                applied.append(t)
            except Exception as exc:  # noqa: BLE001
                log.exception("randomizer %s failed", t)
                raise RuntimeError(f"{t}: {exc}") from exc
        self.count += 1
        return {"applied": applied, "count": self.count}

    # -- randomizers ------------------------------------------------------------------------------
    def _randomize_objects(self) -> None:
        sim = self.sim
        if sim.prop_view is None:
            return
        sim._release()  # noqa: SLF001
        sim.gripper_cmd_closed = False
        c = self.config["objects"]
        cx, cy = cfg.RANDOM_AREA_CENTER_XY
        hx, hy = cfg.RANDOM_AREA_HALF_EXTENT
        margin = 0.06
        lo = np.maximum([cx - hx, cy - hy], sim.table_min + margin)
        hi = np.minimum([cx + hx, cy + hy], sim.table_max - margin)
        base = np.array(self.stage.GetPrimAtPath(cfg.WORLD).GetAttribute("teleop:robotBase").Get() or (0, 0, 0))
        placed: list[tuple[np.ndarray, float]] = []
        # only the active object set + the always-present items; fixed props (the tray) stay where they are
        movable = [i for i, p in enumerate(sim.props) if sim.is_active(p) and not p.fixed]
        for i, p in enumerate(sim.props):
            if p.fixed and sim.is_active(p):
                c_xy = sim.prop_centers()[p.name][:2]
                placed.append((c_xy, p.radius))
        pos = np.zeros((len(movable), 3))
        quat = np.zeros((len(movable), 4))
        # big items first so the small ones fill the gaps
        order = sorted(range(len(movable)), key=lambda k: -sim.props[movable[k]].radius)
        for k in order:
            i = movable[k]
            prop = sim.props[i]
            r = prop.radius
            xy = None
            for _ in range(200):
                cand = np.array([self.rng.uniform(lo[0] + r, hi[0] - r), self.rng.uniform(lo[1] + r, hi[1] - r)])
                if np.linalg.norm(cand - base[:2]) < 0.28 + r:
                    continue
                if all(np.linalg.norm(cand - q) > r + rq + c["minSpacing"] for q, rq in placed):
                    xy = cand
                    break
            if xy is None:  # fall back: stack it a bit higher at a random spot
                xy = np.array([self.rng.uniform(lo[0], hi[0]), self.rng.uniform(lo[1], hi[1])])
            placed.append((xy, r))
            yaw = self.rng.uniform(-math.pi, math.pi) if c["randomYaw"] else 0.0
            R = rot_z(yaw) @ quat_wxyz_to_matrix(prop.default_quat)
            center_offset = R @ prop.center_local
            p = np.array([xy[0] - center_offset[0], xy[1] - center_offset[1], sim.table_top + prop.bottom_offset + 0.01])
            pos[k] = p
            quat[k] = matrix_to_quat_wxyz(R)
        if movable:
            sim.prop_view.set_world_poses(pos, quat, indices=movable)
            n = len(movable)
            sim.prop_view.set_velocities(np.zeros((n, 3)), np.zeros((n, 3)), indices=movable)

    def _randomize_lighting(self) -> None:
        c = self.config["lighting"]
        for name, default in LIGHT_DEFAULTS.items():
            prim = self.stage.GetPrimAtPath(f"{cfg.LIGHTS_ROOT}/{name}")
            if not prim:
                continue
            light = UsdLux.LightAPI(prim)
            scale = self.rng.uniform(c["intensityScaleMin"], c["intensityScaleMax"])
            self._set(prim, light.GetIntensityAttr().GetName(), float(default * scale))
            self._set(prim, light.GetEnableColorTemperatureAttr().GetName(), True, Sdf.ValueTypeNames.Bool)
            self._set(
                prim,
                light.GetColorTemperatureAttr().GetName(),
                float(self.rng.uniform(c["colorTempMin"], c["colorTempMax"])),
            )
            if name == "Key" and c["randomDirection"]:
                rot = Gf.Vec3f(self.rng.uniform(15, 70), self.rng.uniform(-25, 25), self.rng.uniform(-180, 180))
                self._set(prim, "xformOp:rotateXYZ", rot, Sdf.ValueTypeNames.Float3)
            if name == "Dome" and c["randomDirection"]:
                with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
                    attr = prim.GetAttribute("xformOp:rotateZ")
                    if not attr:
                        from pxr import UsdGeom

                        UsdGeom.Xformable(prim).AddRotateZOp()
                        attr = prim.GetAttribute("xformOp:rotateZ")
                    attr.Set(float(self.rng.uniform(0, 360)))
                self._authored.add((str(prim.GetPath()), "xformOp:rotateZ"))
                self._authored.add((str(prim.GetPath()), "xformOpOrder"))

    def _random_material(self, mat_path: str, metallic_range=(0.0, 1.0)) -> None:
        shader = self.stage.GetPrimAtPath(f"{mat_path}/Shader")
        if not shader:
            return
        h, s, v = self.rng.random(), self.rng.uniform(0.0, 0.8), self.rng.uniform(0.15, 0.95)
        rgb = colorsys.hsv_to_rgb(h, s, v)
        self._set(shader, "inputs:diffuse_color_constant", Gf.Vec3f(*rgb), Sdf.ValueTypeNames.Color3f)
        self._set(shader, "inputs:reflection_roughness_constant", float(self.rng.uniform(0.1, 0.9)))
        self._set(shader, "inputs:metallic_constant", float(self.rng.uniform(*metallic_range)))

    def _bind(self, prim_path: str, mat_path: str) -> None:
        prim = self.stage.GetPrimAtPath(prim_path)
        mat = UsdShade.Material(self.stage.GetPrimAtPath(mat_path))
        if not prim or not mat:
            return
        # MaterialBindingAPI is applied at scene build time: applying it here would change apiSchemas, which
        # resyncs the prim and makes PhysX re-read rigid body poses from USD (undoing teleports).
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            UsdShade.MaterialBindingAPI(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
        self._authored.add((prim_path, "material:binding"))

    def _randomize_materials(self) -> None:
        c = self.config["materials"]
        if c["table"]:
            self._random_material(f"{cfg.LOOKS_ROOT}/DR_Table")
            self._bind(cfg.TABLE_PRIM, f"{cfg.LOOKS_ROOT}/DR_Table")
            self._bind(cfg.PEDESTAL_PRIM, f"{cfg.LOOKS_ROOT}/DR_Table")
        if c["floor"]:
            self._random_material(f"{cfg.LOOKS_ROOT}/DR_Floor", (0.0, 0.3))
            self._bind("/World/Environment", f"{cfg.LOOKS_ROOT}/DR_Floor")

    def _randomize_camera(self) -> None:
        # jitter the camera the operator is looking through and the dataset camera independently
        names = {self.sim.camera if self.sim.camera in self.sim.camera_base_tf else "operator", "dataset"}
        for name in names:
            self._jitter_camera(name)

    def _jitter_camera(self, name: str) -> None:
        c = self.config["camera"]
        base = self.sim.camera_base_tf.get(name)
        if base is None:
            return
        j = c["positionJitter"]
        a = math.radians(c["rotationJitterDeg"])
        jitter = Gf.Matrix4d().SetRotate(
            Gf.Rotation(Gf.Vec3d(1, 0, 0), math.degrees(self.rng.uniform(-a, a)))
            * Gf.Rotation(Gf.Vec3d(0, 1, 0), math.degrees(self.rng.uniform(-a, a)))
            * Gf.Rotation(Gf.Vec3d(0, 0, 1), math.degrees(self.rng.uniform(-a, a)))
        )
        jitter.SetTranslateOnly(Gf.Vec3d(*(self.rng.uniform(-j, j) for _ in range(3))))
        prim = self.stage.GetPrimAtPath(f"{cfg.CAMERAS_ROOT}/{name.capitalize()}")
        if prim:
            self._set(prim, "xformOp:transform", jitter * base, Sdf.ValueTypeNames.Matrix4d)

    def _randomize_physics(self) -> None:
        sim = self.sim
        if sim.prop_view is None:
            return
        c = self.config["physics"]
        masses = np.array([p.default_mass * self.rng.uniform(c["massScaleMin"], c["massScaleMax"]) for p in sim.props])
        sim.prop_view.set_masses(masses.reshape(-1, 1))
        for prop in sim.props:
            mu = self.rng.uniform(c["frictionMin"], c["frictionMax"])
            for p in Usd.PrimRange(self.stage.GetPrimAtPath(prop.root)):
                if p.HasAPI(UsdPhysics.MaterialAPI):
                    self._set(p, "physics:staticFriction", float(mu))
                    self._set(p, "physics:dynamicFriction", float(mu * 0.9))
