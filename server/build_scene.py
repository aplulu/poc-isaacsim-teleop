"""Build the VR teleoperation scene as a USD file.

Run with Isaac Sim's python:

    C:\\isaacsim\\python.bat server\\build_scene.py

The script references Isaac Sim / SimReady assets (FANUC CRX-10iA/L, a stainless table, fruit and
kitchen items), measures them with a bounding box cache to lay them out, and saves the composed stage
to ``scene/vr_teleop_scene.usda``. ``main.py`` then simply opens that USD file.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--output", default=None, help="Output USD path")
parser.add_argument("--asset-root", default=None, help="Override Isaac Sim asset root URL")
parser.add_argument(
    "--gripper",
    choices=["robotiq_2f85", "virtual"],
    default="robotiq_2f85",
    help="robotiq_2f85: physical parallel gripper (default). virtual: visual tool + attach-style grasp",
)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": True})

import carb  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

from teleop import config as cfg  # noqa: E402


def wait_for_loading() -> None:
    ctx = omni.usd.get_context()
    for _ in range(2000):
        simulation_app.update()
        _msg, files_loaded, total_files = ctx.get_stage_loading_status()
        if files_loaded >= total_files:
            break
    for _ in range(10):
        simulation_app.update()


def world_bounds(stage: Usd.Stage, path: str) -> Gf.Range3d:
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    return cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()


def set_xform(prim: Usd.Prim, translate=(0, 0, 0), yaw_deg=0.0, scale=None, roll_x_deg=0.0) -> None:
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()

    def precision(attr_name: str):
        # referenced assets may already define the op with float precision; the types must match
        a = prim.GetAttribute(attr_name)
        if a and str(a.GetTypeName()) in ("quatf", "float3"):
            return UsdGeom.XformOp.PrecisionFloat
        return UsdGeom.XformOp.PrecisionDouble

    p_t, p_o, p_s = precision("xformOp:translate"), precision("xformOp:orient"), precision("xformOp:scale")
    xf.AddTranslateOp(p_t).Set((Gf.Vec3f if p_t == UsdGeom.XformOp.PrecisionFloat else Gf.Vec3d)(*translate))
    q = (Gf.Rotation(Gf.Vec3d(1, 0, 0), roll_x_deg) * Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw_deg)).GetQuat()
    xf.AddOrientOp(p_o).Set(Gf.Quatf(q) if p_o == UsdGeom.XformOp.PrecisionFloat else Gf.Quatd(q))
    xf.AddScaleOp(p_s).Set((Gf.Vec3f if p_s == UsdGeom.XformOp.PrecisionFloat else Gf.Vec3d)(*(scale or (1, 1, 1))))


def unit_scale_for(stage: Usd.Stage, asset_url: str) -> float:
    """Scale factor needed to bring an asset authored in another unit into meters."""
    try:
        layer = Sdf.Layer.FindOrOpen(asset_url)
        if layer and layer.pseudoRoot:
            mpu = layer.pseudoRoot.GetInfo("metersPerUnit") if layer.pseudoRoot.HasInfo("metersPerUnit") else 1.0
            return float(mpu) if mpu else 1.0
    except Exception as exc:  # pragma: no cover - diagnostics only
        carb.log_warn(f"metersPerUnit lookup failed for {asset_url}: {exc}")
    return 1.0


def add_reference(stage: Usd.Stage, path: str, url: str) -> Usd.Prim:
    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(url)
    return prim


def make_omnipbr(stage: Usd.Stage, path: str, color, roughness=0.5, metallic=0.0) -> UsdShade.Material:
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/Shader")
    shader.SetSourceAsset("OmniPBR.mdl", "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateIdAttr("OmniPBR")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(metallic)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    mat.CreateDisplacementOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    mat.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return mat


def make_visual_box(stage, path, size, translate, material=None):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xf.AddScaleOp().Set(Gf.Vec3f(*size))
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)
    return cube


CONCAVE_PROPS = {"bowl", "coffee_cup", "banana", "tray"}


def configure_prop_body(body: Usd.Prim, spec, active: bool) -> None:
    """PhysX settings that keep props from being launched when the gripper is pushed into them."""
    body.AddAppliedSchema("PhysxRigidBodyAPI")
    # default depenetration velocity is huge: an object squeezed by the (very stiff) arm would fly away
    body.CreateAttribute("physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float).Set(0.3)
    body.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(16)
    body.CreateAttribute("physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(2)
    body.CreateAttribute("physxRigidBody:enableCCD", Sdf.ValueTypeNames.Bool).Set(not spec.fixed)  # no CCD on kinematic
    body.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(not active)
    if spec.fixed:
        UsdPhysics.RigidBodyAPI(body).CreateKinematicEnabledAttr(True)


def up_axis_rotation(url: str) -> float:
    """X rotation (deg) that turns a Y-up asset upright in this Z-up stage."""
    try:
        layer = Sdf.Layer.FindOrOpen(url)
        if layer and layer.pseudoRoot.HasInfo("upAxis") and layer.pseudoRoot.GetInfo("upAxis") == "Y":
            return 90.0
    except Exception:  # noqa: BLE001
        pass
    return 0.0


class _CustomSpec:
    def __init__(self, name, asset, xy, category, target_size):
        self.name, self.asset, self.xy, self.category = name, asset, xy, category
        self.yaw_deg, self.fixed, self.target_size = 0.0, False, target_size


def find_custom_props() -> list:
    """User models from assets/props/<set>/ (converted to USD when needed)."""
    out = []
    if not os.path.isdir(cfg.CUSTOM_PROPS_DIR):
        return out
    for category in sorted(os.listdir(cfg.CUSTOM_PROPS_DIR)):
        folder = os.path.join(cfg.CUSTOM_PROPS_DIR, category)
        if not os.path.isdir(folder) or category not in cfg.OBJECT_SETS:
            continue
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(cfg.CUSTOM_PROP_EXTENSIONS))
        for i, fname in enumerate(files):
            src = os.path.join(folder, fname)
            usd = src if fname.lower().endswith((".usd", ".usda", ".usdc", ".usdz")) else convert_to_usd(src)
            if usd is None:
                continue
            name = "".join(ch if ch.isalnum() else "_" for ch in os.path.splitext(fname)[0]).strip("_").lower()
            slot = cfg.CUSTOM_PROP_SLOTS[i % len(cfg.CUSTOM_PROP_SLOTS)]
            out.append(_CustomSpec(name, usd.replace("\\", "/"), slot, category, cfg.CUSTOM_PROP_SIZE))
            print(f"[build_scene] custom prop {name} ({category}) from {fname}")
    return out


def convert_to_usd(src: str) -> str | None:
    import asyncio

    import omni.kit.app

    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.kit.asset_converter", True)
    import omni.kit.asset_converter as ac

    out_dir = os.path.join(cfg.PROJECT_ROOT, "assets", "_converted")
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, os.path.splitext(os.path.basename(src))[0] + ".usd")
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return dst

    async def run() -> bool:
        ctx = ac.AssetConverterContext()
        ctx.ignore_animations = True
        ctx.ignore_camera = True
        ctx.ignore_light = True
        ctx.use_meter_as_world_unit = True
        ctx.embed_textures = True
        task = ac.get_instance().create_converter_task(src, dst, None, ctx)
        ok = await task.wait_until_finished()
        if not ok:
            carb.log_error(f"[build_scene] conversion failed for {src}: {task.get_error_message()}")
        return ok

    fut = asyncio.ensure_future(run())
    while not fut.done():
        simulation_app.update()
    return dst if fut.result() else None


def set_collision_approximation(root_prim: Usd.Prim, approximation: str) -> int:
    """Replace SimReady SDF colliders (very slow on CPU PhysX) with convex approximations.

    Instanced sub-assets are de-instanced first so their meshes can be edited in this stage.
    """
    while True:  # de-instancing invalidates the traversal, so restart until no instance is left
        inst = next((p for p in Usd.PrimRange(root_prim) if p.IsInstance()), None)
        if inst is None:
            break
        inst.SetInstanceable(False)
    n = 0
    for p in Usd.PrimRange(root_prim):
        if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr().Set(approximation)
            n += 1
    return n


def add_robotiq(stage: Usd.Stage, root: str, ee_m: Gf.Matrix4d, j6_m: Gf.Matrix4d) -> float:
    """Mount a Robotiq 2F-85 on the flange and merge it into the robot articulation.

    The gripper's own ArticulationRootAPI is removed and its base link is fixed to J6_link, so PhysX builds a
    single articulation (arm J1..J6 + gripper finger_joint with mimic joints). Returns the TCP distance from
    the flange along the approach axis.
    """
    url = root + cfg.GRIPPER_ASSET
    grip = add_reference(stage, cfg.GRIPPER_PRIM, url)
    vsets = grip.GetVariantSets()
    for vs_name in vsets.GetNames():
        vs = vsets.GetVariantSet(vs_name)
        names = vs.GetVariantNames()
        pick = next((n for n in names if "mimic" in n.lower()), None) or next((n for n in names if "physx" in n.lower()), None)
        print(f"[build_scene] gripper variant set {vs_name}: {names} -> {pick or vs.GetVariantSelection()}")
        if pick:
            vs.SetVariantSelection(pick)
    wait_for_loading()
    # gripper +Z (approach) -> flange +X ; gripper +Y (opening) stays +Y
    rot = Gf.Rotation(Gf.Vec3d(0, 1, 0), 90.0)
    ee_rot = ee_m.ExtractRotation()
    q = (rot * ee_rot).GetQuat()
    xf = UsdGeom.Xformable(grip)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(ee_m.ExtractTranslation())
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(q.GetReal(), q.GetImaginary()))

    base = None
    for p in Usd.PrimRange(grip):
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            p.RemoveAppliedSchema("PhysxArticulationAPI")
            print(f"[build_scene] removed articulation root from {p.GetPath()}")
        if base is None and p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetName() in ("base_link", "robotiq_base_link"):
            base = p
    if base is None:
        raise RuntimeError("Robotiq base link not found")
    T_base = UsdGeom.Xformable(base).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    rel = T_base * j6_m.GetInverse()  # base pose expressed in J6_link (row-vector convention)
    joint = UsdPhysics.FixedJoint.Define(stage, f"{cfg.ROBOT_PRIM}/joints/gripper_mount")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(f"{cfg.ROBOT_PRIM}/{cfg.EE_LINK}")])
    joint.CreateBody1Rel().SetTargets([base.GetPath()])
    joint.CreateLocalPos0Attr(Gf.Vec3f(rel.ExtractTranslation()))
    rq = rel.ExtractRotation().GetQuat()
    joint.CreateLocalRot0Attr(Gf.Quatf(rq.GetReal(), Gf.Vec3f(rq.GetImaginary())))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    joint.CreateExcludeFromArticulationAttr(False)

    # Stronger finger drive for teleoperation (asset default is tuned for slow scripted grasps)
    for p in Usd.PrimRange(grip):
        if p.GetName() == "finger_joint":
            drive = UsdPhysics.DriveAPI.Get(p, "angular") or UsdPhysics.DriveAPI.Apply(p, "angular")
            drive.CreateStiffnessAttr(cfg.GRIPPER_STIFFNESS)
            drive.CreateDampingAttr(cfg.GRIPPER_DAMPING)
            drive.CreateMaxForceAttr(cfg.GRIPPER_MAX_FORCE)
    # Grippy finger pads
    pad_mat = UsdShade.Material.Define(stage, f"{cfg.LOOKS_ROOT}/GripperPadPhysics")
    pm = UsdPhysics.MaterialAPI.Apply(pad_mat.GetPrim())
    pm.CreateStaticFrictionAttr(1.5)
    pm.CreateDynamicFrictionAttr(1.2)
    pm.CreateRestitutionAttr(0.0)
    # The pad friction must win against slippery props (metal parts): combine with "max" instead of "average"
    pad_prim = pad_mat.GetPrim()
    pad_prim.AddAppliedSchema("PhysxMaterialAPI")
    pad_prim.CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("max")
    pad_prim.CreateAttribute("physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token).Set("min")
    for p in Usd.PrimRange(grip):
        if p.HasAPI(UsdPhysics.CollisionAPI) or (p.HasAPI(UsdPhysics.RigidBodyAPI) and "finger" in p.GetName()):
            UsdShade.MaterialBindingAPI.Apply(p).Bind(
                pad_mat, UsdShade.Tokens.weakerThanDescendants, "physics"
            )

    gb = world_bounds(stage, cfg.GRIPPER_PRIM)
    length = gb.GetMax()[0] - ee_m.ExtractTranslation()[0]
    print(f"[build_scene] gripper bounds {gb.GetMin()} {gb.GetMax()} length along approach {length:.3f}")
    return max(0.05, length - 0.025)  # TCP between the finger pads


def add_mirror(stage: Usd.Stage, table_top: float, table_bounds: Gf.Range3d) -> None:
    """A wall mirror on the far long side of the table, facing the operator camera."""
    mirror_mat = make_omnipbr(stage, f"{cfg.LOOKS_ROOT}/Mirror", (0.97, 0.97, 0.97), 0.0, 1.0)
    frame_mat = make_omnipbr(stage, f"{cfg.LOOKS_ROOT}/MirrorFrame", (0.08, 0.08, 0.09), 0.5, 0.2)
    cx = (table_bounds.GetMin()[0] + table_bounds.GetMax()[0]) / 2
    y = table_bounds.GetMax()[1] + 0.12
    w, h = cfg.MIRROR_SIZE
    root = UsdGeom.Xform.Define(stage, cfg.MIRROR_PRIM)
    root.AddTranslateOp().Set(Gf.Vec3d(cx, y, table_top + 0.05 + h / 2))
    glass = make_visual_box(stage, f"{cfg.MIRROR_PRIM}/Glass", (w, 0.004, h), (0, -0.004, 0), mirror_mat)
    frame = make_visual_box(stage, f"{cfg.MIRROR_PRIM}/Frame", (w + 0.05, 0.02, h + 0.05), (0, 0.008, 0), frame_mat)
    UsdPhysics.CollisionAPI.Apply(frame.GetPrim())
    stand_h = table_top + 0.05
    stand = make_visual_box(stage, f"{cfg.MIRROR_PRIM}/Stand", (0.06, 0.06, stand_h), (0, 0.03, -h / 2 - stand_h / 2), frame_mat)
    del glass, stand


def main() -> None:
    root = args.asset_root or get_assets_root_path()
    if root is None:
        carb.log_error("Isaac Sim asset root not found. Check your network / asset settings.")
        return
    print(f"[build_scene] asset root: {root}")

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, cfg.WORLD).GetPrim()
    stage.SetDefaultPrim(world)

    # -- physics scene --------------------------------------------------------------------------
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    scene.GetPrim().AddAppliedSchema("PhysxSceneAPI")
    scene.GetPrim().CreateAttribute("physxScene:enableCCD", Sdf.ValueTypeNames.Bool).Set(True)

    # -- environment (grid floor) ---------------------------------------------------------------
    add_reference(stage, "/World/Environment", root + cfg.ENVIRONMENT_ASSET)

    # -- lights (used by domain randomization) --------------------------------------------------
    UsdGeom.Scope.Define(stage, cfg.LIGHTS_ROOT)
    dome = UsdLux.DomeLight.Define(stage, f"{cfg.LIGHTS_ROOT}/Dome")
    dome.CreateIntensityAttr(800.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    key = UsdLux.DistantLight.Define(stage, f"{cfg.LIGHTS_ROOT}/Key")
    key.CreateIntensityAttr(2500.0)
    key.CreateAngleAttr(1.0)
    set_xform(key.GetPrim(), (0, 0, 3))
    UsdGeom.Xformable(key).ClearXformOpOrder()
    UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(35.0, 0.0, 30.0))
    spot = UsdLux.SphereLight.Define(stage, f"{cfg.LIGHTS_ROOT}/Fill")
    spot.CreateIntensityAttr(30000.0)
    spot.CreateRadiusAttr(0.15)
    UsdGeom.Xformable(spot).AddTranslateOp().Set(Gf.Vec3d(0.6, -1.2, 2.0))

    looks = UsdGeom.Scope.Define(stage, cfg.LOOKS_ROOT)  # noqa: F841
    pedestal_mat = make_omnipbr(stage, f"{cfg.LOOKS_ROOT}/PedestalGray", (0.18, 0.19, 0.21), 0.4, 0.3)
    tool_mat = make_omnipbr(stage, f"{cfg.LOOKS_ROOT}/ToolDark", (0.05, 0.05, 0.06), 0.35, 0.6)
    pad_mat = make_omnipbr(stage, f"{cfg.LOOKS_ROOT}/ToolPad", (0.85, 0.35, 0.05), 0.6, 0.0)
    # Randomizable materials (bound only when the corresponding DR option is enabled)
    make_omnipbr(stage, f"{cfg.LOOKS_ROOT}/DR_Table", (0.6, 0.6, 0.6), 0.4, 0.5)
    make_omnipbr(stage, f"{cfg.LOOKS_ROOT}/DR_Floor", (0.4, 0.4, 0.4), 0.7, 0.0)

    # -- table ----------------------------------------------------------------------------------
    table = add_reference(stage, cfg.TABLE_PRIM, root + cfg.TABLE_ASSET)
    wait_for_loading()
    s = unit_scale_for(stage, root + cfg.TABLE_ASSET)
    set_xform(table, (0, 0, 0), 0.0, (s, s, s))
    b = world_bounds(stage, cfg.TABLE_PRIM)
    size = b.GetSize()
    print(f"[build_scene] table raw bounds min={b.GetMin()} max={b.GetMax()} size={size}")
    # Long side along Y (in front of the robot); rotate if needed
    yaw = 90.0 if size[0] > size[1] else 0.0
    set_xform(table, (0, 0, 0), yaw, (s, s, s))
    b = world_bounds(stage, cfg.TABLE_PRIM)
    height = max(b.GetSize()[2], 1e-3)
    z_scale = cfg.TABLE_TOP_HEIGHT / height if abs(height - cfg.TABLE_TOP_HEIGHT) > 0.15 else 1.0
    set_xform(table, (0, 0, 0), yaw, (s, s, s * z_scale))
    b = world_bounds(stage, cfg.TABLE_PRIM)
    c = b.GetMidpoint()
    tx, ty = cfg.TABLE_CENTER_XY[0] - c[0], cfg.TABLE_CENTER_XY[1] - c[1]
    set_xform(table, (tx, ty, -b.GetMin()[2]), yaw, (s, s, s * z_scale))
    b = world_bounds(stage, cfg.TABLE_PRIM)
    table_top = b.GetMax()[2]
    print(f"[build_scene] table placed bounds min={b.GetMin()} max={b.GetMax()} top={table_top:.3f}")
    # The table mesh's own collider (SDF, or a convex decomposition that inflates the top by ~2 cm) is replaced
    # by an exact invisible box for the top surface, so objects rest exactly on the visible table top.
    set_collision_approximation(table, "convexHull")
    for p in Usd.PrimRange(table):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)
    top = UsdGeom.Cube.Define(stage, f"{cfg.TABLE_PRIM}_TopCollider")
    top.CreateSizeAttr(1.0)
    top.CreatePurposeAttr(UsdGeom.Tokens.guide)  # not rendered
    thickness = 0.04
    UsdGeom.Xformable(top).AddTranslateOp().Set(
        Gf.Vec3d(b.GetMidpoint()[0], b.GetMidpoint()[1], table_top - thickness / 2)
    )
    UsdGeom.Xformable(top).AddScaleOp().Set(Gf.Vec3f(b.GetSize()[0], b.GetSize()[1], thickness))
    UsdPhysics.CollisionAPI.Apply(top.GetPrim())
    print(f"[build_scene] table top collider: {b.GetSize()[0]:.3f} x {b.GetSize()[1]:.3f} at z={table_top:.3f}")
    # Make every rigid body of the table kinematic (static furniture)
    for p in Usd.PrimRange(table):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(p).CreateKinematicEnabledAttr(True)
    table.CreateAttribute("teleop:tableTop", Sdf.ValueTypeNames.Double).Set(table_top)
    table.CreateAttribute("teleop:tableMin", Sdf.ValueTypeNames.Double2).Set(Gf.Vec2d(b.GetMin()[0], b.GetMin()[1]))
    table.CreateAttribute("teleop:tableMax", Sdf.ValueTypeNames.Double2).Set(Gf.Vec2d(b.GetMax()[0], b.GetMax()[1]))

    # -- robot pedestal + robot -----------------------------------------------------------------
    rx = min(cfg.ROBOT_BASE_XY[0], b.GetMin()[0] - 0.16)
    ry = cfg.ROBOT_BASE_XY[1]
    pedestal = UsdGeom.Cube.Define(stage, cfg.PEDESTAL_PRIM)
    pedestal.CreateSizeAttr(1.0)
    UsdGeom.Xformable(pedestal).AddTranslateOp().Set(Gf.Vec3d(rx, ry, table_top / 2))
    UsdGeom.Xformable(pedestal).AddScaleOp().Set(Gf.Vec3f(0.30, 0.30, table_top))
    UsdPhysics.CollisionAPI.Apply(pedestal.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(pedestal.GetPrim()).Bind(pedestal_mat)

    robot = add_reference(stage, cfg.ROBOT_PRIM, root + cfg.ROBOT_ASSET)
    set_xform(robot, (rx, ry, table_top), 0.0)
    wait_for_loading()
    # The asset ships with 32 position iterations; that alone costs ~7 ms per physics step on the CPU.
    # 12 keeps the arm + gripper stable while allowing real-time stepping at 60 Hz.
    root_joint = stage.GetPrimAtPath(f"{cfg.ROBOT_PRIM}/root_joint")
    if root_joint:
        root_joint.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(
            cfg.ARTICULATION_POSITION_ITERATIONS
        )

    # -- simple visual tool (two-finger gripper look) on the flange -----------------------------
    ee = stage.GetPrimAtPath("/World/Robot/J6_link/flange/ee_link")
    if not ee.IsValid():
        carb.log_error("ee_link not found in robot asset")
    else:
        ee_m = UsdGeom.Xformable(ee).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        j6_m = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Robot/J6_link")).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        def rows(m):
            return [[round(m[i][j], 3) for j in range(4)] for i in range(4)]

        print(f"[build_scene] ee_link world: {rows(ee_m)}")
        print(f"[build_scene] J6_link world: {rows(j6_m)}")
        tcp_len = 0.13
        if args.gripper == "robotiq_2f85":
            tcp_len = add_robotiq(stage, root, ee_m, j6_m)
        else:
            tool = UsdGeom.Xform.Define(stage, cfg.TOOL_PRIM)  # noqa: F841
            # The flange (ee_link) approaches along its +X axis; the visual tool is modelled along +X.
            # Visual only (no colliders): grasping is done by the server's attach-style gripper.
            make_visual_box(stage, f"{cfg.TOOL_PRIM}/Body", (0.06, 0.10, 0.07), (0.03, 0, 0), tool_mat)
            UsdGeom.Xform.Define(stage, f"{cfg.TOOL_PRIM}/Fingers")
            for side, sy in (("L", 1.0), ("R", -1.0)):
                fx = UsdGeom.Xform.Define(stage, f"{cfg.TOOL_PRIM}/Fingers/Finger{side}")
                fx.AddTranslateOp().Set(Gf.Vec3d(0, sy * 0.04, 0))
                make_visual_box(
                    stage, f"{cfg.TOOL_PRIM}/Fingers/Finger{side}/Geom", (0.08, 0.012, 0.025), (0.10, 0, 0), pad_mat
                )
        world.CreateAttribute("teleop:gripperType", Sdf.ValueTypeNames.String).Set(args.gripper)
        world.CreateAttribute("teleop:tcpOffset", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(tcp_len, 0, 0))

    # -- mirror (behind the table, facing the operator) -----------------------------------------
    add_mirror(stage, table_top, b)

    # -- props ----------------------------------------------------------------------------------
    UsdGeom.Scope.Define(stage, cfg.PROPS_ROOT)
    specs = list(cfg.PROPS) + find_custom_props()
    parked = 0
    for spec in specs:
        path = f"{cfg.PROPS_ROOT}/{spec.name}"
        url = spec.asset if spec.asset.startswith(("file:", "/", "C:", "D:")) and os.path.exists(spec.asset) else root + spec.asset
        prim = add_reference(stage, path, url)
        prim.CreateAttribute("teleop:category", Sdf.ValueTypeNames.String).Set(spec.category)
        wait_for_loading()
        s = unit_scale_for(stage, url)
        up_fix = up_axis_rotation(url)
        set_xform(prim, (0, 0, 0), spec.yaw_deg, (s, s, s), up_fix)
        pb = world_bounds(stage, path)
        if pb.IsEmpty():
            carb.log_warn(f"[build_scene] {spec.name}: empty bounds (asset failed to load?)")
            continue
        target = getattr(spec, "target_size", None)
        if target:  # user models: normalise the size
            s *= target / max(pb.GetSize())
            set_xform(prim, (0, 0, 0), spec.yaw_deg, (s, s, s), up_fix)
            pb = world_bounds(stage, path)
        c = pb.GetMidpoint()
        wx = cfg.WORK_AREA_CENTER_XY[0] + spec.xy[0]
        wy = cfg.WORK_AREA_CENTER_XY[1] + spec.xy[1]
        active = spec.category in ("item", cfg.DEFAULT_OBJECT_SET)
        if active:
            pos = (wx - c[0], wy - c[1], table_top - pb.GetMin()[2] + 0.003)
        else:  # parked far away until its object set is selected (gravity off)
            px, py, pz = cfg.PARKING_ORIGIN
            pos = (px - c[0], py + 0.5 * parked - c[1], pz - pb.GetMin()[2])
            parked += 1
        set_xform(prim, pos, spec.yaw_deg, (s, s, s), up_fix)
        prim.CreateAttribute("teleop:homeXY", Sdf.ValueTypeNames.Double2).Set(Gf.Vec2d(wx, wy))
        prim.CreateAttribute("teleop:fixed", Sdf.ValueTypeNames.Bool).Set(bool(spec.fixed))
        # Some assets (Isaac Lab factory bolts/nuts) are single-body articulations; props must be plain rigid
        # bodies (they are teleported through a rigid-body view, which corrupts the solver for articulations)
        for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if p.HasAPI(UsdPhysics.ArticulationRootAPI) and not p.IsInstanceProxy():
                p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                p.RemoveAppliedSchema("PhysxArticulationAPI")
                print(f"[build_scene]   removed articulation root {p.GetPath()}")
        # ... and some (factory bolts) are pinned to the world with a fixed joint: free them
        for p in list(Usd.PrimRange(prim)):
            if p.IsA(UsdPhysics.Joint):
                j = UsdPhysics.Joint(p)
                if not j.GetBody0Rel().GetTargets() or not j.GetBody1Rel().GetTargets():
                    p.SetActive(False)
                    print(f"[build_scene]   disabled world joint {p.GetPath()}")
        has_rb = any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(prim))
        if not has_rb:
            UsdPhysics.RigidBodyAPI.Apply(prim)
            mass = cfg.CUSTOM_PROP_MASS if getattr(spec, "target_size", None) else None
            if mass:
                UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
            for p in Usd.PrimRange(prim):
                if p.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(p)
                    UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexHull")
        approx = "convexDecomposition" if spec.name in CONCAVE_PROPS else "convexHull"
        n = set_collision_approximation(prim, approx)
        for body in [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)]:
            configure_prop_body(body, spec, active)
        print(f"[build_scene] prop {spec.name} [{spec.category}{'' if active else ', parked'}]: "
              f"size={[round(v, 3) for v in pb.GetSize()]} rigid_body={has_rb} colliders={n} ({approx})")

    # -- cameras --------------------------------------------------------------------------------
    UsdGeom.Scope.Define(stage, cfg.CAMERAS_ROOT)
    work = Gf.Vec3d(cfg.WORK_AREA_CENTER_XY[0], cfg.WORK_AREA_CENTER_XY[1], table_top)
    presets = {
        # Operator view: standing beside the robot (to its right, slightly behind), looking at the table
        "Operator": (Gf.Vec3d(rx - 0.25, ry - 1.05, table_top + 0.80), work + Gf.Vec3d(0.0, 0.05, 0.0)),
        "Front": (Gf.Vec3d(work[0] + 1.3, 0.0, table_top + 0.65), Gf.Vec3d(work[0] - 0.2, 0, table_top + 0.1)),
        "Top": (Gf.Vec3d(work[0] - 0.05, 0.0, table_top + 1.5), work + Gf.Vec3d(0.0, 0, 0)),
        # Third-person camera for the VLA dataset: narrower FOV, framed on the work area
        "Dataset": (Gf.Vec3d(rx - 0.05, ry - 0.95, table_top + 0.75), work + Gf.Vec3d(0.0, 0.08, 0.0)),
    }
    for name, (eye, target) in presets.items():
        cam = UsdGeom.Camera.Define(stage, f"{cfg.CAMERAS_ROOT}/{name}")
        cam.CreateFocalLengthAttr(18.0 if name == "Dataset" else 14.0)
        cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateVerticalApertureAttr(20.955 * 9 / 16)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        up = Gf.Vec3d(1, 0, 0) if name == "Top" else Gf.Vec3d(0, 0, 1)  # straight-down view: image up = +X
        m = Gf.Matrix4d().SetLookAt(eye, target, up).GetInverse()
        UsdGeom.Xformable(cam).AddTransformOp().Set(m)
    wrist = UsdGeom.Camera.Define(stage, cfg.WRIST_CAMERA)
    wrist.CreateFocalLengthAttr(12.0)
    wrist.CreateHorizontalApertureAttr(20.955)
    wrist.CreateVerticalApertureAttr(20.955 * 9 / 16)
    wrist.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50.0))
    # Mounted on the flange above the gripper body, looking along the approach axis (+X) at the fingertips.
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(0.02, 0.0, 0.075), Gf.Vec3d(0.24, 0.0, 0.0), Gf.Vec3d(0, 0, 1)).GetInverse()
    UsdGeom.Xformable(wrist).AddTransformOp().Set(m)

    # Domain randomization binds materials at runtime. Apply the binding API now: applying it while the
    # simulation runs would resync the prims and reset rigid body poses.
    for path in (cfg.TABLE_PRIM, cfg.PEDESTAL_PRIM, "/World/Environment"):
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path))

    # -- metadata -------------------------------------------------------------------------------
    world.CreateAttribute("teleop:assetRoot", Sdf.ValueTypeNames.String).Set(root)
    world.CreateAttribute("teleop:robotBase", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(rx, ry, table_top))

    out = args.output or cfg.SCENE_PATH
    os.makedirs(os.path.dirname(out), exist_ok=True)
    stage.GetRootLayer().Export(out)
    print(f"[build_scene] saved scene to {out}")


try:
    main()
except Exception:  # fast shutdown would otherwise swallow the traceback
    import traceback

    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    raise
finally:
    simulation_app.close()
