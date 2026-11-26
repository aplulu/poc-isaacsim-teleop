"""Shared configuration for the VR teleoperation scene and server."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SERVER_ROOT = os.path.join(PROJECT_ROOT, "server")
SCENE_PATH = os.path.join(PROJECT_ROOT, "scene", "vr_teleop_scene.usda")
WEBAPP_DIST = os.path.join(PROJECT_ROOT, "webapp", "dist")
CERT_DIR = os.path.join(SERVER_ROOT, "certs")
DEPS_DIR = os.path.join(SERVER_ROOT, "_deps")

# --------------------------------------------------------------------------------------------------
# Assets (paths are relative to the Isaac Sim asset root, e.g. ".../Assets/Isaac/6.0")
# --------------------------------------------------------------------------------------------------
ROBOT_ASSET = "/Isaac/Robots/Fanuc/crx10ia_l/crx10ia_l.usd"
ENVIRONMENT_ASSET = "/Isaac/Environments/Grid/default_environment.usd"
TABLE_ASSET = "/Isaac/SimReady/Industrial/Warehouse/Equipment/Stainless_Metal_Table_A03/sm_table_metal_stainless_a03_01.usd"

_FRUIT = "/Isaac/SimReady/Residential/Kitchen/Food/Fruit"
_KITCHEN = "/Isaac/SimReady/Residential/Kitchen"
_AUTO = "/Isaac/SimReady/Industrial/Machines/Automotive"
_TOOLS = "/Isaac/SimReady/Industrial/Tools/Hand"


@dataclass
class PropSpec:
    name: str
    asset: str
    # Home position on the table, relative to WORK_AREA_CENTER_XY (meters, world X/Y)
    xy: tuple[float, float]
    yaw_deg: float = 0.0
    # "fruit" / "part" / "small": graspable object sets (only the active set is on the table)
    # "item": always on the table (task targets / distractors)
    category: str = "fruit"
    fixed: bool = False  # kinematic and never moved by domain randomization (e.g. the target tray)


# Graspable object sets that can be switched from the web app / VR menu
OBJECT_SETS = {
    "fruit": "果物",
    "part": "部品",
    "tool": "工具",
    "electrical": "電装品",
    "fastener": "ボルト・ナット",
    "smallpart": "小部品",
    "small": "小物",
}
DEFAULT_OBJECT_SET = "part"

PROPS: list[PropSpec] = [
    # fruit set
    PropSpec("apple_red", f"{_FRUIT}/Red_Delicious_Apple_A01/sm_food_produce_apple_redDelicous_a01_01.usd", (-0.05, -0.15)),
    PropSpec("banana", f"{_FRUIT}/Banana_A01/sm_food_produce_banana_a01_01.usd", (0.12, 0.02), 30.0),
    # part set (SimReady) - all fit the 85 mm stroke of the 2F-85
    PropSpec("socket", "/Isaac/SimReady/Industrial/Tools/Hand/Socket_7_8in_A01/sm_handTool_socket_7_8in_a01_01.usd", (0.05, 0.02), category="part"),
    PropSpec("pulley", f"{_AUTO}/Engine_Components/Idler_Pulley_A01/sm_engineComponent_idler_pulley_a01_01.usd", (-0.12, 0.05), category="part"),
    PropSpec("relay", f"{_AUTO}/Consumables/Relay_Fuse_A01/sm_consumable_fuse_relay_a01_01.usd", (-0.12, -0.12), category="part"),
    PropSpec("injector", f"{_AUTO}/Engine_Components/Fuel_Injector_A01/sm_fuelSystem_fuel_injector_a01_01.usd", (0.10, -0.22), 30.0, category="part"),
    # small-item set
    PropSpec("golf_ball", "/Isaac/SimReady/Residential/Recreation/Golf_Ball_A01/sm_misc_golfBall_a01_01.usd", (-0.12, 0.05), category="small"),
    PropSpec("candle", "/Isaac/SimReady/Residential/Decor/Candles/Candle_A01/sm_decor_candle_a01_01.usd", (0.05, 0.02), category="small"),
    PropSpec("rubiks_cube", "/Isaac/Props/Rubiks_Cube/rubiks_cube.usd", (-0.10, -0.15), 20.0, category="small"),
    PropSpec("tape_roll", "/Isaac/SimReady/Industrial/Hardware/Tapes/Tape_Refill_A02/sm_misc_tape_refill_a02_01.usd", (0.12, -0.22), category="small"),
    # tool set
    PropSpec("screwdriver", f"{_TOOLS}/Screwdriver_Phillips_Ph1_A01/sm_handTool_screwdriver_phillips_ph1_a01_01.usd", (-0.10, 0.06), category="tool"),
    PropSpec("torx_driver", f"{_TOOLS}/Screwdriver_Torx_T30_A01/sm_handTool_screwdriver_torx_t30_a01_01.usd", (0.12, -0.24), 90.0, category="tool"),
    PropSpec("extension_bar", f"{_TOOLS}/Extension_Bar_A01/sm_handTool_extension_bar_a01_01.usd", (-0.10, -0.15), 20.0, category="tool"),
    PropSpec("socket_small", f"{_TOOLS}/Socket_3_4in_A01/sm_handTool_socket_3_4in_a01_01.usd", (0.06, 0.02), category="tool"),
    # electrical set (automotive fuses)
    PropSpec("mega_fuse", f"{_AUTO}/Consumables/Mega_Orange_Fuse_A01/sm_consumable_fuse_mega_orange_a01_01.usd", (-0.10, 0.03), category="electrical"),
    PropSpec("midi_fuse", f"{_AUTO}/Consumables/Midi_White_Fuse_A01/sm_consumable_fuse_midi_white_a01_01.usd", (0.08, 0.0), category="electrical"),
    PropSpec("micro_fuse", f"{_AUTO}/Consumables/Micro_Green_Fuse_A01/sm_consumable_fuse_micro_green_a01_01.usd", (-0.06, -0.20), 30.0, category="electrical"),
    # fastener set (Isaac Lab factory bolts and nuts)
    PropSpec("bolt_m20", "/Isaac/Props/Factory/factory_bolt_m20_loose/factory_bolt_m20_loose.usd", (-0.10, 0.04), category="fastener"),
    PropSpec("nut_m20", "/Isaac/Props/Factory/factory_nut_m20_loose/factory_nut_m20_loose.usd", (0.08, 0.02), category="fastener"),
    PropSpec("bolt_m16", "/Isaac/Props/Factory/factory_bolt_m16_loose/factory_bolt_m16_loose.usd", (-0.10, -0.16), category="fastener"),
    PropSpec("nut_m16", "/Isaac/Props/Factory/factory_nut_m16_loose/factory_nut_m16_loose.usd", (0.10, -0.24), category="fastener"),
    # small-part set
    PropSpec("bearing", f"{_AUTO}/Consumables/Bearing_Ball_B01/sm_consumable_bearing_ball_b01_01.usd", (-0.10, 0.04), category="smallpart"),
    PropSpec("bearing_small", f"{_AUTO}/Consumables/Bearing_Ball_D01/sm_consumable_bearing_ball_d01_01.usd", (0.08, 0.02), category="smallpart"),
    PropSpec("valve_lifter", f"{_AUTO}/Engine_Components/Engine_Component_Valve_Lifter_A01/sm_engineComponent_valve_lifter_a01_01.usd", (-0.10, -0.16), category="smallpart"),
    PropSpec("tape", "/Isaac/SimReady/Industrial/Hardware/Tapes/Tape_Refill_A03/sm_misc_tape_refill_a03_01.usd", (0.10, -0.24), category="smallpart"),
    # always present
    PropSpec("tray", "/Isaac/SimReady/Industrial/Warehouse/Containers/Gray_Crate_A14_01/sm_crate_gray_a14_01.usd", (0.06, 0.52), 90.0, category="item", fixed=True),
    PropSpec("coffee_cup", f"{_KITCHEN}/Dishware/Coffee_Cup_A01/sm_food_beverage_coffeeCup_a01_01.usd", (0.20, -0.48), category="item"),
    PropSpec("canned_peach", f"{_KITCHEN}/Food/Canned_Goods/Canned_Peach_A01/sm_food_canned_peach_a01_01.usd", (-0.02, -0.45), category="item"),
    PropSpec(
        "cube",
        "/Isaac/SimReady/Residential/Furnishing/Kitchen/Kitchenware/Cube_3in_A01/sm_misc_cube_3in_a01_01.usd",
        (0.24, -0.12),
        category="item",
    ),
]

# User-supplied models are imported from assets/props/<set>/ by build_scene.py (<set> = fruit / part / small).
# Supported: .usd .usda .usdc .usdz (as is) and .glb .gltf .obj .fbx (converted with omni.kit.asset_converter).
CUSTOM_PROPS_DIR = os.path.join(PROJECT_ROOT, "assets", "props")
CUSTOM_PROP_EXTENSIONS = (".usd", ".usda", ".usdc", ".usdz", ".glb", ".gltf", ".obj", ".fbx")
CUSTOM_PROP_SIZE = 0.07  # largest side after scaling (m); the 2F-85 opens 85 mm
CUSTOM_PROP_MASS = 0.1  # kg
CUSTOM_PROP_SLOTS = [(-0.05, -0.05), (0.13, -0.05), (0.02, -0.25), (-0.14, 0.08), (0.14, 0.10), (-0.15, -0.22)]
# Inactive object sets are parked here (far away, gravity off, out of every camera's view)
PARKING_ORIGIN = (-40.0, 0.0, 3.0)

# --------------------------------------------------------------------------------------------------
# Scene layout (meters, Z-up)
# --------------------------------------------------------------------------------------------------
TABLE_CENTER_XY = (0.55, 0.0)  # table center in world XY
TABLE_TOP_HEIGHT = 0.80  # desired height of the table top (table is scaled to fit when needed)
ROBOT_BASE_XY = (0.0, 0.0)  # robot stands on its own pedestal at the table's edge
WORK_AREA_CENTER_XY = (0.52, 0.0)  # home positions are relative to this point
# domain randomization scatters props here (the tray side of the table, y > 0.2, is kept free)
RANDOM_AREA_CENTER_XY = (0.52, -0.17)
RANDOM_AREA_HALF_EXTENT = (0.22, 0.36)
VIEWPORT_EYE = (1.7, -1.3, 1.62)  # Isaac Sim window camera
VIEWPORT_TARGET = (0.45, 0.0, 0.88)

# Paths inside the composed scene
WORLD = "/World"
ROBOT_PRIM = "/World/Robot"
TABLE_PRIM = "/World/Table"
PEDESTAL_PRIM = "/World/Pedestal"
PROPS_ROOT = "/World/Props"
LIGHTS_ROOT = "/World/Lights"
CAMERAS_ROOT = "/World/Cameras"
LOOKS_ROOT = "/World/Looks"
EE_LINK = "J6_link"  # link to which the tool is attached
TOOL_PRIM = "/World/Robot/J6_link/flange/ee_link/Tool"  # visual tool ("virtual" gripper only)
WRIST_CAMERA = "/World/Robot/J6_link/flange/ee_link/WristCamera"
MIRROR_PRIM = "/World/Mirror"
MIRROR_SIZE = (0.9, 0.8)  # width (along X), height

# The RGB annotator returns the image rendered in the previous Kit update: a frame grabbed after update N shows the
# camera pose set before update N-1. The pose sent with a frame (for world-locked display) must be that one.
IMMERSIVE_RENDER_LAG_FRAMES = 1
# Operator avatar (VRM): loaded only if a file is named (--avatar FILE, else env TELEOP_AVATAR) or the default file
# exists. It is converted to USD at startup (skipped while the converted files are up to date).
AVATAR_DIR = os.path.join(PROJECT_ROOT, "assets", "avatars")
AVATAR_CONVERTED_DIR = os.path.join(AVATAR_DIR, "_converted")
AVATAR_ENV = "TELEOP_AVATAR"
AVATAR_DEFAULT_FILE = os.path.join(AVATAR_DIR, "avatar.vrm")


def avatar_file(cli: str | None = "") -> tuple[str | None, str]:
    """(VRM path or None, where it came from / why there is none).

    cli: None = disabled (--no-avatar), "" = not given, else the --avatar path. A named file that does not exist
    gives None (no fallback to the default file, so a typo does not silently load another avatar). Relative paths
    are tried against the working directory, then the project root."""
    if cli is None:
        return None, "disabled (--no-avatar)"
    named, source = (cli, "--avatar") if cli else (os.environ.get(AVATAR_ENV, "").strip().strip('"'), AVATAR_ENV)
    if named:
        for p in (named, os.path.join(PROJECT_ROOT, named)):
            if os.path.isfile(p):
                return os.path.abspath(p), source
        return None, f"{source} file not found: {named}"
    if os.path.isfile(AVATAR_DEFAULT_FILE):
        return AVATAR_DEFAULT_FILE, "default file"
    return None, f"none (set {AVATAR_ENV} to a .vrm file or put one at {AVATAR_DEFAULT_FILE})"
AVATAR_PRIM = "/World/Avatar"
HAND_GRAB_DISTANCE = 0.06  # [m] free viewpoint: grab a prop whose surface is this close to the controller grip
AVATAR_BODIES_ROOT = "/World/AvatarBodies"  # kinematic colliders that follow the avatar
AVATAR_PARKING = (-60.0, 0.0, 10.0)  # where the colliders wait while collisions are off
AVATAR_SCALE = None  # None = match the user's standing eye height (free viewpoint) / the camera height
AVATAR_WRIST_OFFSET = (0.015, -0.01, 0.07)  # wrist relative to the WebXR grip pose (right hand; x mirrored)
AVATAR_HAND_PITCH_DEG = 15.0  # knuckles point this much below the controller's -Z
AVATAR_MAX_STRETCH = 1.3  # arms may stretch this much to reach the controllers (small avatars)
AVATAR_STEP_DIST = 0.18  # [m] a foot steps when it is this far from where it should be
AVATAR_FINGERS_BATCHED = True  # one vectorized pass per hand (False: the bone-by-bone reference)
AVATAR_SPRINGS_WARP = True  # spring bones as a Warp kernel (False: the numpy reference implementation)
AVATAR_IK_WARP = os.environ.get("TELEOP_AVATAR_IK_WARP", "1") != "0"  # the IK as a Warp kernel (needs AVATAR_SPRINGS_WARP; False: the numpy reference)

# How the web app sends its XR samples: "vmc" (VMC Protocol / OSC bundles, default) or "json". The server accepts
# both either way; this only tells the web app which one to use (/api/info "motionFormat").
MOTION_FORMAT = "json" if os.environ.get("TELEOP_MOTION_FORMAT", "vmc").strip().lower() == "json" else "vmc"
AVATAR_NEAR_CLIP = 0.2  # [m] near clip of first-person cameras (hides the inside of the avatar's head)

# Gripper: Robotiq 2F-85 merged into the robot articulation (see build_scene.add_robotiq)
GRIPPER_ASSET = "/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd"
GRIPPER_PRIM = "/World/Gripper"
ARTICULATION_POSITION_ITERATIONS = 12
GRIPPER_JOINT = "finger_joint"
GRIPPER_OPEN = 0.0  # finger_joint position (rad)
GRIPPER_CLOSED = 0.80  # rad (joint limit is ~0.82)
GRIPPER_STIFFNESS = 40.0
GRIPPER_DAMPING = 2.0
GRIPPER_MAX_FORCE = 60.0
# Arm joint torque limits (N m) J1..J6 - enough for teleoperation with gravity off, low enough that the arm
# stalls instead of crushing through objects
ARM_MAX_EFFORT = [150.0, 150.0, 100.0, 30.0, 30.0, 20.0]

# Tool center point offset from the ee_link frame (the flange approaches along its local +X axis)
TOOL_TCP_OFFSET = (0.13, 0.0, 0.0)

# Robot home configuration (degrees) for J1..J6
HOME_JOINTS_DEG = [0.0, 10.0, -30.0, 0.0, -50.0, 0.0]


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8443
    use_tls: bool = True
    jwt_ttl_seconds: int = 8 * 3600
    pairing_ttl_seconds: int = 120
    pairing_max_attempts: int = 5
    pairing_max_pending: int = 8
    pairing_rate_limit_per_minute: int = 6
    video_width: int = 1280
    video_height: int = 720
    video_fps: int = 30
    video_bitrate: int = 8_000_000
    video_codec: str = "auto"  # auto/nvenc (H.264 on the GPU, libx264 fallback) | x264 | vp8
    stereo: bool = True  # initial mode; can be toggled at runtime (set_stereo)
    stereo_ipd: float = 0.064  # replaced by the headset's measured IPD when the user enters VR
    stereo_width: int = 960  # per eye; the stream is 2x wide (side by side)
    stereo_height: int = 540
    immersive_fov: float = 100.0  # free-viewpoint eye camera FOV (square)
    immersive_size: int = 1024  # per eye (square); the stream is 2x wide (1024..2048)
    max_peers: int = 4
    state_hz: float = 20.0
    extra_hosts: list[str] = field(default_factory=list)
    # dataset recording (LeRobot v2.1)
    record: bool = True
    dataset_root: str = os.path.join(PROJECT_ROOT, "datasets")
    dataset_name: str = "crx10ial_teleop"
    record_fps: int = 15
    record_width: int = 256
    record_height: int = 256
    # PhysX steps per simulated second. The timeline follows the wall clock, so every rendered frame runs
    # (frame time x rate) steps: at 60 Hz physics took ~40% of the loop (4 steps of ~7.5 ms per 75 ms frame).
    physics_hz: int = 30
    avatar: str | None = ""  # "" = env TELEOP_AVATAR / AVATAR_DEFAULT_FILE (see avatar_file), None = disabled, else a .vrm path
    record_codec: str = "libsvtav1"  # LeRobot default (AV1); "libx264" for H.264
