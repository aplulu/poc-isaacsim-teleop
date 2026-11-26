"""Isaac Sim VR teleoperation server.

    C:\\isaacsim\\python.bat server\\main.py [--port 8443] [--no-stereo] [--headless] [--no-tls]

* Starts Isaac Sim (SimulationApp) and opens ``scene/vr_teleop_scene.usda`` (build it first with build_scene.py).
* Starts an HTTPS server (aiohttp, separate thread) that serves the web app, the REST API and WebRTC signaling.
* Shows a "VR Teleop - Pairing" window with the 6-digit code of each connection request.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from teleop import config as cfg  # noqa: E402

parser = argparse.ArgumentParser(description="Isaac Sim VR teleop server")
parser.add_argument("--scene", default=cfg.SCENE_PATH)
parser.add_argument("--host", default="0.0.0.0")
parser.add_argument("--port", type=int, default=8443)
parser.add_argument("--no-tls", action="store_true", help="Serve plain HTTP (WebXR then only works on localhost)")
parser.add_argument(
    "--stereo", action=argparse.BooleanOptionalAction, default=True,
    help="Start with side-by-side stereo video (default; --no-stereo for mono; toggle in the web app)",
)
parser.add_argument("--stereo-size", type=int, nargs=2, default=[960, 540], metavar=("W", "H"), help="Per-eye stereo size")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--bitrate", type=int, default=8_000_000)
parser.add_argument("--video-codec", choices=["auto", "nvenc", "x264", "vp8"], default="auto",
                    help="auto/nvenc: H.264 on the GPU (falls back to libx264), x264: H.264 on the CPU, vp8: VP8")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--viewport", choices=["auto", "on", "off"], default="auto",
                    help="Isaac Sim's own viewport: auto = paused while someone receives the stream (always with "
                    "--headless), on = always rendered, off = never")
parser.add_argument("--extra-host", action="append", default=[], help="Extra host name for the TLS certificate")
parser.add_argument("--no-record", action="store_true", help="Disable dataset recording")
parser.add_argument("--profile", action="store_true", help="Log main-loop timings every 5 s")
parser.add_argument("--dataset", default="crx10ial_teleop", help="Dataset name (folder under --dataset-root)")
parser.add_argument("--dataset-root", default=cfg.ServerConfig().dataset_root)
parser.add_argument("--record-fps", type=int, default=15)
parser.add_argument("--record-size", type=int, nargs=2, default=[256, 256], metavar=("W", "H"))
parser.add_argument("--avatar", default="", help="VRM file for the operator avatar (default: env TELEOP_AVATAR, else assets/avatars/avatar.vrm if it exists)")
parser.add_argument("--no-avatar", action="store_true", help="Do not load the operator avatar")
parser.add_argument("--physics-hz", type=int, default=30, help="PhysX steps per second (60 = finer, costs ~2x)")
parser.add_argument("--kit-trace", default="", metavar="JSON",
                    help="Diagnostics: arm the Kit CPU profiler; the perf_probe 'kit_trace' captures N frames, "
                    "then the server exits and writes a Chrome trace here (tools/kit_trace_report.py)")
parser.add_argument("--codec", choices=["libsvtav1", "libx264"], default="libsvtav1", help="Dataset video codec")
args, _unknown = parser.parse_known_args()

if not os.path.exists(args.scene):
    print(f"Scene not found: {args.scene}\nRun: C:\\isaacsim\\python.bat server\\build_scene.py", file=sys.stderr)
    sys.exit(1)

from isaacsim import SimulationApp  # noqa: E402

simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "width": 1600,
        "height": 900,
        # Real-time simulation: the timeline advances by the wall-clock frame time (physics substeps at
        # --physics-hz) instead of a fixed 1/60 s per rendered frame - otherwise the teleoperated arm runs in slow
        # motion whenever rendering drops below 60 fps.
        "extra_args": [
            "--/app/runLoops/main/manualModeEnabled=false",
            "--/app/player/useFixedTimeStepping=false",
            "--/persistent/simulation/minFrameRate=15",
        ]
        + (
            [
                "--/app/profilerBackend=cpu",
                "--/app/profileFromStart=true",
                "--/profiler/enabled=true",
                "--/plugins/carb.profiler-cpu.plugin/saveProfile=true",
                "--/plugins/carb.profiler-cpu.plugin/compressProfile=false",
                f"--/plugins/carb.profiler-cpu.plugin/filePath={os.path.abspath(args.kit_trace)}",
            ]
            if args.kit_trace
            else []
        ),
    }
)

# Third-party deps (aiortc, PyJWT) live in server/_deps (see install_deps.bat). They are appended so
# Isaac Sim's own packages (numpy, cryptography, aiohttp...) take precedence.
if os.path.isdir(cfg.DEPS_DIR) and cfg.DEPS_DIR not in sys.path:
    sys.path.append(cfg.DEPS_DIR)
    if hasattr(os, "add_dll_directory"):
        for sub in ("av.libs", "pylibsrtp.libs", "google_crc32c"):
            d = os.path.join(cfg.DEPS_DIR, sub)
            if os.path.isdir(d):
                os.add_dll_directory(d)

import carb  # noqa: E402
import isaacsim.core.experimental.utils.app as app_utils  # noqa: E402
import omni.usd  # noqa: E402

from teleop.auth import AuthManager  # noqa: E402
from teleop.bridge import CommandQueue, FrameHub, StateHub, TeleopInput  # noqa: E402
from teleop.certs import ensure_certificate, local_ipv4_addresses, primary_lan_ip  # noqa: E402
from teleop.http_server import TeleopHttpServer  # noqa: E402
from teleop.perf import PERF  # noqa: E402
from teleop.sim import SimController  # noqa: E402

os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.__stdout__),
        logging.FileHandler(os.path.join(HERE, "logs", "teleop.log"), encoding="utf-8"),
    ],
    force=True,  # Kit installs its own root handlers; ours must win so messages reach the console/file
)
for noisy in ("aioice", "aiortc", "aiohttp.access"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("teleop.main")


def open_scene(path: str) -> None:
    ctx = omni.usd.get_context()
    ctx.open_stage(path)
    for _ in range(5000):
        simulation_app.update()
        _msg, loaded, total = ctx.get_stage_loading_status()
        if ctx.get_stage() is not None and loaded >= total and not ctx.is_new_stage():
            break
    for _ in range(30):
        simulation_app.update()


def main() -> None:
    scfg = cfg.ServerConfig(
        host=args.host,
        port=args.port,
        use_tls=not args.no_tls,
        video_width=args.width,
        video_height=args.height,
        video_fps=args.fps,
        video_bitrate=args.bitrate,
        video_codec=args.video_codec,
        stereo=args.stereo,
        stereo_width=args.stereo_size[0],
        stereo_height=args.stereo_size[1],
        extra_hosts=args.extra_host,
        record=not args.no_record,
        dataset_root=os.path.abspath(args.dataset_root),
        dataset_name=args.dataset,
        record_fps=args.record_fps,
        record_width=args.record_size[0],
        record_height=args.record_size[1],
        record_codec=args.codec,
        avatar=None if args.no_avatar else args.avatar,
        physics_hz=max(15, min(args.physics_hz, 240)),
    )
    log.info("opening scene %s", args.scene)
    open_scene(os.path.abspath(args.scene))

    frames, state, teleop, commands = FrameHub(), StateHub(), TeleopInput(), CommandQueue()
    auth = AuthManager(
        jwt_ttl_seconds=scfg.jwt_ttl_seconds,
        pairing_ttl_seconds=scfg.pairing_ttl_seconds,
        max_attempts=scfg.pairing_max_attempts,
        max_pending=scfg.pairing_max_pending,
        rate_limit_per_minute=scfg.pairing_rate_limit_per_minute,
    )
    sim = SimController(scfg, frames, state, teleop, commands)
    sim.setup()

    cert = ensure_certificate(cfg.CERT_DIR, scfg.extra_hosts) if scfg.use_tls else None
    server = TeleopHttpServer(scfg, auth, frames, state, teleop, commands, cfg.WEBAPP_DIST, cert)
    server.start()
    scheme = "https" if cert else "http"
    lan = primary_lan_ip()
    urls = [f"{scheme}://{lan}:{scfg.port}/"] + [
        f"{scheme}://{ip}:{scfg.port}/" for ip in local_ipv4_addresses() if ip not in (lan,)
    ]
    for u in urls:
        log.info("web app: %s", u)

    if not args.headless:
        # Isaac Sim's own viewport: a closer look at the table and the robot
        try:
            from isaacsim.core.rendering_manager import ViewportManager

            ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=cfg.VIEWPORT_EYE, target=cfg.VIEWPORT_TARGET)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not set the viewport camera: %s", exc)

    # the GUI viewport is a third full render every frame: pause it when nobody looks at it
    sim.viewport_mode = "off" if args.headless and args.viewport == "auto" else args.viewport

    window = None
    if args.headless:
        # No window to show the code in: print it to the Isaac Sim console instead
        auth.on_new_request = lambda r: log.warning(
            "PAIRING CODE for %s (%s): %s", r.device_name, r.remote_ip, r.code
        )
    else:
        from teleop.ui_window import PairingWindow

        window = PairingWindow(auth, urls, server.revoke_session, state.get)

    app_utils.play()
    tracer = None
    if args.kit_trace:
        import carb.profiler

        tracer = carb.profiler.acquire_profiler_interface()
        tracer.set_capture_mask(0)  # nothing is recorded until the kit_trace probe
        sim.kit_tracer = tracer
        log.warning("Kit profiler armed: perf_probe kit_trace writes %s and stops the server", args.kit_trace)
    last = time.monotonic()
    prof = [0.0, 0.0, 0.0, 0, last]
    try:
        while simulation_app.is_running():
            now = time.monotonic()
            dt = now - last
            last = now
            t0 = time.perf_counter()
            if tracer is not None:
                tracer.begin(1, "teleop.pre")
            sim.pre_update(dt)
            t1 = time.perf_counter()
            if tracer is not None:
                tracer.end(1)
                tracer.begin(1, "teleop.kit")
            simulation_app.update()
            t2 = time.perf_counter()
            if tracer is not None:
                tracer.end(1)
                tracer.begin(1, "teleop.post")
            sim.post_update(dt)
            if tracer is not None:
                tracer.end(1)
                if sim.kit_trace_done:
                    log.warning("Kit trace captured; stopping the server to write %s", args.kit_trace)
                    break
            if window is not None:
                window.update()
            t3 = time.perf_counter()
            sfx = f"@{PERF.tag}" if PERF.tag else ""
            PERF.add("loop.pre" + sfx, (t1 - t0) * 1e3)
            PERF.add("loop.kit" + sfx, (t2 - t1) * 1e3)
            PERF.add("loop.post" + sfx, (t3 - t2) * 1e3)
            if args.profile or frames.has_subscribers:
                PERF.maybe_log()  # every 5 s while someone watches the stream
            if args.profile:
                prof[0] += t1 - t0
                prof[1] += t2 - t1
                prof[2] += t3 - t2
                prof[3] += 1
                if now - prof[4] > 5.0:
                    n = max(prof[3], 1)
                    log.info(
                        "loop: pre %.1f ms | kit update %.1f ms | post %.1f ms | %.1f fps",
                        prof[0] / n * 1e3, prof[1] / n * 1e3, prof[2] / n * 1e3, n / (now - prof[4]),
                    )
                    prof[:] = [0.0, 0.0, 0.0, 0, now]
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # noqa: BLE001
        carb.log_error(f"teleop main loop crashed: {exc}")
        raise
    finally:
        if sim.rec is not None:
            sim.rec.close()
        server.stop()
        if window is not None:
            window.destroy()
        simulation_app.close()


if __name__ == "__main__":
    main()
