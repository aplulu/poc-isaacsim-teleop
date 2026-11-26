"""HTTPS server (aiohttp) running in its own thread: REST API, WebRTC signaling and the built web app."""

from __future__ import annotations

import asyncio
import logging
import os
import ssl
import threading
import time

import numpy as np
from aiohttp import web

from .auth import AuthError, AuthManager
from .bridge import CommandQueue, FrameHub, StateHub, TeleopInput
from . import config as cfg_mod
from .config import ServerConfig
from .nvenc import install as install_video_codec
from .webrtc import PeerManager, configure_bitrate

log = logging.getLogger("teleop.http")

PUBLIC_API = {"/api/info", "/api/pair/request", "/api/pair/confirm"}
COMMAND_TIMEOUT = 15.0

# Commands the web app may run on the simulation thread
ALLOWED_COMMANDS = {
    "reset",
    "perf_probe",
    "randomize",
    "get_dr_config",
    "set_dr_config",
    "set_camera",
    "set_gripper",
    "go_home",
    "move_tcp",
    "set_mirror",
    "set_avatar",
    "set_resolution",
    "set_stereo",
    "set_view",
    "set_object_set",
    "task_templates",
    "rec_start",
    "rec_stop",
    "rec_discard",
    "rec_status",
    "rec_settings",
    "rec_delete_last",
    "set_teleop_settings",
    "pause",
    "resume",
}


def _client_ip(request: web.Request) -> str:
    return request.remote or "unknown"


def _json_error(status: int, message: str, code: str = "error") -> web.Response:
    return web.json_response({"error": code, "message": message}, status=status)


class TeleopHttpServer:
    def __init__(
        self,
        cfg: ServerConfig,
        auth: AuthManager,
        frames: FrameHub,
        state: StateHub,
        teleop: TeleopInput,
        commands: CommandQueue,
        static_dir: str,
        cert: tuple[str, str] | None,
    ) -> None:
        self.cfg = cfg
        self.auth = auth
        self.state = state
        self.commands = commands
        self.static_dir = static_dir
        self.cert = cert
        configure_bitrate(cfg.video_bitrate)
        install_video_codec(cfg.video_codec)
        self.peers = PeerManager(
            auth=auth,
            frames=frames,
            state=state,
            teleop=teleop,
            commands=commands,
            fps=cfg.video_fps,
            state_hz=cfg.state_hz,
            max_peers=cfg.max_peers,
            video_codec=cfg.video_codec,
        )
        self.loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._runner: web.AppRunner | None = None
        self._ready = threading.Event()
        self.error: Exception | None = None
        self._bg_tasks: set[asyncio.Future] = set()

    # -- lifecycle --------------------------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="teleop-http", daemon=True)
        self._thread.start()
        self._ready.wait(15)
        if self.error:
            raise self.error

    def _run(self) -> None:
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.peers.bind_loop(self.loop)
        try:
            self.loop.run_until_complete(self._serve())
        except Exception as exc:  # noqa: BLE001
            self.error = exc
            log.exception("HTTP server failed to start")
            self._ready.set()
            return
        self._ready.set()
        self.loop.run_forever()

    async def _serve(self) -> None:
        app = web.Application(middlewares=[self._error_middleware, self._auth_middleware], client_max_size=2 * 1024**2)
        r = app.router
        r.add_get("/api/info", self.h_info)
        r.add_post("/api/pair/request", self.h_pair_request)
        r.add_post("/api/pair/confirm", self.h_pair_confirm)
        r.add_get("/api/session", self.h_session)
        r.add_post("/api/session/logout", self.h_logout)
        r.add_post("/api/webrtc/offer", self.h_offer)
        r.add_get("/api/state", self.h_state)
        r.add_post("/api/command", self.h_command)
        r.add_post("/api/reset", self._shortcut("reset"))
        r.add_post("/api/randomize", self._shortcut("randomize"))
        r.add_get("/api/dr/config", self._shortcut("get_dr_config"))
        r.add_put("/api/dr/config", self._shortcut("set_dr_config"))
        r.add_post("/api/camera", self._shortcut("set_camera"))
        r.add_get("/api/snapshot", self.h_snapshot)
        r.add_route("*", "/api/{tail:.*}", self.h_not_found)
        r.add_get("/{tail:.*}", self.h_static)

        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        ssl_ctx = None
        if self.cert:
            ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_ctx.load_cert_chain(*self.cert)
        site = web.TCPSite(self._runner, self.cfg.host, self.cfg.port, ssl_context=ssl_ctx)
        await site.start()
        log.info("listening on %s://%s:%d", "https" if ssl_ctx else "http", self.cfg.host, self.cfg.port)

    def stop(self) -> None:
        if not self.loop:
            return

        async def _shutdown() -> None:
            await self.peers.close_all()
            if self._runner:
                await self._runner.cleanup()

        try:
            asyncio.run_coroutine_threadsafe(_shutdown(), self.loop).result(5)
        except Exception:  # noqa: BLE001
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)

    def revoke_session(self, session_id: str) -> None:
        """Called from the Kit UI thread."""
        self.auth.revoke(session_id)
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.peers.close_session(session_id), self.loop)

    @property
    def controller_peer(self) -> str | None:
        return self.peers.controller

    # -- middlewares ------------------------------------------------------------------------------
    @web.middleware
    async def _error_middleware(self, request: web.Request, handler):
        try:
            resp = await handler(request)
        except AuthError as exc:
            return _json_error(exc.status, str(exc), exc.code)
        except web.HTTPException:
            raise
        except asyncio.TimeoutError:
            return _json_error(504, "Simulation did not respond in time", "timeout")
        except (ValueError, RuntimeError) as exc:  # rejected command (bad input / wrong state)
            return _json_error(400, str(exc), "bad_request")
        except Exception as exc:  # noqa: BLE001
            log.exception("request failed: %s", request.path)
            return _json_error(500, str(exc), "internal")
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        if request.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @web.middleware
    async def _auth_middleware(self, request: web.Request, handler):
        if request.path.startswith("/api/") and request.path not in PUBLIC_API:
            header = request.headers.get("Authorization", "")
            token = header[7:].strip() if header.lower().startswith("bearer ") else ""
            request["session"] = self.auth.verify_token(token)
        return await handler(request)

    # -- handlers ---------------------------------------------------------------------------------
    async def _read_json(self, request: web.Request) -> dict:
        if not request.can_read_body:
            return {}
        try:
            data = await request.json()
        except ValueError as exc:
            raise web.HTTPBadRequest(text="invalid JSON") from exc
        if not isinstance(data, dict):
            raise web.HTTPBadRequest(text="JSON object expected")
        return data

    async def h_info(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "name": "Isaac Sim VR Teleop",
                "robot": "FANUC CRX-10iA/L",
                "codeLength": 6,
                "pairingTtl": self.cfg.pairing_ttl_seconds,
                "video": {
                    "width": self.cfg.stereo_width * 2 if self.cfg.stereo else self.cfg.video_width,
                    "height": self.cfg.stereo_height if self.cfg.stereo else self.cfg.video_height,
                    "fps": self.cfg.video_fps,
                    "stereo": self.cfg.stereo,
                },
                "motionFormat": cfg_mod.MOTION_FORMAT,
                "serverTime": time.time(),
            }
        )

    async def h_pair_request(self, request: web.Request) -> web.Response:
        body = await self._read_json(request)
        req = self.auth.create_pairing_request(
            str(body.get("deviceName", "")), _client_ip(request), request.headers.get("User-Agent", "")
        )
        log.info("pairing requested by %s (%s)", req.device_name, req.remote_ip)
        return web.json_response(
            {"requestId": req.request_id, "expiresAt": req.expires_at, "expiresIn": int(req.expires_at - time.time())}
        )

    async def h_pair_confirm(self, request: web.Request) -> web.Response:
        body = await self._read_json(request)
        token, session = self.auth.confirm_pairing(
            str(body.get("requestId", "")), str(body.get("code", "")), _client_ip(request)
        )
        log.info("paired %s (%s)", session.device_name, session.remote_ip)
        return web.json_response(
            {
                "token": token,
                "tokenType": "Bearer",
                "expiresAt": session.expires_at,
                "sessionId": session.session_id,
                "deviceName": session.device_name,
            }
        )

    async def h_session(self, request: web.Request) -> web.Response:
        s = request["session"]
        return web.json_response(
            {"sessionId": s.session_id, "deviceName": s.device_name, "expiresAt": s.expires_at, "issuedAt": s.issued_at}
        )

    async def h_logout(self, request: web.Request) -> web.Response:
        s = request["session"]
        self.auth.revoke(s.session_id)
        # close the peer connection in the background: aiortc's close can take minutes when the other side is
        # still connected (the logout request used to time out on the client)
        task = asyncio.ensure_future(self.peers.close_session(s.session_id))
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)
        return web.json_response({"ok": True})

    async def h_offer(self, request: web.Request) -> web.Response:
        body = await self._read_json(request)
        if body.get("type") != "offer" or not body.get("sdp"):
            raise web.HTTPBadRequest(text="SDP offer expected")
        answer = await self.peers.handle_offer(request["session"], body["sdp"], "offer")
        return web.json_response(answer)

    async def h_state(self, request: web.Request) -> web.Response:
        return web.json_response(self.state.get())

    async def _run_command(self, name: str, payload: dict):
        if name not in ALLOWED_COMMANDS:
            raise web.HTTPBadRequest(text=f"unknown command: {name}")
        fut = self.commands.submit(name, payload)
        return await asyncio.wait_for(asyncio.wrap_future(fut), COMMAND_TIMEOUT)

    async def h_command(self, request: web.Request) -> web.Response:
        body = await self._read_json(request)
        result = await self._run_command(str(body.get("name", "")), body.get("payload") or {})
        return web.json_response({"ok": True, "result": result})

    def _shortcut(self, name: str):
        async def handler(request: web.Request) -> web.Response:
            payload = await self._read_json(request) if request.method != "GET" else {}
            result = await self._run_command(name, payload)
            return web.json_response({"ok": True, "result": result})

        return handler

    async def h_snapshot(self, request: web.Request) -> web.Response:
        """Current camera image as JPEG (preview / debugging without a WebRTC peer)."""
        frames = self.peers.frames
        seq = frames.seq
        frames.snapshot_wanted = True
        for _ in range(100):
            if frames.seq != seq:
                break
            await asyncio.sleep(0.02)
        img = frames.latest
        if img is None:
            return _json_error(503, "no frame yet", "no_frame")
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(np.ascontiguousarray(img[:, :, :3])).save(buf, format="JPEG", quality=88)  # frames are RGBA
        return web.Response(body=buf.getvalue(), content_type="image/jpeg")

    async def h_not_found(self, request: web.Request) -> web.Response:
        return _json_error(404, "not found", "not_found")

    async def h_static(self, request: web.Request) -> web.StreamResponse:
        root = os.path.realpath(self.static_dir)
        rel = request.match_info.get("tail", "") or "index.html"
        path = os.path.realpath(os.path.join(root, rel))
        if not path.startswith(root + os.sep) and path != root:
            raise web.HTTPForbidden()
        if not os.path.isfile(path):
            path = os.path.join(root, "index.html")  # SPA fallback
        if not os.path.isfile(path):
            return web.Response(
                text="Web app is not built yet. Run `npm install && npm run build` in webapp/.",
                content_type="text/plain",
                status=503,
            )
        resp = web.FileResponse(path)
        if "/assets/" in path.replace("\\", "/"):
            resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            resp.headers["Cache-Control"] = "no-cache"
        return resp
