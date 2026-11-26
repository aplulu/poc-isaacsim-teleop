"""WebRTC (aiortc): video stream of the Isaac Sim camera + DataChannels for teleoperation."""

from __future__ import annotations

import asyncio
import fractions
import json
import logging
import struct
import time
import uuid

import av
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.codecs import h264, vpx

from .auth import AuthManager, Session
from .bridge import CommandQueue, FrameHub, StateHub, TeleopInput
from .nvenc import prefer_video_codec
from .perf import PERF, instrument_encoders
from .vmc import decode_sample as decode_vmc

log = logging.getLogger("teleop.webrtc")

VIDEO_CLOCK_RATE = 90000


def configure_bitrate(bitrate: int) -> None:
    """aiortc's default encoder bitrates are tuned for the internet; on a LAN we can afford much more."""
    for mod in (h264, vpx):
        mod.MAX_BITRATE = max(bitrate, mod.MIN_BITRATE)
        mod.DEFAULT_BITRATE = min(max(bitrate // 2, mod.MIN_BITRATE), mod.MAX_BITRATE)
    instrument_encoders()


def _negotiated_mime(pc) -> str | None:
    for t in pc.getTransceivers():
        if t.kind == "video" and getattr(t, "_codecs", None):
            return t._codecs[0].mimeType.lower()  # noqa: SLF001
    return None


class SimVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, hub: FrameHub, loop: asyncio.AbstractEventLoop, fps: int, kind: str = "raw") -> None:
        """kind "raw": RGBA frames, encoded by aiortc's encoder for this peer; "h264": access units of the shared
        GPU encoder, only packetised here."""
        super().__init__()
        self._hub = hub
        self.stream_kind = kind
        self._queue = hub.subscribe(loop, kind)
        self.sender = None  # RTCRtpSender, to forward its keyframe requests (PLI / FIR) to the shared encoder
        self._fps = fps
        self._start: float | None = None
        self._pts = 0

    on_meta = None  # callback(pts, meta): tells the client which pose a frame was rendered with

    async def recv(self) -> av.VideoFrame | av.Packet:
        frame, meta, t_pub = await self._queue.get()
        t0 = time.perf_counter()
        PERF.add("track.queue", (t0 - t_pub) * 1e3)
        # Pace output to the configured fps and generate monotonically increasing timestamps
        now = time.monotonic()
        if self._start is None:
            self._start = now
        self._pts = max(self._pts + 1, int((now - self._start) * VIDEO_CLOCK_RATE))
        if isinstance(frame, bytes):
            # pre-encoded: aiortc's sender only packetises it. Its keyframe requests would be lost (they only
            # reach an encoder it runs itself), so pass them on.
            s = self.sender
            if s is not None and getattr(s, "_RTCRtpSender__force_keyframe", False):
                s._RTCRtpSender__force_keyframe = False  # noqa: SLF001
                self._hub._need_key()  # noqa: SLF001
            vf = av.Packet(frame)
        else:
            # wrap the array without copying (from_ndarray copies the whole frame: ~12 ms at 3072x1536); the
            # frame array is never written again after publishing
            vf = av.VideoFrame.from_numpy_buffer(frame, format="rgba" if frame.shape[2] == 4 else "rgb24")
        vf.pts = self._pts
        vf.time_base = fractions.Fraction(1, VIDEO_CLOCK_RATE)
        PERF.add("track.convert", (time.perf_counter() - t0) * 1e3)
        if meta is not None and self.on_meta is not None:
            # the RTP timestamp of this frame will be (random origin + pts) -> the client can match it
            self.on_meta(self._pts, meta)
        return vf

    def stop(self) -> None:
        self._hub.unsubscribe(self._queue)
        super().stop()


class Peer:
    def __init__(self, peer_id: str, session: Session, pc: RTCPeerConnection) -> None:
        self.peer_id = peer_id
        self.session = session
        self.pc = pc
        self.events = None  # reliable DataChannel
        self.teleop = None  # unreliable DataChannel
        self.motion_format: str | None = None  # "json" / "vmc", whatever the XR samples last came as
        self.tasks: list[asyncio.Task] = []
        self.track: SimVideoTrack | None = None


class PeerManager:
    def __init__(
        self,
        *,
        auth: AuthManager,
        frames: FrameHub,
        state: StateHub,
        teleop: TeleopInput,
        commands: CommandQueue,
        fps: int,
        state_hz: float,
        max_peers: int,
        video_codec: str = "auto",
    ) -> None:
        self.video_codec = video_codec
        self.auth = auth
        self.frames = frames
        self.state = state
        self.teleop = teleop
        self.commands = commands
        self.fps = fps
        self.state_hz = state_hz
        self.max_peers = max_peers
        self.peers: dict[str, Peer] = {}
        self._controller: str | None = None  # peer that currently owns the arm
        self._last_xr: dict[str, float] = {}  # peer -> time of its last XR sample

    @property
    def controller(self) -> str | None:
        return self._controller

    async def handle_offer(self, session: Session, sdp: str, sdp_type: str) -> dict:
        loop = asyncio.get_running_loop()
        # One peer per session: a reconnect replaces the old connection
        for p in [p for p in self.peers.values() if p.session.session_id == session.session_id]:
            await self.close_peer(p.peer_id)
        if len(self.peers) >= self.max_peers:
            raise RuntimeError("Too many connected peers")

        pc = RTCPeerConnection()
        peer = Peer(uuid.uuid4().hex[:12], session, pc)
        self.peers[peer.peer_id] = peer
        self.auth.set_peer_count(session.session_id, +1)

        @pc.on("connectionstatechange")
        async def on_state() -> None:
            log.info("peer %s connection state: %s", peer.peer_id, pc.connectionState)
            if pc.connectionState in ("failed", "closed"):
                await self.close_peer(peer.peer_id)

        @pc.on("datachannel")
        def on_datachannel(channel) -> None:
            if channel.label == "teleop":
                peer.teleop = channel
            elif channel.label == "events":
                peer.events = channel
                peer.tasks.append(loop.create_task(self._state_loop(peer)))

            @channel.on("message")
            def on_message(message) -> None:
                self._on_message(peer, channel, message)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=sdp_type))
        # aiortc sends with the first common codec in the offer's order (browsers list VP8 first)
        prefer_video_codec(pc, self.video_codec)

        # H.264 negotiated and the GPU encoder running: this peer gets the shared encoder's packets
        kind = "h264" if self.frames.gpu_h264 and _negotiated_mime(pc) == "video/h264" else "raw"
        peer.track = SimVideoTrack(self.frames, loop, self.fps, kind)

        def send_frame_meta(pts: int, meta: dict, p: Peer = peer) -> None:
            if p.events is not None:
                _safe_send(p.events, json.dumps({"t": "fmeta", "pts": pts, **meta}))

        peer.track.on_meta = send_frame_meta
        peer.track.sender = pc.addTrack(peer.track)
        log.info("peer %s video: %s (%s)", peer.peer_id, _negotiated_mime(pc), "GPU encoder" if kind == "h264" else "per-peer encoder")

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        if self._controller is None:
            self._controller = peer.peer_id
        return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "peerId": peer.peer_id}

    def _on_message(self, peer: Peer, channel, message) -> None:
        if isinstance(message, (bytes, bytearray)):
            # binary: XR samples as VMC Protocol (OSC) bundles (the web app's "VMC" motion format)
            try:
                msg = decode_vmc(bytes(message))
            except (ValueError, struct.error) as exc:
                log.debug("peer %s: bad VMC packet (%s)", peer.peer_id, exc)
                return
            if msg is None:
                return
        elif isinstance(message, str):
            try:
                msg = json.loads(message)
            except ValueError:
                return
            if not isinstance(msg, dict):
                return
        else:
            return
        t = msg.get("t")
        if t == "xr":
            fmt = msg.pop("fmt", "json")
            if peer.motion_format != fmt:
                peer.motion_format = fmt
                log.info("peer %s motion format: %s", peer.peer_id, "VMC Protocol (OSC)" if fmt == "vmc" else "JSON")
            now = time.monotonic()
            self._last_xr[peer.peer_id] = now
            owner = self._controller
            # Hand the arm over when the owner is gone (page reloads create and close several peers within a
            # second, which could leave the ownership on a closed peer) or has stopped sending XR samples.
            if owner is None or owner not in self.peers or now - self._last_xr.get(owner, 0.0) > 1.0:
                if owner != peer.peer_id:
                    log.info("arm control -> peer %s (%s)", peer.peer_id, peer.session.device_name)
                self._controller = peer.peer_id
            if self._controller == peer.peer_id:
                msg["_rx"] = time.time() * 1000  # wall clock at arrival (latency diagnostics)
                self.teleop.update(peer.peer_id, msg)
        elif t == "ping":
            channel.send(json.dumps({"t": "pong", "id": msg.get("id"), "ts": msg.get("ts"), "server": time.time()}))
        elif t == "perf":
            # the web app's receive-side numbers (decode, jitter buffer, pose -> display)
            data = msg.get("data")
            if isinstance(data, dict) and len(json.dumps(data)) < 4000:
                PERF.set_client({"device": peer.session.device_name, **data})
        elif t == "take_control":
            self._controller = peer.peer_id
            self.teleop.clear()
        elif t == "cmd":
            # Commands over the DataChannel (used by the in-VR buttons): same set as the REST API
            from .http_server import ALLOWED_COMMANDS

            name = str(msg.get("name", ""))
            req_id = msg.get("id")
            if name not in ALLOWED_COMMANDS:
                _safe_send(channel, json.dumps({"t": "cmd_result", "id": req_id, "name": name, "ok": False, "error": "unknown command"}))
                return
            fut = self.commands.submit(name, msg.get("payload") or {})

            def done(f, ch=channel, rid=req_id, n=name) -> None:
                try:
                    res = {"t": "cmd_result", "id": rid, "name": n, "ok": True, "result": f.result()}
                except Exception as exc:  # noqa: BLE001
                    res = {"t": "cmd_result", "id": rid, "name": n, "ok": False, "error": str(exc)}
                try:
                    self._loop.call_soon_threadsafe(_safe_send, ch, json.dumps(res))
                except Exception:  # noqa: BLE001
                    pass

            fut.add_done_callback(done)

    async def _state_loop(self, peer: Peer) -> None:
        period = 1.0 / max(self.state_hz, 1.0)
        while peer.peer_id in self.peers:
            ch = peer.events
            if ch is not None and ch.readyState == "open":
                st = self.state.get()
                st["t"] = "state"
                st["controller"] = self._controller == peer.peer_id
                st["peerId"] = peer.peer_id
                _safe_send(ch, json.dumps(st))
            await asyncio.sleep(period)

    async def close_peer(self, peer_id: str) -> None:
        peer = self.peers.pop(peer_id, None)
        if peer is None:
            return
        for t in peer.tasks:
            t.cancel()
        if peer.track:
            peer.track.stop()
        self.auth.set_peer_count(peer.session.session_id, -1)
        self._last_xr.pop(peer_id, None)
        if self._controller == peer_id:
            self._controller = next(iter(self.peers), None)
            self.teleop.clear(peer_id)
        await peer.pc.close()

    async def close_session(self, session_id: str) -> None:
        for p in [p for p in self.peers.values() if p.session.session_id == session_id]:
            await self.close_peer(p.peer_id)

    async def close_all(self) -> None:
        for pid in list(self.peers):
            await self.close_peer(pid)

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop


def _safe_send(channel, data: str) -> None:
    try:
        if channel.readyState == "open":
            channel.send(data)
    except Exception:  # noqa: BLE001
        pass
