"""Thread-safe plumbing between the Kit main thread (simulation) and the network thread (asyncio)."""

from __future__ import annotations

import asyncio
import concurrent.futures
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Command:
    name: str
    payload: dict
    future: concurrent.futures.Future


class CommandQueue:
    """Commands from HTTP/WebRTC handlers executed on the simulation thread."""

    def __init__(self) -> None:
        self._q: queue.Queue[Command] = queue.Queue()

    def submit(self, name: str, payload: dict | None = None) -> concurrent.futures.Future:
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._q.put(Command(name, payload or {}, fut))
        return fut

    def drain(self, max_items: int = 32) -> list[Command]:
        out = []
        for _ in range(max_items):
            try:
                out.append(self._q.get_nowait())
            except queue.Empty:
                break
        return out


class TeleopInput:
    """Latest XR controller/head sample (overwritten at the headset's frame rate)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sample: dict | None = None
        self._stamp = 0.0
        self._owner: str | None = None

    def update(self, owner: str, sample: dict) -> None:
        with self._lock:
            self._sample = sample
            self._stamp = time.monotonic()
            self._owner = owner

    def latest(self, max_age: float = 0.5) -> tuple[str | None, dict | None]:
        with self._lock:
            if self._sample is None or time.monotonic() - self._stamp > max_age:
                return self._owner, None
            return self._owner, self._sample

    def info(self) -> dict:
        """Diagnostics for the UI: age of the last sample and the control hand's grip/trigger values."""
        with self._lock:
            if self._sample is None:
                return {"age": None}
            return {"age": round(time.monotonic() - self._stamp, 2), "sample": self._sample}

    def clear(self, owner: str | None = None) -> None:
        with self._lock:
            if owner is None or owner == self._owner:
                self._sample = None


class _Sub:
    __slots__ = ("loop", "q", "kind", "need_key")

    def __init__(self, loop, q, kind) -> None:
        self.loop, self.q, self.kind = loop, q, kind
        self.need_key = True  # H.264 packets: wait for a keyframe before forwarding


class FrameHub:
    """Distributes the newest frame to every video track without ever blocking the sim.

    Two kinds of subscribers: "raw" tracks get RGBA arrays (latest only, older ones are dropped) and encode them
    themselves; "h264" tracks get the access units of the shared GPU encoder (gpu_stream.py). Those must arrive in
    order: when one has to be dropped, the track skips ahead to the next keyframe and one is requested
    (``on_need_key``)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: list[_Sub] = []
        self.latest: np.ndarray | None = None
        self.seq = 0
        self.snapshot_wanted = False  # set by /api/snapshot to render a frame without a video peer
        self.on_need_key = None  # callable() -> None, set by the GPU encoder
        self.gpu_h264 = False  # the shared GPU encoder runs (H.264 peers get packets)

    def subscribe(self, loop: asyncio.AbstractEventLoop, kind: str = "raw") -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1 if kind == "raw" else 4)
        with self._lock:
            self._subs.append(_Sub(loop, q, kind))
        if kind == "h264":
            self._need_key()
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subs = [s for s in self._subs if s.q is not q]

    @property
    def has_subscribers(self) -> bool:
        with self._lock:
            return bool(self._subs)

    def has_kind(self, kind: str) -> bool:
        with self._lock:
            return any(s.kind == kind for s in self._subs)

    def _need_key(self) -> None:
        cb = self.on_need_key
        if cb is not None:
            cb()

    def set_latest(self, frame: np.ndarray) -> None:
        """A CPU copy for /api/snapshot only (not sent to the tracks)."""
        self.latest = frame
        self.seq += 1

    def publish(self, frame: np.ndarray, meta: dict | None = None) -> None:
        """RGBA frame for the raw tracks. ``meta`` travels with it (e.g. the head pose it was rendered with)."""
        self.latest = frame
        self.seq += 1
        with self._lock:
            subs = [s for s in self._subs if s.kind == "raw"]
        for s in subs:
            s.loop.call_soon_threadsafe(_put_latest, s.q, (frame, meta, time.perf_counter()))

    def publish_packet(self, data: bytes, keyframe: bool, meta: dict | None, t_pub: float) -> None:
        """H.264 access unit for the h264 tracks (called from the encoder thread)."""
        with self._lock:
            subs = [s for s in self._subs if s.kind == "h264"]
        for s in subs:
            if s.need_key and not keyframe:
                continue
            s.need_key = False
            s.loop.call_soon_threadsafe(self._put_packet, s, (data, meta, t_pub))

    def _put_packet(self, s: _Sub, item: Any) -> None:
        if s.q.full():  # the track fell behind: drop what is queued and restart at the next keyframe
            while not s.q.empty():
                s.q.get_nowait()
            s.need_key = True
            self._need_key()
            return
        s.q.put_nowait(item)


def _put_latest(q: asyncio.Queue, item: Any) -> None:
    if q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
    q.put_nowait(item)


class StateHub:
    """Latest simulation state snapshot (JSON-serializable dict)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: dict = {}

    def set(self, state: dict) -> None:
        with self._lock:
            self._state = state

    def get(self) -> dict:
        with self._lock:
            return dict(self._state)
