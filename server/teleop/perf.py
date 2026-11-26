"""Video pipeline timings: rolling averages of named sections, shared by the sim thread and the WebRTC loop.

Server sections (ms):
    loop.pre / loop.kit / loop.post   main loop split (kit = physics + rendering of every render product)
    sim.avatar                        avatar IK + skinning + springs
    grab.read                         annotator get_data (GPU -> CPU copy) for all active eyes
    grab.concat                       side-by-side concatenation
    pose.to_publish                   head pose applied -> frame published (render + readback, includes lag)
    track.queue                       published -> taken by the video track
    track.convert                     numpy -> av.VideoFrame
    enc.encode                        H.264 (NVENC) / libx264 / VP8 encode (aiortc's encoder thread)
Counters: frames published / encoded per second, encoded bytes per second.
The web app reports its own numbers (decode, jitter buffer, pose -> display) with a "perf" DataChannel message.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

log = logging.getLogger("teleop.perf")

WINDOW = 8.0  # [s] sections and rates cover this much recent time


class Perf:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sec: dict[str, deque] = {}
        self._cnt: dict[str, deque] = {}  # name -> deque of (t, amount)
        self.client: dict = {}
        self.client_t = 0.0
        self._last_log = time.monotonic()
        self.encoder = ""
        self.tag: str | None = None  # perf probe in progress: main-loop sections are recorded as name@tag

    def add(self, name: str, ms: float) -> None:
        with self._lock:
            d = self._sec.get(name)
            if d is None:
                d = self._sec[name] = deque(maxlen=1000)
            d.append((time.monotonic(), ms))

    def count(self, name: str, amount: float = 1.0) -> None:
        now = time.monotonic()
        with self._lock:
            d = self._cnt.get(name)
            if d is None:
                d = self._cnt[name] = deque(maxlen=2000)
            d.append((now, amount))

    def set_client(self, data: dict) -> None:
        with self._lock:
            self.client = data
            self.client_t = time.monotonic()

    def summary(self) -> dict:
        now = time.monotonic()
        out: dict = {"sections": {}, "rates": {}, "encoder": self.encoder}
        with self._lock:
            for k, d in self._sec.items():
                v = sorted(ms for t, ms in d if now - t <= WINDOW)
                if not v:
                    continue
                out["sections"][k] = {
                    "avg": round(sum(v) / len(v), 2),
                    "p95": round(v[min(len(v) - 1, int(len(v) * 0.95))], 2),
                }
            for k, d in self._cnt.items():
                recent = [a for t, a in d if now - t <= WINDOW]
                out["rates"][k] = round(sum(recent) / WINDOW, 1)
            if self.client and now - self.client_t < 5.0:
                out["client"] = dict(self.client)
        return out

    def maybe_log(self, period: float = 5.0) -> None:
        now = time.monotonic()
        if now - self._last_log < period:
            return
        self._last_log = now
        s = self.summary()
        if not s["sections"]:
            return
        sec = " | ".join(f"{k} {v['avg']:.1f}/{v['p95']:.1f}" for k, v in sorted(s["sections"].items()))
        rates = " ".join(f"{k}={v}" for k, v in sorted(s["rates"].items()))
        log.info("perf avg/p95 ms: %s || %s", sec, rates)
        if "client" in s:
            log.info("perf client: %s", s["client"])


PERF = Perf()


def instrument_encoders() -> None:
    """Time aiortc's encoders (they run in a thread pool) and count the encoded bytes."""
    from aiortc.codecs import h264, vpx

    for cls in (h264.H264Encoder, vpx.Vp8Encoder):
        if getattr(cls, "_perf_wrapped", False):
            continue
        orig = cls.encode

        def encode(self, frame, force_keyframe=False, _orig=orig, _name=cls.__name__):
            t0 = time.perf_counter()
            payloads, ts = _orig(self, frame, force_keyframe)
            PERF.add("enc.encode", (time.perf_counter() - t0) * 1e3)
            PERF.count("enc.frames")
            PERF.count("enc.bytes", float(sum(len(p) for p in payloads)))
            PERF.encoder = getattr(self, "perf_name", _name)
            return payloads, ts

        cls.encode = encode
        cls._perf_wrapped = True
