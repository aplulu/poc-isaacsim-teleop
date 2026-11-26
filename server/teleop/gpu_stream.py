"""Stream frames straight from the GPU: annotator (CUDA) -> side-by-side buffer (Warp kernel) -> NVENC.

The CPU path read every stream image back to host memory (``get_data``), copied the eyes side by side and handed
RGBA to PyAV, which uploaded it again for NVENC: at 2x2048x2048 ~21 ms on the main thread plus ~25 ms in the
encoder thread. Here the image never leaves the GPU:

* main thread: the annotators return Warp arrays on the GPU; a kernel copies the eyes into a ring buffer slot
  (one 32-bit word per pixel), the stream is synchronised and the slot is queued (newest wins),
* encoder thread: PyNvVideoCodec encodes the slot (RGBA = NVENC "ABGR"), ``EndEncode`` after every frame so the
  frame comes out at once (otherwise NVENC holds 3 frames back), and the FFmpeg ``h264_metadata`` bitstream
  filter writes the colour description (NVENC converts with BT.709 limited range but does not signal it, and
  decoders would assume BT.601),
* the H.264 access units go to every video track as packets (aiortc only packetises them). One encoder serves all
  peers, so a keyframe is requested whenever a peer starts, loses packets or sends a PLI.
"""

from __future__ import annotations

import io
import logging
import queue
import sys
import threading
import time

from .perf import PERF

log = logging.getLogger("teleop.gpu_stream")

PIC_FORCE_IDR = 2
PIC_OUTPUT_SPSPPS = 4
RING = 3

_import_error: str | None = None


def available() -> bool:
    """PyNvVideoCodec importable and a CUDA device present (checked once)."""
    global _import_error
    if _import_error is not None:
        return False
    try:
        import PyNvVideoCodec  # noqa: F401
        import warp as wp

        wp.init()
        if not wp.is_cuda_available():
            raise RuntimeError("no CUDA device")
        return True
    except Exception as exc:  # noqa: BLE001
        _import_error = str(exc)
        log.warning("GPU stream encoding unavailable (%s); using the CPU path", exc)
        return False


def _kernels():
    import warp as wp

    @wp.kernel
    def place(src: wp.array2d(dtype=wp.uint32), dst: wp.array2d(dtype=wp.uint32), x0: int):
        i, j = wp.tid()
        dst[i, x0 + j] = src[i, j]

    return place


class _CudaFrame:
    """What PyNvVideoCodec expects for a device input: an object whose cuda() returns the array."""

    def __init__(self, arr) -> None:
        self._arr = arr

    def cuda(self):
        return self._arr


class GpuH264Stream:
    def __init__(self, on_packet, fps: int) -> None:
        """on_packet(data: bytes, keyframe: bool, meta: dict, t_pub: float) is called from the encoder thread."""
        import warp as wp

        self.wp = wp
        self.device = "cuda:0"
        self._place = _kernels()
        self.on_packet = on_packet
        self.fps = fps
        self._ring: list = []
        self._ring_shape: tuple[int, int] | None = None
        self._slot = 0
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._want_key = threading.Event()
        self._want_key.set()
        self._enc = None
        self._enc_key: tuple | None = None
        self._bsf = None
        self._stop = False
        self.bitrate = 8_000_000
        self.frames = 0
        self._thread = threading.Thread(target=self._run, name="gpu-h264", daemon=True)
        self._thread.start()
        wp.load_module(module=sys.modules[__name__], device=self.device)

    # -- main thread -----------------------------------------------------------------------------------
    def request_keyframe(self) -> None:
        self._want_key.set()

    def submit(self, eyes: list, meta: dict) -> bool:
        """eyes: Warp arrays (h, w, 4) uint8 on the GPU, same height. Returns False if they cannot be used."""
        wp = self.wp
        t0 = time.perf_counter()
        h = int(eyes[0].shape[0])
        if any(int(e.shape[0]) != h or e.ndim != 3 or int(e.shape[2]) != 4 for e in eyes):
            return False
        w = sum(int(e.shape[1]) for e in eyes)
        if self._ring_shape != (h, w):
            self._ring = [wp.zeros((h, w), dtype=wp.uint32, device=self.device) for _ in range(RING)]
            self._ring_shape = (h, w)
        dst = self._ring[self._slot]
        x = 0
        for e in eyes:
            ew = int(e.shape[1])
            src = wp.array(ptr=e.ptr, dtype=wp.uint32, shape=(h, ew), strides=(int(e.strides[0]), 4), device=self.device)
            wp.launch(self._place, dim=(h, ew), inputs=[src, dst, x], device=self.device)
            x += ew
        # the encoder thread waits for this event before reading the slot (the main thread does not wait)
        evt = wp.Event(self.device)
        wp.record_event(evt)
        PERF.add("grab.concat", (time.perf_counter() - t0) * 1e3)
        item = (self._slot, evt, meta, time.perf_counter())
        self._slot = (self._slot + 1) % RING
        try:
            self._q.put_nowait(item)
        except queue.Full:  # the encoder is still busy with an older frame: replace it
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put_nowait(item)
        return True

    def close(self) -> None:
        self._stop = True
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass

    # -- encoder thread ----------------------------------------------------------------------------------
    def _encoder(self, h: int, w: int):
        import av
        import PyNvVideoCodec as nvc
        from av.bitstream import BitStreamFilterContext

        rate = int(self.bitrate)
        key = (h, w, rate)
        if self._enc is not None and self._enc_key == key:
            return self._enc
        if self._enc is not None:
            try:
                self._enc.EndEncode()
            except Exception:  # noqa: BLE001
                pass
        self._enc = nvc.CreateEncoder(
            w, h, "ABGR", False,
            codec="h264", profile="baseline", preset="P1", tuning_info="ultra_low_latency",
            # constant quality would be nicer; VBR with a cap keeps static scenes cheap and motion bounded
            rc="vbr", fps=self.fps, bitrate=min(rate, 20_000_000), maxbitrate=rate,
            vbvbufsize=rate // max(self.fps, 1) * 2, gop=self.fps * 10, idrperiod=self.fps * 10, bf=0,
            repeatspspps=1,
        )
        c = av.open(io.BytesIO(), "w", format="h264")
        st = c.add_stream("h264", rate=self.fps)
        st.width, st.height = w, h
        self._bsf = BitStreamFilterContext(
            "h264_metadata=colour_primaries=1:transfer_characteristics=1:matrix_coefficients=1:video_full_range_flag=0", st
        )
        self._bsf_owner = c
        self._enc_key = key
        self._want_key.set()
        log.info("video: H.264 straight from the GPU (NVENC), %dx%d, up to %.1f Mbit/s", w, h, rate / 1e6)
        return self._enc

    def _run(self) -> None:
        import av

        while not self._stop:
            item = self._q.get()
            if item is None:
                return
            slot, evt, meta, t_sub = item
            try:
                h, w = self._ring_shape
                enc = self._encoder(h, w)
                key = self._want_key.is_set()
                if key:
                    self._want_key.clear()
                t0 = time.perf_counter()
                self.wp.synchronize_event(evt)
                out = enc.Encode(_CudaFrame(self._as_rgba(slot)), PIC_FORCE_IDR | PIC_OUTPUT_SPSPPS if key else 0)
                out += enc.EndEncode()
                data = b"".join(o["data"] for o in out)
                if not data:
                    continue
                data = b"".join(bytes(p) for p in self._bsf.filter(av.Packet(data)))
                PERF.add("enc.encode", (time.perf_counter() - t0) * 1e3)
                PERF.count("enc.frames")
                PERF.count("enc.bytes", float(len(data)))
                PERF.encoder = "NVENC H.264 (GPU)"
                self.frames += 1
                self.on_packet(data, key, meta, t_sub)
            except Exception:  # noqa: BLE001 - keep the thread alive; the next frame retries
                log.exception("GPU encode failed")
                self._enc = None
                self._enc_key = None
                time.sleep(0.2)

    def _as_rgba(self, slot: int):
        """The uint32 slot seen as (h, w, 4) uint8, the layout NVENC's ABGR input expects."""
        wp = self.wp
        h, w = self._ring_shape
        a = self._ring[slot]
        return wp.array(ptr=a.ptr, dtype=wp.uint8, shape=(h, w, 4), device=self.device)


def h264_is_keyframe(data: bytes) -> bool:
    """True if the access unit contains an IDR slice (NAL type 5)."""
    i = 0
    n = len(data)
    while i + 3 < n:
        if data[i] == 0 and data[i + 1] == 0 and (data[i + 2] == 1 or (data[i + 2] == 0 and i + 3 < n and data[i + 3] == 1)):
            j = i + (3 if data[i + 2] == 1 else 4)
            if j < n and data[j] & 0x1F == 5:
                return True
            i = j
        else:
            i += 1
    return False

