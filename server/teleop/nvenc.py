"""Video codec choice for the WebRTC stream: H.264 on the GPU (NVENC) by default.

aiortc encodes on the CPU (libvpx VP8 or libx264). The VP8 encoder took ~11 ms per stereo frame and capped the
stream at ~18-19 fps. PyAV's bundled FFmpeg includes ``h264_nvenc``: the GPU encodes, and it takes the RGBA frames
directly (the colour conversion also happens on the GPU), so the encoder thread mostly waits.

* ``NvencH264Encoder`` replaces aiortc's H.264 encoder; if NVENC cannot open (no NVIDIA GPU / driver, too many
  sessions) it falls back to libx264 for the rest of the run.
* ``prefer_video_codec`` moves H.264 to the front of the negotiated codecs: aiortc sends with the first codec of
  the offer that it supports, and browsers list VP8 first.
"""

from __future__ import annotations

import fractions
import logging

import av
from aiortc import codecs as aiortc_codecs
from aiortc.codecs import h264

log = logging.getLogger("teleop.nvenc")

_nvenc_ok: bool | None = None  # None until the first encoder tried to open NVENC


class NvencH264Encoder(h264.H264Encoder):
    perf_name = "NVENC H.264"

    # aiortc feeds the receiver's REMB estimate in here. With a light stream that estimate starts low and hardly
    # grows (aiortc does not probe), and every >10 % change reopened the encoder (a keyframe + hitch). On the LAN
    # the cap is the configured bitrate (scaled with the frame size, see Sim._apply_bitrate); quality-driven VBR
    # below keeps the actual rate far under it for a mostly static scene.
    @property
    def target_bitrate(self) -> int:
        return h264.MAX_BITRATE

    @target_bitrate.setter
    def target_bitrate(self, bitrate: int) -> None:
        pass

    def _open_nvenc(self, frame: av.VideoFrame):
        cc = av.CodecContext.create("h264_nvenc", "w")
        cc.width = frame.width
        cc.height = frame.height
        # NVENC converts RGB(A) to YUV 4:2:0 itself; anything else goes through yuv420p
        cc.pix_fmt = "rgba" if frame.format.name == "rgba" else "yuv420p"
        cc.bit_rate = self.target_bitrate
        cc.framerate = fractions.Fraction(h264.MAX_FRAME_RATE, 1)
        cc.time_base = fractions.Fraction(1, h264.MAX_FRAME_RATE)
        cc.gop_size = h264.MAX_FRAME_RATE * 10  # keyframes on request (PLI / FIR) and every 10 s
        cc.max_b_frames = 0
        rate = self.target_bitrate
        self._open_rate = rate  # (PyAV reads bit_rate back as None in VBR mode)
        cc.options = {
            "preset": "p1",  # fastest
            "tune": "ull",  # ultra low latency: no lookahead, no reordering
            # constant quality, capped: ~45 dB PSNR at a few Mbit/s for this scene, never above the cap
            "rc": "vbr",
            "cq": "19",
            "maxrate": str(rate),
            "bufsize": str(rate // h264.MAX_FRAME_RATE * 2),  # ~2 frames of VBV: no frame much larger than that
            "zerolatency": "1",
            "delay": "0",
            "forced-idr": "1",  # a forced keyframe is an IDR frame (decodable on its own)
            "profile": "baseline",  # what aiortc signals (profile-level-id 42e01f / 42001f)
        }
        cc.open()
        return cc

    def _encode_frame(self, frame: av.VideoFrame, force_keyframe: bool):
        global _nvenc_ok
        if _nvenc_ok is False:
            self.perf_name = "libx264"
            yield from super()._encode_frame(frame, force_keyframe)
            return
        if self.codec and (
            frame.width != self.codec.width
            or frame.height != self.codec.height
            or abs(self.target_bitrate - self._open_rate) / self._open_rate > 0.1
        ):
            self.codec = None  # resolution / bitrate cap changed: reopen
        if self.codec is None:
            try:
                self.codec = self._open_nvenc(frame)
                if _nvenc_ok is None:
                    log.info("video: H.264 on the GPU (NVENC), %dx%d, %.1f Mbit/s", frame.width, frame.height,
                             self.target_bitrate / 1e6)
                _nvenc_ok = True
            except Exception as exc:  # noqa: BLE001 - fall back to the CPU encoder
                log.warning("NVENC unavailable (%s); encoding H.264 with libx264 on the CPU", exc)
                _nvenc_ok = False
                self.codec = None
                self.perf_name = "libx264"
                yield from super()._encode_frame(frame, force_keyframe)
                return
        frame.pict_type = av.video.frame.PictureType.I if force_keyframe else av.video.frame.PictureType.NONE
        if frame.format.name != self.codec.pix_fmt:
            frame = frame.reformat(format=self.codec.pix_fmt)
        data = b"".join(bytes(p) for p in self.codec.encode(frame))
        if data:
            yield from self._split_bitstream(data)


def install(codec: str) -> None:
    """codec: "auto" / "nvenc" (H.264, NVENC with libx264 fallback), "x264" (H.264 on the CPU), "vp8"."""
    if codec in ("auto", "nvenc"):
        aiortc_codecs.H264Encoder = NvencH264Encoder  # what aiortc's get_encoder() instantiates
    else:
        aiortc_codecs.H264Encoder = h264.H264Encoder


def prefer_video_codec(pc, codec: str) -> None:
    """Put the preferred codec first among the codecs negotiated with the client (call after
    setRemoteDescription, before createAnswer). aiortc lists the common codecs in the offer's order and sends with
    the first one; browsers offer VP8 first. RTX entries stay right after their base codec."""
    mime = "video/vp8" if codec == "vp8" else "video/h264"
    for t in pc.getTransceivers():
        if t.kind != "video" or not getattr(t, "_codecs", None):
            continue
        cs = list(t._codecs)  # noqa: SLF001 - aiortc has no public API for this after the offer is applied
        base = [c for c in cs if c.mimeType.lower() != "video/rtx"]
        rtx = {c.parameters.get("apt"): c for c in cs if c.mimeType.lower() == "video/rtx"}
        base.sort(key=lambda c: c.mimeType.lower() != mime)  # stable: keeps the offer's order otherwise
        out = []
        for c in base:
            out.append(c)
            if c.payloadType in rtx:
                out.append(rtx[c.payloadType])
        t._codecs = out  # noqa: SLF001
