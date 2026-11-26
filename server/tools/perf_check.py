"""Video pipeline timings per view mode, plus the render-pose lag check of the immersive view.

    C:\\isaacsim\\python.bat server\\main.py --headless --stereo
    C:\\isaacsim\\python.bat server\\tools\\perf_check.py [--seconds 8]

For each scenario: the server's section timings (state "perf") and, measured here, the received fps and the
XR sample -> frame received latency (server + network + encode; the headset adds decode + display).
Lag check: the head yaw steps by 90 deg; frames whose metadata already carries the new pose but whose image still
shows the old view mean the annotator returns an older render than the pose sent with it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import ssl
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "_deps"))

import aiohttp  # noqa: E402
import numpy as np  # noqa: E402
from aiortc import RTCPeerConnection, RTCSessionDescription  # noqa: E402

URL = "https://127.0.0.1:8443"
LOG = os.path.join(os.path.dirname(__file__), "..", "logs", "teleop.log")
ORDER = ["physics.step", "loop.kit", "loop.kit@physics_off", "loop.kit@render_off", "loop.kit@physics_hz_30", "loop.kit@physics_hz_45", "loop.pre", "sim.avatar", "av.ik", "av.springs", "av.write", "av.bodies", "loop.post", "grab.read", "grab.concat", "pose.to_publish",
         "track.queue", "track.convert", "enc.encode"]


def yaw_q(deg: float) -> list[float]:  # rotation about XR +Y (x, y, z, w)
    h = math.radians(deg) / 2
    return [0.0, math.sin(h), 0.0, math.cos(h)]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--kit-trace", choices=["immersive", "immersive-noavatar", "mono", "stereo"],
                    help="server started with --kit-trace: set up this view, capture --trace-frames frames, stop")
    ap.add_argument("--trace-frames", type=int, default=60)
    ap.add_argument("--immersive-size", type=int, default=0, help="per-eye size for the immersive scenarios")
    ap.add_argument("--dlss", default="", help="performance / balanced / quality / auto / native")
    ap.add_argument("--only", default="", help="comma separated: mono, stereo, immersive, immersive-noavatar, lag")
    args = ap.parse_args()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ctx)) as http:
        dev = f"perf-{int(time.time())}"
        rid = (await (await http.post(f"{URL}/api/pair/request", json={"deviceName": dev})).json())["requestId"]
        await asyncio.sleep(0.3)
        code = re.findall(rf"PAIRING CODE for {dev} .*?: (\d{{6}})", open(LOG, encoding="utf-8").read())[-1]
        token = (await (await http.post(f"{URL}/api/pair/confirm", json={"requestId": rid, "code": code})).json())["token"]
        H = {"Authorization": f"Bearer {token}"}

        async def cmd(name, payload=None):
            r = await http.post(f"{URL}/api/command", json={"name": name, "payload": payload or {}}, headers=H)
            return (await r.json()).get("result")

        async def state():
            return await (await http.get(f"{URL}/api/state", headers=H)).json()

        t0 = time.time()

        def now_ms() -> float:
            return (time.time() - t0) * 1000

        t0_ms = t0 * 1000  # server stamps (srx, spub) are wall-clock ms on this same machine

        pc = RTCPeerConnection()
        pc.addTransceiver("video", direction="recvonly")
        metas: dict[int, dict] = {}
        frames: list[tuple[float, int, np.ndarray | None]] = []  # (recv ms, rtp ts, small image)
        keep_images = {"on": False}

        @pc.on("track")
        def on_track(track):
            async def pull():
                while True:
                    try:
                        f = await track.recv()
                    except Exception:  # noqa: BLE001
                        return
                    img = None
                    if keep_images["on"]:
                        a = f.to_ndarray(format="rgb24")
                        img = a[::8, ::8].astype(np.float32)
                    frames.append((now_ms(), int(f.pts), img))

            asyncio.ensure_future(pull())

        teleop = pc.createDataChannel("teleop", ordered=False, maxRetransmits=0)
        events = pc.createDataChannel("events")

        @events.on("message")
        def on_event(m):
            try:
                j = json.loads(m)
            except ValueError:
                return
            if j.get("t") == "fmeta":
                metas[int(j["pts"])] = j

        await pc.setLocalDescription(await pc.createOffer())
        ans = await (await http.post(f"{URL}/api/webrtc/offer", json={"sdp": pc.localDescription.sdp, "type": "offer"}, headers=H)).json()
        await pc.setRemoteDescription(RTCSessionDescription(ans["sdp"], ans["type"]))
        for _ in range(100):
            if teleop.readyState == "open":
                break
            await asyncio.sleep(0.1)

        ident = [0, 0, 0, 1]
        pose = {"head": {"p": [0, 1.55, 0], "q": ident},
                "left": {"p": [-0.2, 1.0, -0.3], "q": ident, "trigger": 0, "grip": 0, "axes": [0, 0]},
                "right": {"p": [0.2, 1.0, -0.3], "q": ident, "trigger": 0, "grip": 0, "axes": [0, 0]}}
        sway = {"on": True, "yaw": 0.0}

        async def sender():
            k = 0
            while True:
                if sway["on"]:  # slow head sway so every frame differs a little
                    pose["head"]["q"] = yaw_q(10 * math.sin(k / 60))
                else:
                    pose["head"]["q"] = yaw_q(sway["yaw"])
                if teleop.bufferedAmount < 2048:  # like the web app: drop samples while the channel is backed up
                    teleop.send(json.dumps({"t": "xr", "ts": now_ms(), **pose}))
                k += 1
                await asyncio.sleep(1 / 72)

        task = asyncio.ensure_future(sender())

        def origin() -> int | None:
            votes: dict[int, int] = {}
            pts = sorted(metas)[-60:]
            for _, rtp, _ in frames[-30:]:
                for p in pts:
                    o = (rtp - p) % 2**32
                    votes[o] = votes.get(o, 0) + 1
            if not votes:
                return None
            o, n = max(votes.items(), key=lambda kv: kv[1])
            return o if n >= 5 else None

        async def scenario(label: str) -> None:
            await asyncio.sleep(2.0)  # settle (render product warm-up, encoder bitrate ramp)
            frames.clear()
            start = now_ms()
            await asyncio.sleep(args.seconds)
            st = await state()
            perf = st.get("perf", {})
            fps_rx = len([f for f in frames if f[0] >= start]) / args.seconds
            measured = list(frames)
            await cmd("perf_probe", {"mode": "render_off", "frames": 30})
            await asyncio.sleep(3.0)
            await cmd("perf_probe", {"mode": "physics_off", "frames": 30})
            await asyncio.sleep(3.0)
            if os.environ.get("PERF_HZ"):
                for hz in (30, 45):
                    await cmd("perf_probe", {"mode": f"physics_hz_{hz}", "frames": 40})
                    await asyncio.sleep(4.0)
            probe = (await state()).get("perf", {}).get("sections", {})
            perf.setdefault("sections", {}).update({k: v for k, v in probe.items() if "@" in k})
            lat, seg = [], []
            o = origin()
            if os.environ.get("PERF_DEBUG"):
                print("debug rtp", [f[1] for f in measured[-5:]], "meta pts", sorted(metas)[-5:], "origin", o)
            if o is not None:
                for t_rx, rtp, _ in measured:
                    m = metas.get((rtp - o) % 2**32)
                    if m and m.get("ts") is not None:
                        lat.append(t_rx - m["ts"])
                        if m.get("srx") and m.get("spub"):
                            seg.append((m["srx"] - t0_ms - m["ts"], m["spub"] - m["srx"], t_rx + t0_ms - m["spub"]))
            print(f"\n=== {label} ===")
            print(f"sim {st.get('fps')} fps | published {perf.get('rates', {}).get('frames.published')} fps | "
                  f"encoded {perf.get('rates', {}).get('enc.frames')} fps ({perf.get('encoder')}) | received {fps_rx:.1f} fps | "
                  f"{perf.get('rates', {}).get('enc.bytes', 0) * 8 / 1e6:.1f} Mbps | physics substeps "
                  f"{perf.get('rates', {}).get('physics.steps')}/s")
            for k in ORDER:
                v = perf.get("sections", {}).get(k)
                if v:
                    print(f"  {k:16s} avg {v['avg']:6.1f}  p95 {v['p95']:6.1f} ms")
            if lat:
                a = np.array(lat)
                print(f"  XR sample -> frame received here: avg {a.mean():.0f}  p50 {np.median(a):.0f}  p95 {np.percentile(a, 95):.0f} ms")
            if seg:
                b = np.array(seg)
                print(f"    = sample sent -> arrived at server {b[:, 0].mean():.0f} ms + arrived -> frame published "
                      f"{b[:, 1].mean():.0f} ms + published -> received here {b[:, 2].mean():.0f} ms")

        if args.kit_trace:
            v = args.kit_trace
            await cmd("set_stereo", {"enabled": v == "stereo"})
            await cmd("set_camera", {"camera": "operator"})
            await cmd("set_view", {"mode": "immersive" if v.startswith("immersive") else "screen", "recenter": True})
            if (await state()).get("avatar"):
                await cmd("set_avatar", {"visible": v != "immersive-noavatar"})
            await asyncio.sleep(5.0)
            st = await state()
            print(f"tracing {v}: sim {st.get('fps')} fps", flush=True)
            await cmd("perf_probe", {"mode": "kit_trace", "frames": args.trace_frames})
            for _ in range(120):  # the server stops after the capture
                await asyncio.sleep(1.0)
                try:
                    await asyncio.wait_for(state(), 2.0)
                except Exception:  # noqa: BLE001
                    break
            print("server stopped (trace written)", flush=True)
            task.cancel()
            return

        only = set(filter(None, args.only.split(",")))

        def want(k: str) -> bool:
            return not only or k in only

        # 1. monitor view, mono 1280x720
        await cmd("set_view", {"mode": "screen"})
        await cmd("set_stereo", {"enabled": False})
        await cmd("set_camera", {"camera": "operator"})
        if want("mono"):
            await scenario("monitor mono (operator camera)")
        # 2. monitor view, stereo
        await cmd("set_stereo", {"enabled": True})
        if want("stereo"):
            await scenario("monitor stereo")
        # 3. immersive with the avatar
        if args.immersive_size:
            print("immersive size:", await cmd("set_resolution", {"immersive": args.immersive_size}))
        if args.dlss:
            print("render:", (await cmd("set_resolution", {"dlss": args.dlss})).get("dlss"))
        await cmd("set_view", {"mode": "immersive", "recenter": True})
        av = (await state()).get("avatar")
        if av:
            await cmd("set_avatar", {"visible": True})
        if want("immersive"):
            await scenario("immersive (avatar shown)")
        # 4. immersive without the avatar
        if av and want("immersive-noavatar"):
            await cmd("set_avatar", {"visible": False})
            await scenario("immersive (avatar hidden)")
            await cmd("set_avatar", {"visible": True})
        if not want("lag"):
            task.cancel()
            await cmd("set_view", {"mode": "screen"})
            await pc.close()
            return

        # 5. render-pose lag: hold yaw 0, then step to 90 deg
        sway["on"] = False
        sway["yaw"] = 0.0
        keep_images["on"] = True
        await asyncio.sleep(2.0)
        o = origin()
        ref_a = np.mean([f[2] for f in frames[-5:] if f[2] is not None], axis=0)
        sway["yaw"] = 90.0
        await asyncio.sleep(2.0)
        ref_b = np.mean([f[2] for f in frames[-5:] if f[2] is not None], axis=0)
        sway["yaw"] = 0.0
        await asyncio.sleep(2.0)
        frames.clear()
        step_t = now_ms()
        sway["yaw"] = 90.0
        await asyncio.sleep(1.5)
        keep_images["on"] = False
        o = origin() or o
        print("\n=== render-pose lag (yaw 0 -> 90 deg) ===")
        if o is None:
            print("  could not match frames to metadata")
        else:
            stale, rows = 0, []
            for t_rx, rtp, img in frames:
                m = metas.get((rtp - o) % 2**32)
                if m is None or img is None or not m.get("head"):
                    continue
                meta_b = abs(m["head"]["q"][1]) > 0.5
                da, db = float(np.abs(img - ref_a).mean()), float(np.abs(img - ref_b).mean())
                img_b = db < da
                rows.append((round(t_rx - step_t), meta_b, img_b, round(da, 1), round(db, 1)))
                if meta_b != img_b:
                    stale += 1
            first_meta = next((r[0] for r in rows if r[1]), None)
            first_img = next((r[0] for r in rows if r[2]), None)
            print(f"  frames whose image does not match the pose in their metadata: {stale}")
            print("  (ms after step, meta new?, image new?, diff to old, diff to new):", rows[:12])
            print(f"  first frame with new metadata at +{first_meta:.0f} ms, first new image at +{first_img:.0f} ms"
                  if first_meta is not None and first_img is not None else f"  rows {rows[:10]}")
        task.cancel()
        await cmd("set_view", {"mode": "screen"})
        await http.post(f"{URL}/api/session/logout", headers=H)
        await pc.close()


asyncio.run(main())
sys.stdout.flush()
os._exit(0)  # aiortc can leave threads behind that keep the process alive
