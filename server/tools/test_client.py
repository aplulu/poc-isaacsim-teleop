"""End-to-end smoke test for the teleop server (pairing -> JWT -> WebRTC video + teleop DataChannel).

    C:\\isaacsim\\python.bat server\\tools\\test_client.py --url https://127.0.0.1:8443 --log <server log file>

The pairing code is read from the server log (``--headless`` servers print it), or typed interactively.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import ssl
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "_deps"))

import aiohttp  # noqa: E402
import numpy as np  # noqa: E402
from aiortc import RTCPeerConnection, RTCSessionDescription  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="https://127.0.0.1:8443")
parser.add_argument("--log", default=None, help="server log file to scrape the pairing code from")
parser.add_argument("--out", default=os.path.join(HERE, "_out"))
args = parser.parse_args()


def read_code(log_path: str, device: str, timeout: float = 10) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout:
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            codes = re.findall(rf"PAIRING CODE for {re.escape(device)} .*?: (\d{{6}})", f.read())
        if codes:
            return codes[-1]
        time.sleep(0.3)
    raise RuntimeError("pairing code not found in log")


def save_png(path: str, img: np.ndarray) -> None:
    try:
        from PIL import Image

        Image.fromarray(img).save(path)
    except ImportError:
        np.save(path + ".npy", img)


async def main() -> None:
    os.makedirs(args.out, exist_ok=True)
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    conn = aiohttp.TCPConnector(ssl=ssl_ctx)
    async with aiohttp.ClientSession(connector=conn) as http:
        base = args.url.rstrip("/")
        info = await (await http.get(f"{base}/api/info")).json()
        print("info:", info)

        r = await http.get(f"{base}/api/state")
        assert r.status == 401, f"unauthenticated state must be 401, got {r.status}"
        r = await http.get(f"{base}/api/state", headers={"Authorization": "Bearer abc.def.ghi"})
        assert r.status == 401, "forged token must be rejected"
        print("unauthenticated access rejected: OK")

        device = f"test-client-{int(time.time())}"
        req = await (await http.post(f"{base}/api/pair/request", json={"deviceName": device})).json()
        print("pair request:", req)
        bad = await http.post(f"{base}/api/pair/confirm", json={"requestId": req["requestId"], "code": "000000x"})
        print("wrong code ->", bad.status, await bad.json())
        code = read_code(args.log, device) if args.log else input("code: ").strip()
        res = await http.post(f"{base}/api/pair/confirm", json={"requestId": req["requestId"], "code": code})
        body = await res.json()
        assert res.status == 200, body
        token = body["token"]
        auth = {"Authorization": f"Bearer {token}"}
        print("paired, JWT issued:", token[:24] + "...")

        print("session:", await (await http.get(f"{base}/api/session", headers=auth)).json())
        dr = await (await http.get(f"{base}/api/dr/config", headers=auth)).json()
        print("dr config:", json.dumps(dr)[:200])

        # -- WebRTC ----------------------------------------------------------------------------
        pc = RTCPeerConnection()
        pc.addTransceiver("video", direction="recvonly")
        teleop = pc.createDataChannel("teleop", ordered=False, maxRetransmits=0)
        events = pc.createDataChannel("events")
        states: list[dict] = []
        frames: list[np.ndarray] = []

        @events.on("message")
        def on_ev(msg):
            m = json.loads(msg)
            if m.get("t") == "state":
                states.append(m)

        @pc.on("track")
        def on_track(track):
            async def pull():
                while True:
                    try:
                        f = await track.recv()
                    except Exception:  # noqa: BLE001
                        return
                    frames.append(f.to_ndarray(format="rgb24"))
                    if len(frames) > 5:
                        frames.pop(0)

            asyncio.ensure_future(pull())

        await pc.setLocalDescription(await pc.createOffer())
        r = await http.post(
            f"{base}/api/webrtc/offer", json={"sdp": pc.localDescription.sdp, "type": "offer"}, headers=auth
        )
        ans = await r.json()
        assert r.status == 200, ans
        await pc.setRemoteDescription(RTCSessionDescription(ans["sdp"], ans["type"]))
        for _ in range(200):
            if frames and states and teleop.readyState == "open":
                break
            await asyncio.sleep(0.1)
        assert frames, "no video frames received"
        print("video frame:", frames[-1].shape, "codec negotiated; states:", len(states))
        save_png(os.path.join(args.out, "frame_initial.png"), frames[-1])
        s0 = states[-1]
        print("initial state:", {k: s0.get(k) for k in ("joints", "tcp", "fps", "camera", "hfov")})

        # -- teleop: clutch + move the controller 15 cm to the right and 10 cm down ---------------
        def xr(p, grip, trigger=0.0):
            return json.dumps(
                {"t": "xr", "right": {"p": p, "q": [0, 0, 0, 1], "grip": grip, "trigger": trigger}, "ts": time.time()}
            )

        p0 = np.array([0.2, 1.2, -0.4])
        for i in range(60):
            teleop.send(xr(p0.tolist(), 1.0))
            await asyncio.sleep(1 / 60)
        for i in range(120):
            a = i / 119
            teleop.send(xr((p0 + a * np.array([0.15, -0.10, 0.0])).tolist(), 1.0))
            await asyncio.sleep(1 / 60)
        for i in range(90):
            teleop.send(xr((p0 + np.array([0.15, -0.10, 0.0])).tolist(), 1.0))
            await asyncio.sleep(1 / 60)
        teleop.send(xr((p0 + np.array([0.15, -0.10, 0.0])).tolist(), 0.0))
        await asyncio.sleep(0.3)
        s1 = states[-1]
        d = np.array(s1["tcp"]["p"]) - np.array(s0["tcp"]["p"])
        print("tcp moved by", np.round(d, 3), "target error", s1.get("targetError"), "joints", s1.get("joints"))
        save_png(os.path.join(args.out, "frame_after_move.png"), frames[-1])

        # -- commands ------------------------------------------------------------------------------
        for name, payload in [
            ("set_dr_config", {"config": {"materials": {"enabled": True}, "camera": {"enabled": True}}}),
            ("randomize", {}),
        ]:
            r = await http.post(f"{base}/api/command", json={"name": name, "payload": payload}, headers=auth)
            print(name, "->", r.status, (await r.text())[:200])
        await asyncio.sleep(1.5)
        save_png(os.path.join(args.out, "frame_randomized.png"), frames[-1])
        for cam in ("front", "wrist", "top"):
            r = await http.post(f"{base}/api/camera", json={"camera": cam}, headers=auth)
            print("camera", cam, "->", r.status, await r.text())
            await asyncio.sleep(3.0)
            save_png(os.path.join(args.out, f"frame_{cam}.png"), frames[-1])
        r = await http.post(f"{base}/api/camera", json={"camera": "operator"}, headers=auth)
        r = await http.post(f"{base}/api/reset", json={"restoreAppearance": True}, headers=auth)
        print("reset ->", r.status, await r.text())
        await asyncio.sleep(1.0)
        save_png(os.path.join(args.out, "frame_reset.png"), frames[-1])
        print("final state:", {k: states[-1].get(k) for k in ("joints", "tcp", "fps", "gripper")})

        r = await http.post(f"{base}/api/session/logout", headers=auth)
        print("logout ->", r.status)
        r = await http.get(f"{base}/api/state", headers=auth)
        assert r.status == 401, "token must be invalid after logout"
        print("token rejected after logout: OK")
        await pc.close()


asyncio.run(main())
