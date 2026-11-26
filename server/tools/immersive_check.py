"""Check the free-viewpoint (head-tracked stereo) view over the API.

    C:\\isaacsim\\python.bat server\\tools\\immersive_check.py
"""

from __future__ import annotations

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
OUT = os.path.join(os.path.dirname(__file__), "_out")


def yaw_quat(deg: float) -> list[float]:
    """XR rotation about +Y (xyzw)."""
    a = math.radians(deg) / 2
    return [0.0, math.sin(a), 0.0, math.cos(a)]


async def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ctx)) as http:
        dev = f"imm-{int(time.time())}"
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

        pc = RTCPeerConnection()
        pc.addTransceiver("video", direction="recvonly")
        teleop = pc.createDataChannel("teleop", ordered=False, maxRetransmits=0)
        events = pc.createDataChannel("events")
        frames: list[np.ndarray] = []
        metas: list[dict] = []

        @events.on("message")
        def on_ev(msg):
            m = json.loads(msg)
            if m.get("t") == "fmeta":
                metas.append(m)

        @pc.on("track")
        def on_track(track):
            async def pull():
                while True:
                    try:
                        f = await track.recv()
                    except Exception:  # noqa: BLE001
                        return
                    frames.append(f.to_ndarray(format="rgb24"))
                    del frames[:-5]

            asyncio.ensure_future(pull())

        await pc.setLocalDescription(await pc.createOffer())
        ans = await (await http.post(f"{URL}/api/webrtc/offer", json={"sdp": pc.localDescription.sdp, "type": "offer"}, headers=H)).json()
        await pc.setRemoteDescription(RTCSessionDescription(ans["sdp"], ans["type"]))
        for _ in range(100):
            if teleop.readyState == "open":
                break
            await asyncio.sleep(0.1)

        t0 = time.time()
        head = {"p": [0.0, 1.6, 0.0], "q": yaw_quat(0)}

        async def stream(seconds, right=None, left=None):
            for _ in range(int(seconds * 60)):
                msg = {"t": "xr", "ts": (time.time() - t0) * 1000, "head": head}
                if right:
                    msg["right"] = right
                if left:
                    msg["left"] = left
                teleop.send(json.dumps(msg))
                await asyncio.sleep(1 / 60)

        from PIL import Image

        print("set_view ->", await cmd("set_view", {"mode": "immersive", "recenter": True}))
        await stream(2.0)
        metas.clear()
        await stream(2.0)
        img = frames[-1]
        print(f"frame {img.shape[1]}x{img.shape[0]}, fmeta messages in 2 s: {len(metas)}, last: "
              f"pts={metas[-1]['pts'] if metas else None} head={metas[-1]['head'] if metas else None}")
        Image.fromarray(img).save(os.path.join(OUT, "immersive_front.jpg"), quality=90)
        # look 60 deg to the left (XR yaw +60)
        head = {"p": [0.0, 1.6, 0.0], "q": yaw_quat(60)}
        await stream(1.5)
        Image.fromarray(frames[-1]).save(os.path.join(OUT, "immersive_left60.jpg"), quality=90)
        print("head yaw 60 -> meta q", metas[-1]["head"]["q"] if metas else None)
        head = {"p": [0.0, 1.6, 0.0], "q": yaw_quat(0)}
        await stream(1.0)

        # hand mapping: clutch, move the controller 10 cm forward in XR (-z) -> gripper should move along
        # the head's forward direction in the sim (anchor yaw 180 deg => world -X)
        s0 = await state()
        ident = [0, 0, 0, 1]
        p0 = np.array([0.2, 1.2, -0.4])
        hand = lambda p, g: {"p": list(p), "q": ident, "grip": g, "trigger": 0, "axes": [0, 0]}  # noqa: E731
        await stream(0.4, right=hand(p0, 1.0))
        await stream(1.2, right=hand(p0 + [0, 0, -0.1], 1.0))
        await stream(0.2, right=hand(p0 + [0, 0, -0.1], 0.0))
        s1 = await state()
        d = np.array(s1["tcp"]["p"]) - np.array(s0["tcp"]["p"])
        print("hand forward 10 cm -> tcp moved", np.round(d, 3), "(expected ~[-0.1, 0, 0])")
        # locomotion: push the left stick forward for 1 s -> anchor moves 0.8 m along -X
        a0 = s1["immersive"]["anchor"]
        await stream(1.0, left={"p": [-0.2, 1.2, -0.4], "q": ident, "grip": 0, "trigger": 0, "axes": [0, -1]})
        a1 = (await state())["immersive"]["anchor"]
        print("locomotion anchor", a0, "->", a1)
        print("set_view screen ->", await cmd("set_view", {"mode": "screen"}))
        await stream(1.0)
        print("frame back to", frames[-1].shape)
        await http.post(f"{URL}/api/session/logout", headers=H)
        await pc.close()


asyncio.run(main())
