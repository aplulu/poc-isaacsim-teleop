"""Free viewpoint: avatar only there, grab a prop with the free hand / both hands, and stream resolution changes.

    C:\\isaacsim\\python.bat server\\main.py --headless
    C:\\isaacsim\\python.bat server\\tools\\hand_grab_check.py
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import ssl
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "_deps"))

import aiohttp  # noqa: E402
import numpy as np  # noqa: E402
from aiortc import RTCPeerConnection, RTCSessionDescription  # noqa: E402
from PIL import Image  # noqa: E402

URL = "https://127.0.0.1:8443"
LOG = os.path.join(os.path.dirname(__file__), "..", "logs", "teleop.log")


async def main() -> None:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ctx)) as http:
        dev = f"grab-{int(time.time())}"
        rid = (await (await http.post(f"{URL}/api/pair/request", json={"deviceName": dev})).json())["requestId"]
        await asyncio.sleep(0.3)
        code = re.findall(rf"PAIRING CODE for {dev} .*?: (\d{{6}})", open(LOG, encoding="utf-8").read())[-1]
        token = (await (await http.post(f"{URL}/api/pair/confirm", json={"requestId": rid, "code": code})).json())["token"]
        H = {"Authorization": f"Bearer {token}"}

        async def cmd(name, payload=None):
            r = await http.post(f"{URL}/api/command", json={"name": name, "payload": payload or {}}, headers=H)
            j = await r.json()
            return j.get("result", j)

        async def state():
            return await (await http.get(f"{URL}/api/state", headers=H)).json()

        async def snap_size():
            await asyncio.sleep(1.0)
            data = await (await http.get(f"{URL}/api/snapshot", headers=H)).read()
            return Image.open(io.BytesIO(data)).size

        pc = RTCPeerConnection()
        pc.addTransceiver("video", direction="recvonly")
        frames: list = []

        @pc.on("track")
        def on_track(track):
            async def pull():
                while True:
                    try:
                        f = await track.recv()
                    except Exception:  # noqa: BLE001
                        return
                    frames[:] = [(f.width, f.height)]

            asyncio.ensure_future(pull())

        teleop = pc.createDataChannel("teleop", ordered=False, maxRetransmits=0)
        pc.createDataChannel("events")
        await pc.setLocalDescription(await pc.createOffer())
        ans = await (await http.post(f"{URL}/api/webrtc/offer", json={"sdp": pc.localDescription.sdp, "type": "offer"}, headers=H)).json()
        await pc.setRemoteDescription(RTCSessionDescription(ans["sdp"], ans["type"]))
        for _ in range(100):
            if teleop.readyState == "open":
                break
            await asyncio.sleep(0.1)

        pose: dict = {}
        t0 = time.time()

        async def sender():
            while True:
                teleop.send(json.dumps({"t": "xr", "ts": (time.time() - t0) * 1000, **pose}))
                await asyncio.sleep(1 / 60)

        task = asyncio.ensure_future(sender())
        ident = [0, 0, 0, 1]

        def hand(p, grip=0.0):
            return {"p": list(map(float, p)), "q": ident, "trigger": 0, "grip": grip, "axes": [0, 0]}

        pose.update(head={"p": [0, 1.55, 0], "q": ident}, left=hand([-0.2, 1.0, -0.2]), right=hand([0.2, 1.0, -0.2]))
        await cmd("set_view", {"mode": "screen"})
        await asyncio.sleep(1.0)
        print("screen view: avatar shown =", (await state())["avatar"]["shown"])
        await cmd("set_view", {"mode": "immersive", "recenter": True})
        await asyncio.sleep(1.0)
        st = await state()
        print("free viewpoint: avatar shown =", st["avatar"]["shown"])
        ax, ay, yaw = st["immersive"]["anchor"]
        for _ in range(5):  # a previous session may have left the head turned: recenter on our straight pose
            if abs(((yaw - 180.0) + 180.0) % 360.0 - 180.0) < 1.0:
                break
            await cmd("set_view", {"mode": "immersive", "recenter": True})
            await asyncio.sleep(1.0)
            st = await state()
            ax, ay, yaw = st["immersive"]["anchor"]

        def xr(w):  # anchor yaw 180 deg: world = anchor + (z, x, y)
            return [w[1] - ay, w[2], w[0] - ax]

        props = {p["name"]: np.array(p["p"]) for p in st["props"]}
        name = "cube" if "cube" in props else next(iter(props))
        c = props[name]
        pose["head"] = {"p": [c[1] - ay, 1.45, c[0] + 0.4 - ax], "q": ident}
        # 1. free (left) hand: approach, grip, lift 25 cm, release
        pose["left"] = hand(xr(c + [0.0, 0.0, 0.10]))
        await asyncio.sleep(0.8)
        pose["left"] = hand(xr(c + [0.0, 0.0, 0.02]))
        await asyncio.sleep(0.8)
        pose["left"] = hand(xr(c + [0.0, 0.0, 0.02]), 1.0)
        await asyncio.sleep(0.5)
        for k in range(20):
            pose["left"] = hand(xr(c + [0.0, 0.0, 0.02 + 0.25 * (k + 1) / 20]), 1.0)
            await asyncio.sleep(0.05)
        await asyncio.sleep(0.5)
        st = await state()
        c2 = next(np.array(p["p"]) for p in st["props"] if p["name"] == name)
        print(f"left hand holds {st['handGrab']}; {name} z {c[2]:.3f} -> {c2[2]:.3f} (expected +0.25)")
        pose["left"] = hand(xr(c2 + [0.0, 0.0, 0.0]), 0.0)
        await asyncio.sleep(1.5)
        st = await state()
        c3 = next(np.array(p["p"]) for p in st["props"] if p["name"] == name)
        print(f"released: holds {st['handGrab']}; {name} z -> {c3[2]:.3f} (fell back to the table)")
        # 2. the control (right) hand does not grab in "robot" mode, it does in "grab" mode
        pose["left"] = hand([-0.2, 1.0, -0.2])
        pose["right"] = hand(xr(c3 + [0.0, 0.0, 0.02]))
        await asyncio.sleep(0.6)
        pose["right"] = hand(xr(c3 + [0.0, 0.0, 0.02]), 1.0)
        await asyncio.sleep(0.6)
        st = await state()
        print("robot mode, right grip: holds", st["handGrab"], "clutch", st["clutch"])
        pose["right"] = hand(xr(c3 + [0.0, 0.0, 0.02]), 0.0)
        await cmd("set_teleop_settings", {"handMode": "grab"})
        await asyncio.sleep(0.6)
        pose["right"] = hand(xr(c3 + [0.0, 0.0, 0.02]), 1.0)
        await asyncio.sleep(0.6)
        st = await state()
        print("grab mode, right grip: holds", st["handGrab"], "clutch", st["clutch"])
        pose["right"] = hand(xr(c3 + [0.0, 0.0, 0.02]), 0.0)
        await asyncio.sleep(0.5)
        await cmd("set_teleop_settings", {"handMode": "robot"})

        # 3. resolution
        print("resolution:", (await state())["resolution"])
        r = await cmd("set_resolution", {"immersive": 1536})
        await asyncio.sleep(1.5)
        print("immersive 1536 ->", r, "snapshot", await snap_size(), "webrtc frame", frames[-1] if frames else None)
        await cmd("set_view", {"mode": "screen"})
        await cmd("set_stereo", {"enabled": False})
        r = await cmd("set_resolution", {"screen": [1920, 1080]})
        await asyncio.sleep(1.5)
        print("screen 1920x1080 ->", r, "snapshot", await snap_size(), "webrtc frame", frames[-1] if frames else None)
        await cmd("set_stereo", {"enabled": True})
        r = await cmd("set_resolution", {"stereo": [1280, 720]})
        await asyncio.sleep(1.5)
        print("stereo 1280x720 ->", r, "snapshot", await snap_size(), "webrtc frame", frames[-1] if frames else None)
        print("bad value ->", await cmd("set_resolution", {"screen": [99999, 3]}))
        # restore the defaults
        print("restore ->", await cmd("set_resolution", {"screen": [1280, 720], "stereo": [960, 540], "immersive": 1024}))
        await cmd("set_camera", {"camera": "operator"})
        await asyncio.sleep(1.0)
        print("after restore snapshot", await snap_size())
        task.cancel()
        await http.post(f"{URL}/api/session/logout", headers=H)
        await pc.close()


asyncio.run(main())
