"""Drive the operator avatar with synthetic headset / controller poses and save snapshots.

    C:\\isaacsim\\python.bat server\\tools\\avatar_check.py

Images go to server/tools/_out/avatar_*.jpg.
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
from aiortc import RTCPeerConnection, RTCSessionDescription  # noqa: E402

URL = "https://127.0.0.1:8443"
LOG = os.path.join(os.path.dirname(__file__), "..", "logs", "teleop.log")
OUT = os.path.join(os.path.dirname(__file__), "_out")


def quat(axis, deg: float) -> list[float]:
    a = math.radians(deg) / 2
    n = math.sqrt(sum(v * v for v in axis))
    return [axis[0] / n * math.sin(a), axis[1] / n * math.sin(a), axis[2] / n * math.sin(a), math.cos(a)]


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz]


async def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ctx)) as http:
        dev = f"avatar-{int(time.time())}"
        rid = (await (await http.post(f"{URL}/api/pair/request", json={"deviceName": dev})).json())["requestId"]
        await asyncio.sleep(0.3)
        code = re.findall(rf"PAIRING CODE for {dev} .*?: (\d{{6}})", open(LOG, encoding="utf-8").read())[-1]
        token = (await (await http.post(f"{URL}/api/pair/confirm", json={"requestId": rid, "code": code})).json())["token"]
        H = {"Authorization": f"Bearer {token}"}

        async def cmd(name, payload=None):
            r = await http.post(f"{URL}/api/command", json={"name": name, "payload": payload or {}}, headers=H)
            return (await r.json()).get("result")

        async def snap(name):
            r = await http.get(f"{URL}/api/snapshot", headers=H)
            data = await r.read()
            with open(os.path.join(OUT, f"avatar_{name}.jpg"), "wb") as f:
                f.write(data)
            print("saved", name, len(data), flush=True)

        pc = RTCPeerConnection()
        pc.addTransceiver("video", direction="recvonly")
        teleop = pc.createDataChannel("teleop", ordered=False, maxRetransmits=0)
        pc.createDataChannel("events")
        await pc.setLocalDescription(await pc.createOffer())
        ans = await (await http.post(f"{URL}/api/webrtc/offer", json={"sdp": pc.localDescription.sdp, "type": "offer"}, headers=H)).json()
        await pc.setRemoteDescription(RTCSessionDescription(ans["sdp"], ans["type"]))
        for _ in range(100):
            if teleop.readyState == "open":
                break
            await asyncio.sleep(0.1)

        pose = {}
        t0 = time.time()

        async def sender():
            while True:
                teleop.send(json.dumps({"t": "xr", "ts": (time.time() - t0) * 1000, **pose}))
                await asyncio.sleep(1 / 60)

        task = asyncio.ensure_future(sender())
        ident = [0, 0, 0, 1]
        down30 = quat([1, 0, 0], -30)

        def hand(p, q=ident, trig=0.0, grip=0.0):
            return {"p": p, "q": q, "trigger": trig, "grip": grip, "axes": [0, 0]}

        # 1. neutral: hands in front at waist height
        pose.update(head={"p": [0, 1.55, 0], "q": ident},
                    left=hand([-0.2, 1.05, -0.3], down30), right=hand([0.2, 1.05, -0.3], down30))
        print("avatar", (await (await http.get(f"{URL}/api/state", headers=H)).json()).get("avatar"))
        for cam in ("front", "top", "operator"):
            await cmd("set_camera", {"camera": cam})
            await asyncio.sleep(1.5)
            await snap(f"neutral_{cam}")
        # 2. reach: right hand forward and up, fist closed; left hand raised; head turned left
        pose.update(head={"p": [0, 1.5, -0.05], "q": qmul(quat([0, 1, 0], 25), quat([1, 0, 0], -20))},
                    left=hand([-0.35, 1.6, -0.25], quat([0, 0, 1], 60), grip=1.0),
                    right=hand([0.15, 1.3, -0.55], down30, trig=1.0, grip=1.0))
        await cmd("set_camera", {"camera": "front"})
        await asyncio.sleep(1.5)
        await snap("reach_front")
        await cmd("set_camera", {"camera": "operator"})
        await asyncio.sleep(1.0)
        await snap("reach_operator")
        # 3. crouch
        pose.update(head={"p": [0, 1.0, -0.1], "q": quat([1, 0, 0], -30)},
                    left=hand([-0.25, 0.7, -0.35], down30), right=hand([0.25, 0.7, -0.35], down30))
        await cmd("set_camera", {"camera": "front"})
        await asyncio.sleep(1.5)
        await snap("crouch_front")
        # 4. free viewpoint, stand straight, hands in front
        pose.update(head={"p": [0, 1.55, 0], "q": ident},
                    left=hand([-0.2, 1.1, -0.35], down30), right=hand([0.2, 1.1, -0.35], down30))
        print("set_view", await cmd("set_view", {"mode": "immersive", "recenter": True}))
        await asyncio.sleep(1.0)
        await cmd("set_view", {"mode": "immersive", "recenter": True})
        await asyncio.sleep(1.5)
        await snap("immersive_first_person")
        # look at the mirror (+Y in the sim = to the user's right when facing -X)
        pose.update(head={"p": [0, 1.55, 0], "q": quat([0, 1, 0], -70)})
        await asyncio.sleep(1.5)
        await snap("immersive_mirror")
        await cmd("set_view", {"mode": "screen"})
        await cmd("set_camera", {"camera": "operator"})
        print("avatar", (await (await http.get(f"{URL}/api/state", headers=H)).json()).get("avatar"))
        task.cancel()
        await http.post(f"{URL}/api/session/logout", headers=H)
        await pc.close()


asyncio.run(main())
