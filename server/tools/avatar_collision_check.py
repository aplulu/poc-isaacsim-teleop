"""Check the avatar's colliders: sweep a hand through a prop (free viewpoint) and block the robot with the control arm.

    C:\\isaacsim\\python.bat server\\main.py --headless
    C:\\isaacsim\\python.bat server\\tools\\avatar_collision_check.py
"""

from __future__ import annotations

import asyncio
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

URL = "https://127.0.0.1:8443"
LOG = os.path.join(os.path.dirname(__file__), "..", "logs", "teleop.log")


async def main() -> None:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ctx)) as http:
        dev = f"avcol-{int(time.time())}"
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
        rest_l = {"p": [-0.2, 1.0, -0.2], "q": ident, "trigger": 0, "grip": 0, "axes": [0, 0]}
        pose.update(head={"p": [0, 1.55, 0], "q": ident}, left=rest_l, right=dict(rest_l, p=[0.2, 1.0, -0.2]))
        await asyncio.sleep(0.5)
        print("screen view (auto -> off):", {k: (await state())["avatar"][k] for k in ("collision", "collisionActive", "bodies")})
        await cmd("set_view", {"mode": "immersive", "recenter": True})
        await asyncio.sleep(1.0)
        st = await state()
        print("free viewpoint (auto -> on):", {k: st["avatar"][k] for k in ("collision", "collisionActive", "bodies")})
        ax, ay, yaw = st["immersive"]["anchor"]  # yaw 180: world = anchor + (z_xr, x_xr, y_xr)
        for _ in range(5):  # a previous session may have left the head turned: recenter on our straight pose
            if abs(((yaw - 180.0) + 180.0) % 360.0 - 180.0) < 1.0:
                break
            await cmd("set_view", {"mode": "immersive", "recenter": True})
            await asyncio.sleep(1.0)
            ax, ay, yaw = (await state())["immersive"]["anchor"]
        print("anchor yaw", round(yaw, 1))

        def xr(w):
            return [w[1] - ay, w[2], w[0] - ax]

        props = {p["name"]: np.array(p["p"]) for p in st.get("props", [])}
        name = "cube" if "cube" in props else next(iter(props))
        c = props[name]
        # walk up to the table (head 0.45 m in front of the prop, on the far side from the robot)
        head_w = c + np.array([0.45, 0.0, 0.0])
        pose["head"] = {"p": [c[1] - ay, 1.45, head_w[0] - ax], "q": ident}
        await asyncio.sleep(1.5)
        before = (await state())
        c = np.array(next(p["p"] for p in before["props"] if p["name"] == name))
        # sweep the right hand through the prop along -Y (world), at its height
        for k in range(60):
            y = c[1] + 0.25 - 0.5 * k / 59
            w = np.array([c[0] + 0.06, y, c[2] + 0.02])  # grip a bit in front: the wrist/hand body sits behind
            pose["right"] = {"p": xr(w), "q": ident, "trigger": 0, "grip": 0, "axes": [0, 0]}
            await asyncio.sleep(1 / 40)
        await asyncio.sleep(1.0)
        after = await state()
        c2 = np.array(next(p["p"] for p in after["props"] if p["name"] == name))
        print(f"{name}: {np.round(c, 3)} -> {np.round(c2, 3)}  moved {np.linalg.norm(c2 - c):.3f} m")

        # collisions off: the same sweep must not move it
        await cmd("set_avatar", {"collision": "off"})
        await asyncio.sleep(0.5)
        c = c2
        for k in range(60):
            y = c[1] + 0.25 - 0.5 * k / 59
            pose["right"] = {"p": xr([c[0] + 0.06, y, c[2] + 0.02]), "q": ident, "trigger": 0, "grip": 0, "axes": [0, 0]}
            await asyncio.sleep(1 / 40)
        await asyncio.sleep(1.0)
        c3 = np.array(next(p["p"] for p in (await state())["props"] if p["name"] == name))
        print(f"collision off: moved {np.linalg.norm(c3 - c):.3f} m (expected ~0)")
        await cmd("set_avatar", {"collision": "auto"})

        # robot: hold the left forearm in the gripper's path and jog the TCP into it
        st = await state()
        tcp = np.array(st["tcp"]["p"])
        block = tcp + np.array([0.0, 0.0, -0.12])
        # the right (control) hand: in the default hand mode the left hand can grab, so it has no colliders
        pose["left"] = rest_l
        pose["right"] = {"p": xr(block - np.array([0.05, 0.0, 0.0])), "q": ident, "trigger": 0, "grip": 0, "axes": [0, 0]}
        pose["head"] = {"p": [tcp[1] - ay, 1.5, tcp[0] + 0.45 - ax], "q": ident}
        await asyncio.sleep(1.5)
        r = await cmd("move_tcp", {"delta": [0.0, 0.0, -0.25]})
        await asyncio.sleep(2.5)
        tcp2 = np.array((await state())["tcp"]["p"])
        print(f"robot TCP z {tcp[2]:.3f} -> {tcp2[2]:.3f} (target {tcp[2] - 0.25:.3f}; blocked if well above) move_tcp={r}")
        await cmd("go_home")
        await cmd("set_view", {"mode": "screen"})
        task.cancel()
        await http.post(f"{URL}/api/session/logout", headers=H)
        await pc.close()


asyncio.run(main())
