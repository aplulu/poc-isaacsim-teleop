"""Check that controller motion maps to the expected gripper motion for each camera view.

    C:\\isaacsim\\python.bat server\\tools\\mapping_check.py

For every camera it clutches, moves the (simulated) right controller 10 cm to the right / up / forward,
and reports the TCP displacement expressed in that camera's image axes (right, up, into-the-image).
A correct mapping shows a large positive value on the matching axis.
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


def quat_to_mat(q):  # wxyz
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


async def main() -> None:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ctx)) as http:
        dev = f"map-{int(time.time())}"
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
        dc = pc.createDataChannel("teleop", ordered=False, maxRetransmits=0)
        pc.createDataChannel("events")
        await pc.setLocalDescription(await pc.createOffer())
        ans = await (await http.post(f"{URL}/api/webrtc/offer", json={"sdp": pc.localDescription.sdp, "type": "offer"}, headers=H)).json()
        await pc.setRemoteDescription(RTCSessionDescription(ans["sdp"], ans["type"]))
        for _ in range(100):
            if dc.readyState == "open":
                break
            await asyncio.sleep(0.1)

        t0 = time.time()

        async def stream(p, q, grip, seconds):
            n = int(seconds * 60)
            for _ in range(n):
                dc.send(json.dumps({"t": "xr", "ts": (time.time() - t0) * 1000, "right": {"p": list(p), "q": list(q), "grip": grip, "trigger": 0}}))
                await asyncio.sleep(1 / 60)

        await cmd("set_teleop_settings", {"rotationMode": "yaw", "smoothing": 0.5})
        p0 = np.array([0.2, 1.2, -0.4])
        ident = [0, 0, 0, 1]
        moves = {"right": np.array([0.1, 0, 0]), "up": np.array([0, 0.1, 0]), "forward": np.array([0, 0, -0.1])}
        ok = True
        for cam in ("operator", "front", "top", "dataset", "wrist"):
            await cmd("reset")
            await asyncio.sleep(1.0)
            await cmd("set_camera", {"camera": cam})
            for name, d in moves.items():
                s0 = await state()
                # wrist view: measure in the camera frame captured at clutch time
                await stream(p0, ident, 1.0, 0.4)
                s_mid = await state()
                await stream(p0 + d, ident, 1.0, 1.2)
                await stream(p0 + d, ident, 0.0, 0.2)
                s1 = await state()
                dp = np.array(s1["tcp"]["p"]) - np.array(s0["tcp"]["p"])
                cam_axes = s_mid.get("cameraAxes")
                if cam_axes:
                    R = np.array(cam_axes)  # columns: right, up, back
                    img = np.array([R[:, 0] @ dp, R[:, 1] @ dp, -R[:, 2] @ dp])
                    axis = {"right": 0, "up": 1, "forward": 2}[name]
                    sign = 1.0
                    if cam == "top" and name != "right":
                        # straight-down view: hand up raises the gripper (towards the camera = out of the
                        # image), hand forward moves it towards the top of the image
                        axis, sign = {"up": (2, -1.0), "forward": (1, 1.0)}[name]
                    good = sign * img[axis] > 0.05 if cam in ("wrist", "top") else img[axis] > 0.03
                    ok &= bool(good)
                    print(f"{cam:8s} hand {name:7s} -> image (right, up, in) = {np.round(img, 3)}  {'OK' if good else 'NG'}")
                # undo
                await stream(p0 + d, ident, 1.0, 0.3)
                await stream(p0, ident, 1.0, 1.0)
                await stream(p0, ident, 0.0, 0.2)
        # rotation: roll the controller 40 deg about its forward axis in yaw mode -> gripper must stay vertical
        await cmd("reset")
        await cmd("set_camera", {"camera": "operator"})
        await asyncio.sleep(1.0)
        a = np.radians(40) / 2
        roll = [0, 0, np.sin(a), np.cos(a)]  # xyzw about XR z (towards the user)
        await stream(p0, ident, 1.0, 0.4)
        await stream(p0, roll, 1.0, 1.2)
        s = await state()
        approach = quat_to_mat(s["tcp"]["q"])[:, 0]  # flange approach axis (+X)
        tilt = np.degrees(np.arccos(np.clip(-approach[2], -1, 1)))
        print(f"yaw mode, controller rolled 40 deg -> gripper tilt from vertical {tilt:.1f} deg {'OK' if tilt < 3 else 'NG'}")
        ok &= tilt < 3
        await stream(p0, roll, 0.0, 0.2)
        print("ALL OK" if ok else "SOME CHECKS FAILED")
        await http.post(f"{URL}/api/session/logout", headers=H)
        await pc.close()


asyncio.run(main())
