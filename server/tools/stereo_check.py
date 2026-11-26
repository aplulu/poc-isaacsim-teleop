"""Switch the stream to stereo over the API and check frame size, left/right disparity and frame rate.

    C:\\isaacsim\\python.bat server\\tools\\stereo_check.py
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
OUT = os.path.join(os.path.dirname(__file__), "_out")


def disparity(left: np.ndarray, right: np.ndarray, max_shift: int = 80) -> int:
    """Horizontal shift (px) that best aligns the right image to the left one (centre band)."""
    h = left.shape[0]
    a = left[h // 3: 2 * h // 3].mean(axis=2)
    b = right[h // 3: 2 * h // 3].mean(axis=2)
    best, best_err = 0, 1e18
    for s in range(0, max_shift):
        err = np.mean(np.abs(a[:, max_shift:-max_shift] - b[:, max_shift - s: b.shape[1] - max_shift - s]))
        if err < best_err:
            best, best_err = s, err
    return best


async def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ctx)) as http:
        dev = f"stereo-{int(time.time())}"
        rid = (await (await http.post(f"{URL}/api/pair/request", json={"deviceName": dev})).json())["requestId"]
        await asyncio.sleep(0.3)
        code = re.findall(rf"PAIRING CODE for {dev} .*?: (\d{{6}})", open(LOG, encoding="utf-8").read())[-1]
        token = (await (await http.post(f"{URL}/api/pair/confirm", json={"requestId": rid, "code": code})).json())["token"]
        H = {"Authorization": f"Bearer {token}"}

        async def cmd(name, payload=None):
            r = await http.post(f"{URL}/api/command", json={"name": name, "payload": payload or {}}, headers=H)
            return (await r.json()).get("result")

        pc = RTCPeerConnection()
        pc.addTransceiver("video", direction="recvonly")
        pc.createDataChannel("events")
        frames: list[tuple[float, np.ndarray]] = []

        @pc.on("track")
        def on_track(track):
            async def pull():
                while True:
                    try:
                        f = await track.recv()
                    except Exception:  # noqa: BLE001
                        return
                    frames.append((time.time(), f.to_ndarray(format="rgb24")))
                    del frames[:-90]

            asyncio.ensure_future(pull())

        await pc.setLocalDescription(await pc.createOffer())
        ans = await (await http.post(f"{URL}/api/webrtc/offer", json={"sdp": pc.localDescription.sdp, "type": "offer"}, headers=H)).json()
        await pc.setRemoteDescription(RTCSessionDescription(ans["sdp"], ans["type"]))

        async def measure(label: str, seconds: float = 4.0):
            frames.clear()
            await asyncio.sleep(seconds)
            st = await (await http.get(f"{URL}/api/state", headers=H)).json()
            n = len(frames)
            fps = (n - 1) / (frames[-1][0] - frames[0][0]) if n > 1 else 0
            img = frames[-1][1]
            print(f"{label}: frame {img.shape[1]}x{img.shape[0]}, received {fps:.1f} fps, sim {st['fps']} fps, stereo={st['stereo']} ipd={st['ipd']}")
            return img

        await asyncio.sleep(2)
        await measure("mono")
        print("set_stereo ->", await cmd("set_stereo", {"enabled": True, "ipd": 0.064}))
        await asyncio.sleep(1.5)
        img = await measure("stereo")
        w = img.shape[1] // 2
        d64 = disparity(img[:, :w], img[:, w:])
        from PIL import Image

        Image.fromarray(img).save(os.path.join(OUT, "stereo_frame.jpg"), quality=90)
        print(f"  disparity at screen centre band (ipd 64 mm): {d64} px")
        await cmd("set_stereo", {"ipd": 0.080})
        await asyncio.sleep(1.0)
        img = await measure("stereo ipd 80", 2.0)
        print(f"  disparity (ipd 80 mm): {disparity(img[:, :w], img[:, w:])} px (should grow ~x1.25)")
        await cmd("set_camera", {"camera": "wrist"})
        img = await measure("stereo wrist", 2.0)
        Image.fromarray(img).save(os.path.join(OUT, "stereo_wrist.jpg"), quality=90)
        await cmd("set_camera", {"camera": "operator"})
        print("set_stereo off ->", await cmd("set_stereo", {"enabled": False, "ipd": 0.064}))
        await asyncio.sleep(1.5)
        await measure("mono again", 2.0)
        await http.post(f"{URL}/api/session/logout", headers=H)
        await pc.close()


asyncio.run(main())
