"""Record scripted pick-and-place episodes through the public API (smoke test for the dataset pipeline).

    C:\\isaacsim\\python.bat server\\tools\\scripted_episode.py --episodes 2 --object tangerine

Pairs with a --headless server (code scraped from server/logs/teleop.log), starts an episode for
"put the <object> in the bowl", drives the TCP with interpolated move_tcp commands, and waits for the
automatic success detection to save the episode.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import time
import urllib.request

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--url", default=os.environ.get("TELEOP_URL", "https://127.0.0.1:8443"))
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--object", default="tangerine")
parser.add_argument("--randomize", action="store_true", help="domain randomization at episode start")
args = parser.parse_args()

LOG = os.path.join(os.path.dirname(__file__), "..", "logs", "teleop.log")
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
token = None


def call(method, path, body=None):
    req = urllib.request.Request(args.url + path, method=method, data=json.dumps(body).encode() if body is not None else None)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, context=ctx) as r:
        return json.loads(r.read())


def cmd(name, payload=None):
    return call("POST", "/api/command", {"name": name, "payload": payload or {}})["result"]


def state():
    return call("GET", "/api/state")


def move(target, speed=0.25, hz=20):
    """Interpolate the TCP target from its current position to ``target`` at ``speed`` m/s."""
    start = np.array(state()["tcp"]["p"])
    target = np.array(target, dtype=float)
    n = max(1, int(np.linalg.norm(target - start) / speed * hz))
    for i in range(1, n + 1):
        cmd("move_tcp", {"p": (start + (target - start) * i / n).tolist()})
        time.sleep(1 / hz)
    time.sleep(0.6)


dev = f"script-{int(time.time() * 1000)}"
rid = call("POST", "/api/pair/request", {"deviceName": dev})["requestId"]
time.sleep(0.3)
code = re.findall(rf"PAIRING CODE for {dev} .*?: (\d{{6}})", open(LOG, encoding="utf-8").read())[-1]
token = call("POST", "/api/pair/confirm", {"requestId": rid, "code": code})["token"]
cmd("rec_settings", {"waitForMotion": False, "randomize": args.randomize, "autoSuccess": True, "maxDuration": 60})

for ep in range(args.episodes):
    cmd("rec_start", {"task": {"id": "put_in_tray", "object": args.object}})
    for _ in range(100):
        if state()["recording"]["state"] == "recording":
            break
        time.sleep(0.1)
    st = state()
    props = {p["name"]: np.array(p["p"]) for p in st["props"]}
    obj, bowl = props[args.object], props["tray"]
    print(f"[ep {ep}] task:", st["recording"]["task"]["text"], "object at", obj.round(3), "bowl at", bowl.round(3))
    # props are reported by their bounding-box center; for flat parts keep the fingertips just above the table
    grasp_z = max(obj[2] + 0.005, st["tableTop"] + 0.03)
    move([obj[0], obj[1], grasp_z + 0.15])
    move([obj[0], obj[1], grasp_z], speed=0.12)
    cmd("set_gripper", {"closed": True})
    time.sleep(1.2)
    move([obj[0], obj[1], grasp_z + 0.2], speed=0.15)
    st = state()
    print(f"[ep {ep}] after lift: gripper", st["gripper"], "object", [p["p"] for p in st["props"] if p["name"] == args.object])
    move([bowl[0], bowl[1], bowl[2] + 0.22], speed=0.25)
    move([bowl[0], bowl[1], bowl[2] + 0.12], speed=0.12)
    st = state()
    print(f"[ep {ep}] before release: gripper", st["gripper"], "object", [p["p"] for p in st["props"] if p["name"] == args.object])
    cmd("set_gripper", {"closed": False})
    time.sleep(1.5)
    st = state()
    print(f"[ep {ep}] after release: object", [p["p"] for p in st["props"] if p["name"] == args.object], "bowl", [p["p"] for p in st["props"] if p["name"] == "bowl"])
    for _ in range(60):
        rec = state()["recording"]
        if rec["state"] == "idle":
            break
        time.sleep(0.2)
    print(f"[ep {ep}]", rec["message"].encode("ascii", "backslashreplace").decode(), rec["state"])
    if rec["state"] != "idle":
        print(f"[ep {ep}] no automatic success - discarding")
        cmd("rec_discard")
    move([0.5, 0.0, 1.1])

time.sleep(2.0)
print("dataset:", json.dumps(state()["recording"]["dataset"], ensure_ascii=False)[:600])
call("POST", "/api/session/logout")
