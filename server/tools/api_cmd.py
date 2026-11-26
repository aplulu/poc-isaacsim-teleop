"""Tiny CLI: pair with a --headless server (code scraped from the log) and run commands.

    C:\\isaacsim\\python.bat server\\tools\\cmd.py randomize '{"targets":["objects"]}' reset '{}'
"""

from __future__ import annotations

import json
import os
import re
import ssl
import sys
import time
import urllib.request

URL = os.environ.get("TELEOP_URL", "https://127.0.0.1:8443")
LOG = os.path.join(os.path.dirname(__file__), "..", "logs", "teleop.log")
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def call(method, path, body=None, token=None):
    req = urllib.request.Request(URL + path, method=method, data=json.dumps(body).encode() if body is not None else None)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, context=ctx) as r:
        return json.loads(r.read())


TOKEN_FILE = os.path.join(os.path.dirname(__file__), "_out", ".api_token")
token = None
if os.path.exists(TOKEN_FILE):  # reuse the session (pairing is rate limited)
    token = open(TOKEN_FILE, encoding="utf-8").read().strip()
    try:
        call("GET", "/api/session", token=token)
    except Exception:  # noqa: BLE001
        token = None
if token is None:
    dev = f"cli-{int(time.time() * 1000)}"
    rid = call("POST", "/api/pair/request", {"deviceName": dev})["requestId"]
    time.sleep(0.3)
    code = re.findall(rf"PAIRING CODE for {dev} .*?: (\d{{6}})", open(LOG, encoding="utf-8").read())[-1]
    token = call("POST", "/api/pair/confirm", {"requestId": rid, "code": code})["token"]
    os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
    open(TOKEN_FILE, "w", encoding="utf-8").write(token)
args = sys.argv[1:]
for name, payload in zip(args[::2], args[1::2]):
    if name == "sleep":
        time.sleep(float(payload))
        continue
    if name == "snap":  # snap <file.jpg>
        req = urllib.request.Request(URL + "/api/snapshot", headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, context=ctx) as r, open(payload, "wb") as f:
            f.write(r.read())
        print("snap ->", payload)
        continue
    print(name, "->", json.dumps(call("POST", "/api/command", {"name": name, "payload": json.loads(payload)}, token))[:300])
st = call("GET", "/api/state", token=token)
print("tcp:", st.get("tcp"), "gripper:", st.get("gripper"), "targetError:", st.get("targetError"))
print("sim:", st.get("simTime"), "fps:", st.get("fps"))
print("dofs:", st.get("dofs"))
print("props:", {p["name"]: p["p"] for p in st.get("props", [])})
rec = st.get("recording") or {}
print("rec:", {k: rec.get(k) for k in ("state", "task", "episode", "frames", "message")}, "dataset:", {k: (rec.get("dataset") or {}).get(k) for k in ("episodes", "frames", "pending", "lastError")})
