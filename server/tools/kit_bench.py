"""Kit update cost with physics playing vs paused (fixed 1/60 s per update), with or without omni.physx.fabric.

    C:\\isaacsim\\python.bat server\\tools\\kit_bench.py [--fabric] [--frames 200]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--fabric", action="store_true")
ap.add_argument("--frames", type=int, default=200)
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402

from teleop import config as cfg  # noqa: E402

if args.fabric:
    ok = omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.physx.fabric", True)
    print("BENCH fabric enabled:", ok, flush=True)

ctx = omni.usd.get_context()
ctx.open_stage(cfg.SCENE_PATH)
for _ in range(100):
    app.update()
    if not ctx.get_stage_loading_status()[2]:
        break
for _ in range(30):
    app.update()


def measure(n):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        app.update()
        ts.append((time.perf_counter() - t0) * 1e3)
    a = np.array(ts[10:])
    return a.mean(), np.percentile(a, 95)


tl = omni.timeline.get_timeline_interface()
print("BENCH paused  update avg %.1f p95 %.1f ms" % measure(args.frames), flush=True)
tl.play()
print("BENCH playing update avg %.1f p95 %.1f ms" % measure(args.frames), flush=True)
from pxr import UsdGeom  # noqa: E402

stage = ctx.get_stage()
cube = next((p for p in stage.GetPrimAtPath(cfg.PROPS_ROOT).GetChildren() if p.GetName() == "cube"), None)
if cube:
    m = UsdGeom.Xformable(cube).ComputeLocalToWorldTransform(0)
    print("BENCH cube USD z after play:", round(m.ExtractTranslation()[2], 4), flush=True)
tl.stop()
app.update()
app.close()
