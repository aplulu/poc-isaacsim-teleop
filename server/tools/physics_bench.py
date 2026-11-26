"""Physics cost of the teleop scene: time one 60 Hz PhysX step under several scene variants.

    C:\\isaacsim\\python.bat server\\tools\\physics_bench.py [--steps 240]

Each variant edits the opened scene in memory (the file is not saved), restarts the simulation and times
simulate() + fetch_results() per step.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=240)
ap.add_argument("--only", default="", help="comma separated variant names")
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.physx  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Sdf, Usd, UsdPhysics  # noqa: E402

from teleop import config as cfg  # noqa: E402

DT = 1.0 / 60.0


def open_scene() -> Usd.Stage:
    ctx = omni.usd.get_context()
    ctx.open_stage(cfg.SCENE_PATH)
    for _ in range(50):
        app.update()
        if not ctx.get_stage_loading_status()[2]:
            break
    return ctx.get_stage()


def props(stage):
    root = stage.GetPrimAtPath(cfg.PROPS_ROOT)
    return [p for p in root.GetChildren()] if root else []


def bodies(prim):
    return [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)]


def is_parked(prim) -> bool:
    for b in bodies(prim):
        a = b.GetAttribute("physxRigidBody:disableGravity")
        if a and a.Get():
            return True
    return False


def v_baseline(stage):
    pass


def v_no_parked(stage):
    for p in props(stage):
        if is_parked(p):
            p.SetActive(False)


def v_no_props(stage):
    for p in props(stage):
        p.SetActive(False)


def v_no_ccd(stage):
    stage.GetPrimAtPath("/World/PhysicsScene").GetAttribute("physxScene:enableCCD").Set(False)
    for p in props(stage):
        for b in bodies(p):
            a = b.GetAttribute("physxRigidBody:enableCCD")
            if a:
                a.Set(False)


def v_prop_iters4(stage):
    for p in props(stage):
        for b in bodies(p):
            b.GetAttribute("physxRigidBody:solverPositionIterationCount").Set(4)


def v_artic_iters4(stage):
    for p in Usd.PrimRange(stage.GetPrimAtPath(cfg.ROBOT_PRIM)):
        a = p.GetAttribute("physxArticulation:solverPositionIterationCount")
        if a and a.HasAuthoredValue():
            a.Set(4)


def v_gpu(stage):
    sp = stage.GetPrimAtPath("/World/PhysicsScene")
    sp.CreateAttribute("physxScene:enableGPUDynamics", Sdf.ValueTypeNames.Bool).Set(True)
    sp.CreateAttribute("physxScene:broadphaseType", Sdf.ValueTypeNames.Token).Set("GPU")


def v_tgs_off_stab(stage):
    sp = stage.GetPrimAtPath("/World/PhysicsScene")
    sp.CreateAttribute("physxScene:enableStabilization", Sdf.ValueTypeNames.Bool).Set(False)
    sp.CreateAttribute("physxScene:solverType", Sdf.ValueTypeNames.Token).Set("TGS")


def v_no_robot(stage):
    stage.GetPrimAtPath(cfg.ROBOT_PRIM).SetActive(False)


VARIANTS = {
    "baseline": v_baseline,
    "no_parked_props": v_no_parked,
    "no_props": v_no_props,
    "no_ccd": v_no_ccd,
    "prop_iters_4": v_prop_iters4,
    "artic_iters_4": v_artic_iters4,
    "no_robot": v_no_robot,
    "gpu_dynamics": v_gpu,
    "tgs_no_stabilization": v_tgs_off_stab,
}


def run(name, fn) -> None:
    stage = open_scene()
    n_props = len(props(stage))
    n_parked = sum(is_parked(p) for p in props(stage))
    n_bodies = sum(len(bodies(p)) for p in props(stage))
    fn(stage)
    tl = omni.timeline.get_timeline_interface()
    tl.play()
    for _ in range(20):
        app.update()
    tl.pause()
    for _ in range(3):
        app.update()
    sim = omni.physx.get_physx_simulation_interface()
    times = []
    t = 0.0
    for _ in range(args.steps):
        t0 = time.perf_counter()
        sim.simulate(DT, t)
        sim.fetch_results()
        times.append((time.perf_counter() - t0) * 1e3)
        t += DT
    a = np.array(times[20:])
    print(f"BENCH {name:22s} step avg {a.mean():6.2f} ms  p50 {np.median(a):6.2f}  p95 {np.percentile(a, 95):6.2f}"
          f"   (props {n_props}, parked {n_parked}, prop bodies {n_bodies})", flush=True)
    tl.stop()
    app.update()


only = [s for s in args.only.split(",") if s]
for name, fn in VARIANTS.items():
    if only and name not in only:
        continue
    try:
        run(name, fn)
    except Exception as exc:  # noqa: BLE001
        print(f"BENCH {name}: failed: {exc}", flush=True)

app.close()
