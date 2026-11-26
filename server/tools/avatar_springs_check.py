"""Compare the fast avatar paths (Warp spring bones, batched fingers) with the reference ones and time both.

    C:\\isaacsim\\python.bat server\\tools\\avatar_springs_check.py
"""

from __future__ import annotations

import glob
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402

from teleop import config as cfg  # noqa: E402
from teleop.avatar import Avatar, xr_pose_matrix  # noqa: E402

usd = glob.glob(os.path.join(cfg.AVATAR_CONVERTED_DIR, "*", "*.usdc"))[0]
meta = usd[: -len(".usdc")] + ".avatar.json"
ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
prim = stage.DefinePrim(cfg.AVATAR_PRIM, "Xform")
prim.GetReferences().AddReference(usd.replace("\\", "/"))
app.update()

ref = Avatar(stage, meta)
ref.spring_solver = None
ref.ik = None
fast = Avatar(stage, meta)
assert fast.spring_solver is not None and fast.ik is not None, "Warp solvers did not load"


def yaw_q(a):
    return [0.0, math.sin(a / 2), 0.0, math.cos(a / 2)]


def head(k):  # walk and turn so that hair / skirt swing, crouch now and then
    a = 0.8 * math.sin(k / 25)
    p = [0.3 * math.sin(k / 40), 1.6 - 0.25 * max(0.0, math.sin(k / 60)), -0.2 * math.cos(k / 35)]
    T = xr_pose_matrix(p, yaw_q(a))
    W = np.eye(4)  # XR (y up) -> world (z up)
    W[:3, :3] = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], dtype=float)
    return W @ T, W


times = {"numpy": [], "warp": []}
worst_ang, worst_tail = 0.0, 0.0
sp = ref.sp_joint
for k in range(300):
    Hw, W = head(k)
    hands = {}
    for side, x in (("left", -0.25), ("right", 0.25)):
        Th = xr_pose_matrix([x, 1.05 + 0.2 * math.sin(k / 15), -0.35], [0, 0, 0, 1])
        hands[side] = {"T": W @ Th, "trigger": 0.5 + 0.5 * math.sin(k / 9), "grip": 0.5 + 0.5 * math.cos(k / 11)}
    if 100 <= k < 140:  # left controller lost: the arm hangs
        hands.pop("left")
    if 200 <= k < 240:  # right controller far away: the arm stretches
        hands["right"]["T"] = W @ xr_pose_matrix([1.2, 1.4, -0.6], [0, 0, 0, 1])
    for name, av in (("numpy", ref), ("warp", fast)):
        cfg.AVATAR_FINGERS_BATCHED = name == "warp" and not os.environ.get("AV_SAMEFINGERS")
        t0 = time.perf_counter()
        av.update(Hw, hands, 1 / 15)
        times[name].append((time.perf_counter() - t0) * 1e3)
    if k < 2:
        continue
    allj = np.arange(ref.n)
    La, Lb = ref._local[:, :3, :3], fast._local[:, :3, :3]
    # largest element difference of the local rotation blocks (bones may carry scale, so no trace-angle here)
    diff = np.abs(La - Lb).max(axis=(1, 2))
    nonsp = np.setdiff1d(allj, fast.spring_solver.subtree_joints)
    worst_body = max(globals().get("worst_body", 0.0), float(diff[nonsp].max()))
    globals()["worst_body"] = worst_body
    if os.environ.get("AV_DEBUG") and k in (5, 100):
        d = diff[nonsp]
        top = np.argsort(d)[-4:]
        names = {j: b for b, j in fast.h.items()}
        print("CHECK debug", k, [(int(nonsp[t]), names.get(int(nonsp[t]), "?"), round(float(d[t]), 3)) for t in top])
    from teleop.avatar import _batch_quat_xyzw  # noqa: E402

    qa = _batch_quat_xyzw(ref._local[ref.anim_joints, :3, :3])
    qd = np.abs(np.einsum("ki,ki->k", qa, fast.spring_solver.rot_np.astype(np.float64)))
    globals()["worst_q"] = max(globals().get("worst_q", 0.0), float(np.degrees(2 * np.arccos(np.clip(qd, -1, 1))).max()))
    Ra, Rb = ref._local[sp, :3, :3], fast._local[sp, :3, :3]
    cos = np.clip((np.einsum("kij,kij->k", Ra, Rb) - 1) / 2, -1, 1)
    worst_ang = max(worst_ang, float(np.degrees(np.arccos(cos)).max()))
    ba, bb = ref.body_poses(), fast.body_poses()
    globals()["worst_bp"] = max(globals().get("worst_bp", 0.0), float(np.abs(ba[0] - bb[0]).max()))
    if k == 50: print("CHECK bodies ref", np.round(ba[0][:3], 3).tolist(), "fast", np.round(bb[0][:3], 3).tolist())
    tc, _ = fast.spring_solver.tails()
    worst_tail = max(worst_tail, float(np.abs(ref._tails[:, 0] - tc).max()))

if os.environ.get("AV_PROFILE"):
    import cProfile
    import pstats

    pr = cProfile.Profile()
    pr.enable()
    for k in range(100):
        Hw, W = head(k)
        hands = {sd: {"T": W @ xr_pose_matrix([x, 1.05, -0.35], [0, 0, 0, 1]), "trigger": 0.3, "grip": 0.5}
                 for sd, x in (("left", -0.25), ("right", 0.25))}
        fast.update(Hw, hands, 1 / 15)
    pr.disable()
    pstats.Stats(pr).sort_stats("tottime").print_stats(18)

for name, t in times.items():
    a = np.array(t[10:])
    print(f"CHECK {name}: avatar.update avg {a.mean():.2f} ms  p95 {np.percentile(a, 95):.2f} ms", flush=True)
from teleop.perf import PERF  # noqa: E402

for k_, v_ in sorted(PERF.summary()["sections"].items()):
    print(f"CHECK   {k_:12s} avg {v_['avg']:.2f}  p95 {v_['p95']:.2f} ms  (both avatars mixed)", flush=True)
print(f"CHECK max body position difference {globals().get('worst_bp', 0):.2e} m")
print(f"CHECK max non-spring joint local-matrix difference {globals().get('worst_body', 0):.2e}")
print(f"CHECK max written joint quaternion difference {globals().get('worst_q', 0):.4f} deg")
print(f"CHECK max spring rotation difference {worst_ang:.4f} deg, max tail difference {worst_tail * 1000:.3f} mm", flush=True)
app.close()
