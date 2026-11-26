"""Per-frame breakdown of a Kit CPU profiler trace written by ``main.py --kit-trace``.

    C:\\isaacsim\\python.bat server\\main.py --headless --kit-trace server\\logs\\kit_trace.json
    C:\\isaacsim\\python.bat server\\tools\\perf_check.py --kit-trace immersive
    C:\\isaacsim\\python.bat server\\tools\\kit_trace_report.py server\\logs\\kit_trace.json [--depth 4] [--min 0.3]

Main thread: the zones inside each main-loop phase (teleop.pre / teleop.kit / teleop.post), as a tree of average
milliseconds per frame (inclusive; "(self)" is the time not covered by a listed child). Other threads: average busy
time per frame while the main loop runs, with their largest zones.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("trace")
ap.add_argument("--depth", type=int, default=4)
ap.add_argument("--min", type=float, default=0.3, help="hide zones below this many ms per frame")
ap.add_argument("--threads", type=int, default=12)
ap.add_argument("--frames", type=int, default=60, help="the last N main-loop frames (the kit_trace capture)")
args = ap.parse_args()

d = json.load(open(args.trace, encoding="utf-8"))
ev = d["traceEvents"] if isinstance(d, dict) else d
tname = {e["tid"]: e["args"]["name"] for e in ev if e.get("ph") == "M" and e.get("name") == "thread_name"}
xs = [e for e in ev if e.get("ph") == "X" and "dur" in e]
for e in xs:  # Python zones carry a " (Python)" suffix
    if e["name"].endswith(" (Python)"):
        e["name"] = e["name"][: -len(" (Python)")]
by_tid: dict[int, list] = defaultdict(list)
for e in xs:
    by_tid[e["tid"]].append(e)
main_tid = next((t for t, es in by_tid.items() if any(e["name"] == "teleop.kit" for e in es)), None)
if main_tid is None:
    raise SystemExit("no teleop.kit zones in the trace (was the server started with --kit-trace?)")


def nest(es):
    """Yield (event, path of names from the outermost zone) for one thread."""
    es = sorted(es, key=lambda e: (e["ts"], -e["dur"]))
    stack: list[tuple[float, str]] = []
    for e in es:
        while stack and stack[-1][0] <= e["ts"] + 1e-3:
            stack.pop()
        path = tuple(n for _, n in stack) + (e["name"],)
        yield e, path
        stack.append((e["ts"] + e["dur"], e["name"]))


def clean(n: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", n))[:90]  # frame numbers / ids differ per frame


main = list(nest(by_tid[main_tid]))
phases = ("teleop.pre", "teleop.kit", "teleop.post")
# whole frames: from one teleop.pre start to the next
pre = sorted((e["ts"], e["ts"] + e["dur"]) for e, p in main if p == ("teleop.pre",))
if len(pre) < 3:
    raise SystemExit("too few frames in the trace")
# the profiler records from startup (the capture mask does not gate every zone): keep the last --frames frames,
# dropping the very last one (partial)
pre = pre[-(args.frames + 1):]
win0, win1 = pre[0][0], pre[-1][0]
nf = len(pre) - 1
frame_ms = (win1 - win0) / 1e3 / nf
print(f"{nf} frames, {frame_ms:.1f} ms per frame ({1e3 / frame_ms:.1f} fps)\n")

agg: dict[tuple, float] = defaultdict(float)
for e, p in main:
    if p[0] in phases and win0 <= e["ts"] < win1 and len(p) <= args.depth + 1:
        agg[tuple(clean(n) for n in p)] += e["dur"] / 1e3 / nf

kids: dict[tuple, list] = defaultdict(list)
for p in agg:
    kids[p[:-1]].append(p)


def show(p: tuple, indent: int) -> None:
    v = agg[p]
    print(f"{'  ' * indent}{v:7.2f}  {p[-1]}")
    ch = sorted(kids.get(p, []), key=lambda c: -agg[c])
    shown = [c for c in ch if agg[c] >= args.min]
    for c in shown:
        show(c, indent + 1)
    if ch:
        rest = v - sum(agg[c] for c in ch)
        hidden = sum(agg[c] for c in ch if agg[c] < args.min)
        if rest >= args.min:
            print(f"{'  ' * (indent + 1)}{rest:7.2f}  (self)")
        if hidden >= args.min:
            print(f"{'  ' * (indent + 1)}{hidden:7.2f}  (smaller zones)")


print("main thread, ms per frame")
for ph in phases:
    if (ph,) in agg:
        show((ph,), 0)
loop_total = sum(agg.get((ph,), 0.0) for ph in phases)
print(f"{frame_ms - loop_total:7.2f}  (between the phases)\n")

# other threads: top-level busy time inside the window (idle fibers / waits do not count)
IDLE = {"Coroutine yield", "Waiting...", "Running fiber", "Wait for task", "Thread waiting..."}
rows = []
for tid, es in by_tid.items():
    if tid == main_tid:
        continue
    top: dict[str, float] = defaultdict(float)
    busy = 0.0
    for e, p in nest(es):
        if len(p) == 1 and win0 <= e["ts"] < win1 and e["name"] not in IDLE:
            busy += e["dur"]
    if busy <= 0:
        continue
    for e, p in nest(es):
        if win0 <= e["ts"] < win1 and len(p) <= 3:
            top[" > ".join(clean(n) for n in p[1:] or p)] += e["dur"] / 1e3 / nf
    rows.append((busy / 1e3 / nf, tname.get(tid, str(tid)), top))
rows.sort(key=lambda r: -r[0])
print("other threads: busy ms per frame (largest zones below)")
for busy, name, top in rows[: args.threads]:
    print(f"{busy:7.2f}  {name}")
    for z, v in sorted(top.items(), key=lambda kv: -kv[1])[:4]:
        if v >= args.min:
            print(f"           {v:7.2f}  {z}")
