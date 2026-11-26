"""Consistency checks for a recorded LeRobot v2.1 dataset (no lerobot install needed).

    C:\\isaacsim\\python.bat server\\tools\\validate_dataset.py datasets\\crx10ial_teleop
"""

from __future__ import annotations

import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "_deps"))

import av  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

root = sys.argv[1]
info = json.load(open(os.path.join(root, "meta", "info.json"), encoding="utf-8"))
episodes = [json.loads(line) for line in open(os.path.join(root, "meta", "episodes.jsonl"), encoding="utf-8")]
tasks = [json.loads(line) for line in open(os.path.join(root, "meta", "tasks.jsonl"), encoding="utf-8")]
stats = [json.loads(line) for line in open(os.path.join(root, "meta", "episodes_stats.jsonl"), encoding="utf-8")]
fps = info["fps"]
errors: list[str] = []
print(f"codebase {info['codebase_version']} robot {info['robot_type']} fps {fps}")
print(f"episodes {info['total_episodes']} frames {info['total_frames']} tasks {info['total_tasks']} videos {info['total_videos']}")
video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]

if info["total_episodes"] != len(episodes) or len(stats) != len(episodes):
    errors.append("episode count mismatch between info / episodes.jsonl / episodes_stats.jsonl")
global_index = 0
for ep in episodes:
    i = ep["episode_index"]
    chunk = i // info["chunks_size"]
    table = pq.read_table(os.path.join(root, info["data_path"].format(episode_chunk=chunk, episode_index=i)))
    df = {c: table.column(c).to_pylist() for c in table.column_names}
    n = table.num_rows
    if n != ep["length"]:
        errors.append(f"ep {i}: parquet rows {n} != length {ep['length']}")
    ts = np.array(df["timestamp"])
    if not np.allclose(np.diff(ts), 1.0 / fps, atol=1e-4):
        errors.append(f"ep {i}: timestamps not spaced 1/fps")
    if df["frame_index"] != list(range(n)) or df["index"] != list(range(global_index, global_index + n)):
        errors.append(f"ep {i}: frame_index/index not contiguous")
    if set(df["episode_index"]) != {i}:
        errors.append(f"ep {i}: episode_index column wrong")
    task_names = {tasks[t]["task"] for t in set(df["task_index"])}
    if task_names != set(ep["tasks"]):
        errors.append(f"ep {i}: task mismatch {task_names} vs {ep['tasks']}")
    for k, f in info["features"].items():
        if f["dtype"] == "float32" and k in df and f["shape"] != [1]:
            arr = np.array(df[k], dtype=np.float32)
            if arr.shape != (n, f["shape"][0]) or not np.isfinite(arr).all():
                errors.append(f"ep {i}: {k} shape {arr.shape} / non-finite values")
    for vk in video_keys:
        path = os.path.join(root, info["video_path"].format(episode_chunk=chunk, video_key=vk, episode_index=i))
        with av.open(path) as c:
            s = c.streams.video[0]
            frames = sum(1 for _ in c.decode(video=0))
            shape = (s.codec_context.height, s.codec_context.width)
            codec = s.codec_context.name
        if frames != n:
            errors.append(f"ep {i}: {vk} has {frames} frames, expected {n}")
        if list(shape) != info["features"][vk]["shape"][:2]:
            errors.append(f"ep {i}: {vk} size {shape}")
    act = np.array(df["action"])
    st = np.array(df["observation.state"])
    print(
        f"ep {i}: {n} frames ({n / fps:.1f}s) task={ep['tasks'][0]!r} video codec={codec} "
        f"gripper cmd closed {int(act[:, 6].sum())} frames, max |action-state| joints {np.abs(act[:, :6] - st[:, :6]).max():.3f} rad"
    )
    global_index += n

print("OK" if not errors else "ERRORS:\n  " + "\n  ".join(errors))
sys.exit(1 if errors else 0)
