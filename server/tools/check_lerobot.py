"""Load a recorded dataset with the real lerobot package (run in a venv with lerobot 0.3.x installed).

    python server/tools/check_lerobot.py datasets/crx10ial_teleop
"""
import sys
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = Path(sys.argv[1]).resolve()
ds = LeRobotDataset(repo_id=f"local/{root.name}", root=root, video_backend="pyav")
print("episodes", ds.num_episodes, "frames", ds.num_frames, "fps", ds.fps)
print("features", list(ds.features))
item = ds[0]
for k, v in item.items():
    print(f"  {k}: {getattr(v, 'shape', v) if hasattr(v, 'shape') else v}")
last = ds[len(ds) - 1]
print("last frame task:", last["task"], "episode", int(last["episode_index"]))
print("stats keys:", sorted(ds.meta.stats)[:8])
# action chunk as used by pi0 / pi0-FAST (delta_timestamps over the next 10 steps)
ds2 = LeRobotDataset(
    repo_id=f"local/{root.name}", root=root, video_backend="pyav",
    delta_timestamps={"action": [i / ds.fps for i in range(10)]},
)
print("action chunk shape:", tuple(ds2[5]["action"].shape))
print("LEROBOT_OK")
