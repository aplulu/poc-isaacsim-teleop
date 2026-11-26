"""Minimal LeRobot dataset writer (codebase v2.1 layout) without depending on the ``lerobot`` package.

Layout (what ``lerobot.datasets.LeRobotDataset`` / openpi expect)::

    <root>/
      meta/info.json
      meta/tasks.jsonl            {"task_index": 0, "task": "put the banana in the bowl"}
      meta/episodes.jsonl         {"episode_index": 0, "tasks": [...], "length": 312}
      meta/episodes_stats.jsonl   per-episode feature statistics (v2.1)
      meta/stats.json             aggregated statistics (v2.0 compatible)
      meta/episodes_extra.jsonl   NOT part of LeRobot: success flag, DR settings, operator, ... (ignored by loaders)
      data/chunk-000/episode_000000.parquet
      videos/chunk-000/<video_key>/episode_000000.mp4

All heavy work (video encoding, parquet, stats) runs on a worker thread so the simulation never blocks.
To use the dataset with LeRobot >= 0.4 (v3.0 format) run its v2.1 -> v3.0 converter.
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass, field

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger("teleop.lerobot")

CODEBASE_VERSION = "v2.1"
CHUNKS_SIZE = 1000
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

CODECS = {
    # ffmpeg encoder -> (codec name stored in info.json, encoder options). LeRobot default: AV1, g=2, crf=30
    "libsvtav1": ("av1", {"crf": "30", "g": "2", "preset": "8"}),
    "libx264": ("h264", {"crf": "23", "g": "2", "preset": "veryfast"}),
}


@dataclass
class FeatureSpec:
    key: str
    dtype: str  # "video" | "float32" | "int64"
    shape: tuple[int, ...]
    names: list[str] | None = None


@dataclass
class _Episode:
    index: int
    task: str
    task_index: int
    containers: dict = field(default_factory=dict)
    streams: dict = field(default_factory=dict)
    rows: dict = field(default_factory=dict)
    image_samples: dict = field(default_factory=dict)
    length: int = 0
    queued: int = 0  # frames handed to the worker (counted on the simulation thread)
    extra: dict = field(default_factory=dict)


def _jsonl_read(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _jsonl_write(path: str, rows: list[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _num_stats(x: np.ndarray) -> dict:
    x = x.astype(np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return {
        "min": x.min(0).tolist(),
        "max": x.max(0).tolist(),
        "mean": x.mean(0).tolist(),
        "std": x.std(0).tolist(),
        "count": [int(x.shape[0])],
    }


def _img_stats(samples: list[np.ndarray]) -> dict:
    """Per-channel stats in [0, 1], shaped (3, 1, 1) like LeRobot's compute_episode_stats."""
    x = np.stack(samples).astype(np.float64) / 255.0  # (N, H, W, 3)
    c = x.reshape(-1, 3)

    def chw(v):
        return [[[float(a)]] for a in v]

    return {
        "min": chw(c.min(0)),
        "max": chw(c.max(0)),
        "mean": chw(c.mean(0)),
        "std": chw(c.std(0)),
        "count": [len(samples)],
    }


def _aggregate(stats_list: list[dict]) -> dict:
    """Weighted aggregation of per-episode stats (same maths as lerobot.datasets.compute_stats)."""
    out: dict = {}
    keys = set().union(*(s.keys() for s in stats_list)) if stats_list else set()
    for k in keys:
        items = [s[k] for s in stats_list if k in s]
        counts = np.array([i["count"][0] for i in items], dtype=np.float64)
        means = np.array([i["mean"] for i in items], dtype=np.float64)
        stds = np.array([i["std"] for i in items], dtype=np.float64)
        w = counts.reshape((-1,) + (1,) * (means.ndim - 1)) / counts.sum()
        mean = (means * w).sum(0)
        var = ((stds**2 + (means - mean) ** 2) * w).sum(0)
        out[k] = {
            "min": np.min(np.array([i["min"] for i in items]), 0).tolist(),
            "max": np.max(np.array([i["max"] for i in items]), 0).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(var).tolist(),
            "count": [int(counts.sum())],
        }
    return out


class LeRobotWriter:
    def __init__(
        self,
        root: str,
        *,
        fps: int,
        features: list[FeatureSpec],
        robot_type: str,
        codec: str = "libsvtav1",
    ) -> None:
        self.root = root
        self.fps = fps
        self.features = features
        self.robot_type = robot_type
        self.codec = codec if codec in CODECS else "libx264"
        self.video_keys = [f.key for f in features if f.dtype == "video"]
        self.data_keys = [f for f in features if f.dtype != "video"]
        os.makedirs(os.path.join(root, "meta"), exist_ok=True)
        self._lock = threading.RLock()
        self._load_meta()
        self._q: queue.Queue = queue.Queue(maxsize=600)
        self._ep: _Episode | None = None
        self._busy = 0
        self.last_error: str | None = None
        self._thread = threading.Thread(target=self._worker, name="lerobot-writer", daemon=True)
        self._thread.start()

    # -- metadata -----------------------------------------------------------------------------------
    def _meta(self, name: str) -> str:
        return os.path.join(self.root, "meta", name)

    def _load_meta(self) -> None:
        info_path = self._meta("info.json")
        if os.path.exists(info_path):
            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)
            mine = {f.key: list(f.shape) for f in self.features}
            theirs = {k: v["shape"] for k, v in info["features"].items() if k not in _DEFAULT_KEYS}
            if info.get("fps") != self.fps or mine != theirs:
                raise RuntimeError(
                    f"Existing dataset at {self.root} has different fps/features "
                    f"(fps {info.get('fps')} vs {self.fps}). Use another --dataset name."
                )
        self.tasks = {t["task"]: t["task_index"] for t in _jsonl_read(self._meta("tasks.jsonl"))}
        self.episodes = _jsonl_read(self._meta("episodes.jsonl"))
        self.episode_stats = _jsonl_read(self._meta("episodes_stats.jsonl"))
        self.extras = _jsonl_read(self._meta("episodes_extra.jsonl"))

    @property
    def total_frames(self) -> int:
        return sum(e["length"] for e in self.episodes)

    def _write_info(self) -> None:
        n = len(self.episodes)
        feats: dict = {}
        for f in self.features:
            d: dict = {"dtype": f.dtype, "shape": list(f.shape), "names": f.names}
            if f.dtype == "video":
                h, w, c = f.shape
                d["names"] = ["height", "width", "channels"]
                d["info"] = {
                    "video.height": h,
                    "video.width": w,
                    "video.codec": CODECS[self.codec][0],
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": self.fps,
                    "video.channels": c,
                    "has_audio": False,
                }
            feats[f.key] = d
        for k, dtype in _DEFAULT_KEYS.items():
            feats[k] = {"dtype": dtype, "shape": [1], "names": None}
        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "total_episodes": n,
            "total_frames": self.total_frames,
            "total_tasks": len(self.tasks),
            "total_videos": n * len(self.video_keys),
            "total_chunks": max(1, math.ceil(n / CHUNKS_SIZE)) if n else 0,
            "chunks_size": CHUNKS_SIZE,
            "fps": self.fps,
            "splits": {"train": f"0:{n}"},
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH if self.video_keys else None,
            "features": feats,
        }
        tmp = self._meta("info.json.tmp")
        with open(tmp, "w", encoding="utf-8") as fp:
            json.dump(info, fp, indent=2)
        os.replace(tmp, self._meta("info.json"))
        _jsonl_write(self._meta("tasks.jsonl"), [{"task_index": i, "task": t} for t, i in sorted(self.tasks.items(), key=lambda x: x[1])])
        _jsonl_write(self._meta("episodes.jsonl"), self.episodes)
        _jsonl_write(self._meta("episodes_stats.jsonl"), self.episode_stats)
        _jsonl_write(self._meta("episodes_extra.jsonl"), self.extras)
        with open(self._meta("stats.json"), "w", encoding="utf-8") as fp:
            json.dump(_aggregate([e["stats"] for e in self.episode_stats]), fp)

    def summary(self) -> dict:
        with self._lock:
            return {
                "root": self.root,
                "episodes": len(self.episodes),
                "frames": self.total_frames,
                "tasks": len(self.tasks),
                "fps": self.fps,
                "codec": CODECS[self.codec][0],
                "pending": self._busy,
                "lastError": self.last_error,
                "recent": [
                    {
                        "episode": x.get("episode_index"),
                        "task": x.get("task"),
                        "length": x.get("length"),
                        "success": x.get("success"),
                        "duration": x.get("duration_s"),
                    }
                    for x in self.extras[-10:][::-1]
                ],
            }

    # -- episode API (called from the simulation thread) ---------------------------------------------
    def begin_episode(self, task: str) -> int:
        with self._lock:
            if self._ep is not None:
                raise RuntimeError("an episode is already open")
            if task not in self.tasks:
                self.tasks[task] = len(self.tasks)
            index = len(self.episodes) + self._busy
            self._ep = _Episode(index=index, task=task, task_index=self.tasks[task])
        self._q.put(("begin", self._ep))
        return index

    def add_frame(self, images: dict[str, np.ndarray], values: dict[str, np.ndarray]) -> None:
        if self._ep is None:
            return
        try:
            self._q.put_nowait(("frame", self._ep, images, values))
            self._ep.queued += 1
        except queue.Full:
            log.warning("writer queue full - dropping frame")

    def end_episode(self, save: bool, extra: dict | None = None) -> None:
        ep, self._ep = self._ep, None
        if ep is None:
            return
        ep.extra = extra or {}
        save = save and ep.queued >= 2  # decided here so episode indices never get gaps
        if save:
            with self._lock:
                self._busy += 1
        self._q.put(("end", ep, save))

    @property
    def recording(self) -> bool:
        return self._ep is not None

    def delete_last_episode(self) -> dict:
        with self._lock:
            if self._busy or self._ep is not None:
                raise RuntimeError("cannot delete while recording/saving")
            if not self.episodes:
                raise RuntimeError("dataset is empty")
            ep = self.episodes.pop()
            idx = ep["episode_index"]
            self.episode_stats = [s for s in self.episode_stats if s["episode_index"] != idx]
            self.extras = [x for x in self.extras if x.get("episode_index") != idx]
            self._remove_files(idx)
            self._write_info()
            return {"deleted": idx}

    def _remove_files(self, idx: int) -> None:
        chunk = idx // CHUNKS_SIZE
        p = os.path.join(self.root, DATA_PATH.format(episode_chunk=chunk, episode_index=idx))
        if os.path.exists(p):
            os.remove(p)
        for k in self.video_keys:
            v = os.path.join(self.root, VIDEO_PATH.format(episode_chunk=chunk, video_key=k, episode_index=idx))
            if os.path.exists(v):
                os.remove(v)

    def close(self) -> None:
        if self._ep is not None:
            self.end_episode(False)
        self._q.put(None)
        self._thread.join(timeout=30)

    # -- worker ------------------------------------------------------------------------------------
    def _worker(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            try:
                kind = item[0]
                if kind == "begin":
                    self._w_begin(item[1])
                elif kind == "frame":
                    self._w_frame(item[1], item[2], item[3])
                elif kind == "end":
                    self._w_end(item[1], item[2])
            except Exception as exc:  # noqa: BLE001
                log.exception("dataset writer error")
                self.last_error = str(exc)

    def _w_begin(self, ep: _Episode) -> None:
        chunk = ep.index // CHUNKS_SIZE
        enc = self.codec
        opts = CODECS[self.codec][1]
        for f in self.features:
            if f.dtype != "video":
                continue
            path = os.path.join(self.root, VIDEO_PATH.format(episode_chunk=chunk, video_key=f.key, episode_index=ep.index))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            c = av.open(path, "w")
            s = c.add_stream(enc, rate=self.fps, options=dict(opts))
            s.height, s.width = f.shape[0], f.shape[1]
            s.pix_fmt = "yuv420p"
            ep.containers[f.key] = c
            ep.streams[f.key] = s
            ep.image_samples[f.key] = []
        for f in self.data_keys:
            ep.rows[f.key] = []

    def _w_frame(self, ep: _Episode, images: dict, values: dict) -> None:
        i = ep.length
        for key, s in ep.streams.items():
            img = images[key]
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = i
            for pkt in s.encode(frame):
                ep.containers[key].mux(pkt)
            if i % 5 == 0:  # subsample for image statistics (downscaled)
                ep.image_samples[key].append(img[::4, ::4])
        for f in self.data_keys:
            ep.rows[f.key].append(np.asarray(values[f.key], dtype=np.float32).reshape(f.shape))
        ep.length += 1

    def _close_videos(self, ep: _Episode) -> None:
        for key, s in ep.streams.items():
            try:
                for pkt in s.encode():
                    ep.containers[key].mux(pkt)
            finally:
                ep.containers[key].close()

    def _w_end(self, ep: _Episode, save: bool) -> None:
        self._close_videos(ep)
        if not save:
            self._remove_files(ep.index)
            return
        try:
            with self._lock:
                base_index = self.total_frames
            n = ep.length
            cols: dict = {}
            stats: dict = {}
            for f in self.data_keys:
                arr = np.stack(ep.rows[f.key]).astype(np.float32).reshape(n, -1)
                cols[f.key] = pa.FixedSizeListArray.from_arrays(pa.array(arr.reshape(-1), pa.float32()), arr.shape[1])
                stats[f.key] = _num_stats(arr)
            frame_index = np.arange(n, dtype=np.int64)
            ts = (frame_index / self.fps).astype(np.float32)
            extra_cols = {
                "timestamp": pa.array(ts, pa.float32()),
                "frame_index": pa.array(frame_index, pa.int64()),
                "episode_index": pa.array(np.full(n, ep.index, np.int64), pa.int64()),
                "index": pa.array(base_index + frame_index, pa.int64()),
                "task_index": pa.array(np.full(n, ep.task_index, np.int64), pa.int64()),
            }
            stats["timestamp"] = _num_stats(ts)
            stats["frame_index"] = _num_stats(frame_index)
            stats["episode_index"] = _num_stats(np.full(n, ep.index))
            stats["index"] = _num_stats(base_index + frame_index)
            stats["task_index"] = _num_stats(np.full(n, ep.task_index))
            for key, samples in ep.image_samples.items():
                if samples:
                    stats[key] = _img_stats(samples)
            table = pa.table({**cols, **extra_cols})
            chunk = ep.index // CHUNKS_SIZE
            path = os.path.join(self.root, DATA_PATH.format(episode_chunk=chunk, episode_index=ep.index))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            pq.write_table(table, path)
            with self._lock:
                self.episodes.append({"episode_index": ep.index, "tasks": [ep.task], "length": n})
                self.episode_stats.append({"episode_index": ep.index, "stats": stats})
                self.extras.append(
                    {"episode_index": ep.index, "task": ep.task, "length": n, "saved_at": time.time(), **ep.extra}
                )
                self._write_info()
            log.info("saved episode %d (%d frames, task=%r)", ep.index, n, ep.task)
        finally:
            with self._lock:
                self._busy -= 1


_DEFAULT_KEYS = {
    "timestamp": "float32",
    "frame_index": "int64",
    "episode_index": "int64",
    "index": "int64",
    "task_index": "int64",
}


def wipe_dataset(root: str) -> None:
    if os.path.isdir(root):
        shutil.rmtree(root)
