"""Episode recording for VLA fine-tuning (pi0 / pi0-FAST via openpi, LeRobot v2.1 format).

Per frame (sampled on *simulation* time at ``record_fps``):

    observation.images.base   third-person camera ("Operator" camera, incl. camera DR jitter)
    observation.images.wrist  wrist camera on the flange
    observation.state         [J1..J6 (rad), gripper (0=open..1=closed)]             float32[7]
    action                    [J1..J6 target (rad), gripper command (0/1)]           float32[7]
    observation.ee_pose       TCP pose in the robot base frame [x y z qw qx qy qz]   float32[7]
    action.ee_pose            commanded TCP pose in the robot base frame             float32[7]

``action[t]`` is the command issued right after ``observation[t]`` was captured (the IK result of the next
control step), i.e. exactly what a policy should output given that observation.

Episode flow: rec_start -> reset (+ domain randomization) -> settle -> wait for the operator's first motion
-> record -> stop (auto success / manual success / failure / discard / timeout).
"""

from __future__ import annotations

import copy
import logging
import os
import random
import time
from typing import TYPE_CHECKING

import numpy as np
import omni.replicator.core as rep

from . import config as cfg
from .lerobot_writer import FeatureSpec, LeRobotWriter
from .mathutil import matrix_to_quat_wxyz
from .tasks import TaskInstance, check_success, list_templates, make_task

if TYPE_CHECKING:
    from .sim import SimController

log = logging.getLogger("teleop.rec")

JOINT_NAMES = ["J1", "J2", "J3", "J4", "J5", "J6"]
SETTLE_SECONDS = 0.8
SUCCESS_HOLD_SECONDS = 0.6


class RecordingManager:
    def __init__(self, sim: SimController, scfg: cfg.ServerConfig) -> None:
        self.sim = sim
        self.scfg = scfg
        self.fps = scfg.record_fps
        h, w = scfg.record_height, scfg.record_width
        root = os.path.join(scfg.dataset_root, scfg.dataset_name)
        self.writer = LeRobotWriter(
            root,
            fps=self.fps,
            robot_type="fanuc_crx10ia_l_robotiq_2f85" if sim.gripper_type == "robotiq_2f85" else "fanuc_crx10ia_l",
            codec=scfg.record_codec,
            features=[
                FeatureSpec("observation.images.base", "video", (h, w, 3)),
                FeatureSpec("observation.images.wrist", "video", (h, w, 3)),
                FeatureSpec("observation.state", "float32", (7,), JOINT_NAMES + ["gripper"]),
                FeatureSpec("action", "float32", (7,), JOINT_NAMES + ["gripper"]),
                FeatureSpec("observation.ee_pose", "float32", (7,), ["x", "y", "z", "qw", "qx", "qy", "qz"]),
                FeatureSpec("action.ee_pose", "float32", (7,), ["x", "y", "z", "qw", "qx", "qy", "qz"]),
            ],
        )
        # dedicated low-resolution render products for the dataset (independent of the VR stream)
        self.cams = {
            "observation.images.base": f"{cfg.CAMERAS_ROOT}/Dataset"
            if sim.stage.GetPrimAtPath(f"{cfg.CAMERAS_ROOT}/Dataset")
            else f"{cfg.CAMERAS_ROOT}/Operator",
            "observation.images.wrist": cfg.WRIST_CAMERA,
        }
        self.annotators: dict[str, object] = {}
        self.render_products = []
        for key, cam in self.cams.items():
            rp = rep.create.render_product(cam, (w, h), name=f"Record_{key.split('.')[-1]}")
            annot = rep.AnnotatorRegistry.get_annotator("rgb")
            annot.attach(rp)
            self.annotators[key] = annot
            self.render_products.append(rp)
        self._rp_active = True
        self._set_rp_active(False)  # only render the dataset cameras while an episode is in progress
        base = sim.stage.GetPrimAtPath(cfg.WORLD).GetAttribute("teleop:robotBase").Get()
        self.base = np.array(base if base is not None else (0, 0, 0), dtype=float)
        self.rng = random.Random()
        self.settings = {
            "waitForMotion": True,  # start recording at the operator's first clutch / gripper action
            "saveFailed": False,  # keep failed episodes (flagged success=false in episodes_extra.jsonl)
            "autoNext": False,  # immediately prepare the next episode after a save
            "autoSuccess": True,  # stop automatically when the task's success predicate holds
            "maxDuration": 120.0,  # seconds of sim time
            "randomize": True,  # domain randomization at episode start
        }
        self.state = "idle"  # idle | preparing | armed | recording
        self.task: TaskInstance | None = None
        self.last_spec: dict | None = None
        self.episode_index: int | None = None
        self.t_state = 0.0
        self.t_start = 0.0
        self.next_sample = 0.0
        self.frames = 0
        self.pending: tuple[dict, dict] | None = None
        self.success_since: float | None = None
        self.motion_ref: tuple[bool, bool] | None = None
        self.meta: dict = {}
        self.message = ""
        self._auto_next_at: float | None = None
        log.info("recording to %s (%d fps, %dx%d, codec %s)", root, self.fps, w, h, scfg.record_codec)

    # ---------------------------------------------------------------------------------------------
    def _set_rp_active(self, active: bool) -> None:
        if active == self._rp_active:
            return
        self._rp_active = active
        for rp in self.render_products:
            try:
                rp.hydra_texture.set_updates_enabled(active)
            except Exception as exc:  # noqa: BLE001
                log.warning("could not toggle render product: %s", exc)

    def _set_objects(self) -> list[str]:
        return [p.name for p in self.sim.props if p.category == self.sim.object_set]

    def _now(self) -> float:
        return float(self.sim.timeline.get_current_time())

    def status(self) -> dict:
        return {
            "state": self.state,
            "task": self.task.to_dict() if self.task else None,
            "episode": self.episode_index,
            "frames": self.frames,
            "duration": round(self.frames / self.fps, 2),
            "message": self.message,
            "settings": dict(self.settings),
            "dataset": self.writer.summary(),
        }

    def command(self, name: str, payload: dict):
        if name == "task_templates":
            return {
                "templates": list_templates(self._set_objects(), self.sim.active_prop_names()),
                "objects": self.sim.active_prop_names(),
                "objectSet": self.sim.object_set,
            }
        if name == "rec_status":
            return self.status()
        if name == "rec_settings":
            for k in self.settings:
                if k in payload:
                    self.settings[k] = type(self.settings[k])(payload[k])
            self.settings["maxDuration"] = float(min(900.0, max(5.0, self.settings["maxDuration"])))
            return dict(self.settings)
        if name == "rec_start":
            return self.start(payload.get("task") or self.last_spec or {"id": "put_in_tray", "object": "random"})
        if name == "rec_stop":
            return self.stop(bool(payload.get("success", True)), reason="manual")
        if name == "rec_discard":
            return self.discard()
        if name == "rec_delete_last":
            return self.writer.delete_last_episode()
        raise ValueError(f"unknown recording command {name}")

    # ---------------------------------------------------------------------------------------------
    def start(self, spec: dict) -> dict:
        if self.state != "idle":
            raise RuntimeError(f"already {self.state}")
        task = make_task(spec, self._set_objects(), self.sim.active_prop_names(), self.rng)
        self.last_spec = dict(spec)
        self.task = task
        self._auto_next_at = None
        self.sim.reset(randomize=bool(self.settings["randomize"]))
        self.state = "preparing"
        self._set_rp_active(True)
        self.t_state = self._now()
        self.frames = 0
        self.pending = None
        self.success_since = None
        self.message = "準備中…"
        log.info("episode prepared: %r", task.text)
        return self.status()

    def _begin(self) -> None:
        sim = self.sim
        self.episode_index = self.writer.begin_episode(self.task.text)
        self.state = "recording"
        self.t_start = self._now()
        self.next_sample = self.t_start
        self.frames = 0
        centers = sim.prop_centers()
        dr = sim.randomizer
        self.meta = {
            "task_id": self.task.template.id,
            "object": self.task.obj,
            "operator": None,
            "gripper": sim.gripper_type,
            "dr_config": copy.deepcopy(dr.config) if dr else None,
            "dr_count": dr.count if dr else None,
            "mirror_visible": dr.mirror_visible if dr else None,
            "initial_objects": {k: [round(float(x), 4) for x in v] for k, v in centers.items()},
            "teleop_settings": dict(sim.settings),
            "started_at": time.time(),
        }
        self.message = "録画中"
        log.info("recording episode %d", self.episode_index)

    def stop(self, success: bool, reason: str) -> dict:
        if self.state in ("preparing", "armed"):
            self.state = "idle"
            self.message = "キャンセルしました"
            return self.status()
        if self.state != "recording":
            raise RuntimeError("not recording")
        self.pending = None
        save = success or bool(self.settings["saveFailed"])
        extra = {
            **self.meta,
            "success": success,
            "end_reason": reason,
            "duration_s": round(self.frames / self.fps, 3),
        }
        self.writer.end_episode(save, extra)
        self.message = (
            f"エピソード {self.episode_index} を{'保存' if save else '破棄'}しました"
            f"（{'成功' if success else '失敗'} / {reason} / {self.frames} frames）"
        )
        log.info(self.message)
        self.state = "idle"
        if save and success and self.settings["autoNext"]:
            self._auto_next_at = time.monotonic() + 1.5
        return self.status()

    def discard(self) -> dict:
        if self.state == "recording":
            self.writer.end_episode(False)
        self.state = "idle"
        self.pending = None
        self.message = "破棄しました"
        return self.status()

    # ---------------------------------------------------------------------------------------------
    def on_pre_update(self) -> None:
        """Called after the IK/gripper command of this control step: completes the pending frame."""
        if self.pending is None or self.state != "recording":
            return
        images, values = self.pending
        self.pending = None
        sim = self.sim
        q_cmd = sim.q_cmd if sim.q_cmd is not None else sim.joint_positions()
        values["action"] = np.concatenate([q_cmd, [1.0 if sim.gripper_cmd_closed else 0.0]])
        T = sim.target_T
        values["action.ee_pose"] = np.concatenate([T[:3, 3] - self.base, matrix_to_quat_wxyz(T[:3, :3])])
        self.writer.add_frame(images, values)
        self.frames += 1

    def on_post_update(self) -> None:
        sim = self.sim
        now = self._now()
        if self._auto_next_at is not None and time.monotonic() >= self._auto_next_at and self.state == "idle":
            self._auto_next_at = None
            try:
                self.start(self.last_spec or {})
            except Exception as exc:  # noqa: BLE001
                self.message = f"自動開始に失敗: {exc}"
        if self.state == "idle":
            self._set_rp_active(False)
            return
        if self.state == "preparing":
            if now - self.t_state >= SETTLE_SECONDS:
                if self.settings["waitForMotion"]:
                    self.state = "armed"
                    self.motion_ref = (sim.clutch, sim.gripper_cmd_closed)
                    self.message = "操作を開始すると録画されます"
                else:
                    self._begin()
            return
        if self.state == "armed":
            moved = sim.clutch or sim.gripper_cmd_closed != self.motion_ref[1]
            if moved:
                self._begin()
            else:
                return
        if self.state != "recording":
            return
        if now + 1e-6 < self.next_sample:
            return
        # keep the sample grid aligned to the recording rate even if the sim stepped irregularly
        self.next_sample += 1.0 / self.fps
        if now - self.next_sample > 1.0 / self.fps:
            self.next_sample = now + 1.0 / self.fps
        images = {}
        for key, annot in self.annotators.items():
            data = annot.get_data()
            if data is None or not hasattr(data, "ndim") or data.ndim != 3:
                return
            images[key] = np.ascontiguousarray(data[:, :, :3])
        q = sim.joint_positions()
        T = sim.tcp_transform()
        values = {
            "observation.state": np.concatenate([q, [sim.gripper_pos if sim.gripper_type != "virtual" else float(sim.gripper_closed)]]),
            "observation.ee_pose": np.concatenate([T[:3, 3] - self.base, matrix_to_quat_wxyz(T[:3, :3])]),
        }
        self.pending = (images, values)

        # termination checks
        duration = self.frames / self.fps
        if duration >= self.settings["maxDuration"]:
            self.stop(False, "timeout")
            return
        if self.settings["autoSuccess"] and self.task and self.task.template.success:
            ok = check_success(self.task, sim, sim.gripper_closed)
            if ok:
                self.success_since = self.success_since if self.success_since is not None else now
                if now - self.success_since >= SUCCESS_HOLD_SECONDS:
                    # make sure the last observation (showing success) is written
                    self.on_pre_update()
                    self.stop(True, "auto")
            else:
                self.success_since = None

    def close(self) -> None:
        if self.state == "recording":
            self.writer.end_episode(False)
        self.writer.close()
