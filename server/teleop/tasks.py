"""Language-conditioned task templates and automatic success checks for data collection."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

# Natural-language names used in instructions (pi0 / PaliGemma are trained on English prompts).
# User-supplied models (assets/props/<set>/<name>.glb ...) use their file name ("teddy_bear" -> "teddy bear").
OBJECT_NAMES = {
    "apple_red": "red apple",
    "banana": "banana",
    "socket": "socket",
    "pulley": "pulley",
    "relay": "relay",
    "injector": "fuel injector",
    "tray": "tray",
    "coffee_cup": "coffee cup",
    "canned_peach": "can of peaches",
    "cube": "cube",
    "golf_ball": "golf ball",
    "candle": "candle",
    "rubiks_cube": "Rubik's cube",
    "tape_roll": "roll of tape",
    "screwdriver": "screwdriver",
    "torx_driver": "torx screwdriver",
    "extension_bar": "extension bar",
    "socket_small": "small socket",
    "mega_fuse": "orange fuse",
    "midi_fuse": "white fuse",
    "micro_fuse": "green fuse",
    "bolt_m20": "large bolt",
    "nut_m20": "large nut",
    "bolt_m16": "small bolt",
    "nut_m16": "small nut",
    "bearing": "bearing",
    "bearing_small": "small bearing",
    "valve_lifter": "valve lifter",
    "tape": "tape",
}

EXTRA_PICKABLE = ["canned_peach", "cube", "coffee_cup"]  # always-present items that can also be picked up


def object_name(obj_id: str) -> str:
    return OBJECT_NAMES.get(obj_id, obj_id.replace("_", " "))


@dataclass
class TaskTemplate:
    id: str
    label: str  # UI label (Japanese)
    template: str  # instruction; {obj} is replaced by the object's name
    targets: str  # "set" = objects of the active set, "set+items" = also the can / cube / cup, "" = none
    success: str | None  # "in_tray" | "lifted" | "near_cup" | None (manual)


TEMPLATES: list[TaskTemplate] = [
    TaskTemplate("put_in_tray", "◯◯をトレイに入れる", "put the {obj} in the tray", "set", "in_tray"),
    TaskTemplate("pick_up", "◯◯を持ち上げる", "pick up the {obj}", "set+items", "lifted"),
    TaskTemplate("place_next_to_cup", "◯◯をカップの横に置く", "place the {obj} next to the coffee cup", "set", "near_cup"),
    TaskTemplate("free", "自由入力（手動判定）", "{text}", "", None),
]


def template_objects(tmpl: TaskTemplate, set_objects: list[str], available: list[str]) -> list[str]:
    if not tmpl.targets:
        return []
    objs = list(set_objects)
    if tmpl.targets == "set+items":
        objs += [o for o in EXTRA_PICKABLE if o in available]
    return objs


def list_templates(set_objects: list[str], available: list[str]) -> list[dict]:
    return [
        {
            "id": t.id,
            "label": t.label,
            "template": t.template,
            "objects": [{"id": o, "name": object_name(o)} for o in template_objects(t, set_objects, available)],
            "autoSuccess": t.success is not None,
        }
        for t in TEMPLATES
    ]


@dataclass
class TaskInstance:
    template: TaskTemplate
    obj: str | None
    text: str

    def to_dict(self) -> dict:
        return {"id": self.template.id, "object": self.obj, "text": self.text, "autoSuccess": self.template.success is not None}


def make_task(spec: dict, set_objects: list[str], available: list[str], rng: random.Random) -> TaskInstance:
    """spec: {"id": template id, "object": id | "random" | None, "text": free text}"""
    tid = spec.get("id") or TEMPLATES[0].id
    # unknown ids (e.g. "put_in_bowl" remembered by an older web app) fall back to the first template
    tmpl = next((t for t in TEMPLATES if t.id == tid), TEMPLATES[0])
    if tmpl.id == "free":
        text = str(spec.get("text") or "").strip()
        if not text:
            raise ValueError("free task requires instruction text")
        return TaskInstance(tmpl, None, text[:300])
    candidates = template_objects(tmpl, set_objects, available)
    if not candidates:
        raise ValueError("no object available for this task (object set is empty)")
    obj = spec.get("object") or "random"
    if obj not in candidates:  # "random", or an object of another set
        obj = rng.choice(candidates)
    return TaskInstance(tmpl, obj, tmpl.template.format(obj=object_name(obj)))


def check_success(task: TaskInstance, sim, gripper_closed: bool) -> bool:
    """Instantaneous success predicate (the caller requires it to hold for a while)."""
    kind = task.template.success
    centers = sim.prop_centers()
    table_top = sim.table_top
    if kind is None or task.obj not in centers:
        return False
    p = centers[task.obj]
    if kind == "in_tray":
        tray = next((t for t in sim.props if t.name == "tray"), None)
        if tray is None or "tray" not in centers:
            return False
        # tray footprint: the tray is placed rotated by 90 deg, test in its own frame with a 3 cm margin
        pos, quat = sim._prop_poses()  # noqa: SLF001
        from .mathutil import quat_wxyz_to_matrix

        R = quat_wxyz_to_matrix(quat[sim.props.index(tray)])
        local = R.T @ (p - centers["tray"])
        half = np.abs(R.T @ tray.size) / 2  # world AABB size -> approx. local extents
        inside = abs(local[0]) < half[0] - 0.03 and abs(local[1]) < half[1] - 0.03
        return bool(inside and p[2] < table_top + tray.size[2] + 0.05 and not gripper_closed)
    if kind == "lifted":
        return bool(p[2] > table_top + 0.12 and gripper_closed)
    if kind == "near_cup":
        c = centers.get("coffee_cup")
        if c is None:
            return False
        d = np.linalg.norm(p[:2] - c[:2])
        return bool(0.06 < d < 0.16 and p[2] < table_top + 0.08 and not gripper_closed)
    return False
