"""VMC Protocol (OSC) motion messages -> the XR sample dict the JSON path produces.

The web app can send its XR samples as VMC Protocol bundles (binary OSC) over the "teleop" DataChannel instead of
JSON; see webapp/src/vmc.ts for the exact messages. Decoded here:

* ``/VMC/Ext/T f``                         -> ``ts`` (ms, the web app's performance.now clock)
* ``/VMC/Ext/Hmd/Pos s fffffff``           -> ``head``
* ``/VMC/Ext/Con/Pos s fffffff``           -> ``left`` / ``right`` (side from the serial: "Left..." / "Right...")
* ``/VMC/Ext/Con i s i i i fff``           -> ``trigger`` / ``grip`` / ``axes`` ("Trigger", "Grip", "Stick")

VMC uses Unity coordinates (left-handed, z forward); they are turned back into WebXR's (right-handed, -z forward):
p = (x, y, -z), q = (-x, -y, z, w). Other VMC addresses (bones, blend shapes, ...) are ignored.
"""

from __future__ import annotations

import struct


class OscError(ValueError):
    pass


def _string(buf: bytes, o: int) -> tuple[str, int]:
    end = buf.find(b"\0", o)
    if end < 0:
        raise OscError("unterminated string")
    return buf[o:end].decode("utf-8", "replace"), (end + 4) & ~3


def _message(buf: bytes) -> tuple[str, list]:
    addr, o = _string(buf, 0)
    if o >= len(buf):
        return addr, []
    tags, o = _string(buf, o)
    if not tags.startswith(","):
        raise OscError("missing type tags")
    args: list = []
    for t in tags[1:]:
        if t == "i":
            args.append(struct.unpack_from(">i", buf, o)[0])
            o += 4
        elif t == "f":
            args.append(struct.unpack_from(">f", buf, o)[0])
            o += 4
        elif t == "d":
            args.append(struct.unpack_from(">d", buf, o)[0])
            o += 8
        elif t == "h":
            args.append(struct.unpack_from(">q", buf, o)[0])
            o += 8
        elif t == "s":
            s, o = _string(buf, o)
            args.append(s)
        elif t == "T":
            args.append(True)
        elif t == "F":
            args.append(False)
        elif t in "NI":
            args.append(None)
        else:
            raise OscError(f"unsupported OSC type {t!r}")
    return addr, args


def iter_messages(buf: bytes, depth: int = 0):
    """(address, args) for every message in an OSC packet (nested bundles flattened)."""
    if buf.startswith(b"#bundle\0"):
        if depth > 4:
            raise OscError("bundle nesting too deep")
        o = 16
        while o + 4 <= len(buf):
            n = struct.unpack_from(">i", buf, o)[0]
            o += 4
            if n <= 0 or o + n > len(buf):
                raise OscError("bad bundle element size")
            yield from iter_messages(buf[o:o + n], depth + 1)
            o += n
    elif buf.startswith(b"/"):
        yield _message(buf)
    else:
        raise OscError("not an OSC packet")


def _pose(args: list) -> tuple[str, dict] | None:
    if len(args) < 8 or not isinstance(args[0], str):
        return None
    try:
        px, py, pz, qx, qy, qz, qw = (float(a) for a in args[1:8])
    except (TypeError, ValueError):
        return None
    return args[0], {"p": [px, py, -pz], "q": [-qx, -qy, qz, qw]}


def _side(serial: str) -> str | None:
    s = serial.lower()
    if s.startswith("left") or "left" in s:
        return "left"
    if s.startswith("right") or "right" in s:
        return "right"
    return None


def decode_sample(buf: bytes) -> dict | None:
    """The XR sample (same keys as the JSON message, ``t`` = "xr") or None if the packet holds no device pose."""
    sample: dict = {"t": "xr", "fmt": "vmc"}
    inputs: dict[str, dict] = {"left": {}, "right": {}}
    for addr, args in iter_messages(buf):
        if addr == "/VMC/Ext/T" and args and isinstance(args[0], (int, float)):
            sample["ts"] = float(args[0]) * 1000.0
        elif addr == "/VMC/Ext/Hmd/Pos":
            r = _pose(args)
            if r:
                sample["head"] = r[1]
        elif addr == "/VMC/Ext/Con/Pos":
            r = _pose(args)
            side = _side(r[0]) if r else None
            if side:
                sample[side] = {**r[1], "trigger": 0.0, "grip": 0.0}
        elif addr == "/VMC/Ext/Con" and len(args) >= 8 and isinstance(args[1], str):
            side = "left" if args[2] else "right"
            try:
                x, y = float(args[5]), float(args[6])
            except (TypeError, ValueError):
                continue
            name = args[1]
            if name == "Trigger":
                inputs[side]["trigger"] = x
            elif name == "Grip":
                inputs[side]["grip"] = x
            elif name == "Stick":
                inputs[side]["axes"] = [x, -y]  # VMC: stick up = +y, Gamepad API: up = -y
    for side, vals in inputs.items():
        if side in sample:
            sample[side].update(vals)
    if not any(k in sample for k in ("head", "left", "right")):
        return None
    sample.setdefault("ts", 0.0)
    return sample
