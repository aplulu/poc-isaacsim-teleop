"""The avatar's per-frame IK (hips, spine, neck/head, arms, fingers, legs) as one NVIDIA Warp kernel on the CPU.

Same algorithm as the numpy path in ``Avatar._solve`` (which stays as the reference): Python still works out the
targets that do not depend on the skeleton (scale, body yaw, head / hand / foot targets, foot stepping state), and
this kernel does everything that walks the skeleton. The numpy path spent ~3.5 ms in ~3000 tiny numpy calls
(``linalg.solve`` per bone, FK refreshes, 3-vector helpers); natively the whole pass is a few microseconds.

The kernel writes the model-space locals into the spring solver's ``local`` buffer, so the spring step that follows
(FK, colliders, springs, SkelAnimation values) reads them without a copy.
"""

from __future__ import annotations

import sys

import numpy as np
import warp as wp

from .avatar_springs import DEVICE, _compose, _rot

# bones[]: 0 hips, 1 neck, 2 head, 3..5 spine chain, 6 spine count, 7 + 3*side arm (upper, lower, hand),
# 13 + 3*side leg (upper, lower, foot); side 0 = left, 1 = right
# kv[] (vec3): 0 neck0 - hips0, 1 + 4*side arm (u0, f0, n0, hang offset), 9 + 3*side leg (u0, f0, n0)
# kf[]: 2*side arm (l1, l2), 4 + 2*side leg (l1, l2)
# per frame: fv 0 hips target, 1 neck target, 2 + side wrist target, 4 + side foot target (all body frame)
#            fm 0 head rotation (body frame), 1 + side hand rotation delta
#            ff 0 spine twist, 1 + 2*side (trigger, grip), 5 + side foot yaw, 7 max arm stretch
#            fi side: controller present


@wp.func
def _unit(v: wp.vec3) -> wp.vec3:
    n = wp.length(v)
    out = v
    if n > 1.0e-9:
        out = v / n
    return out


@wp.func
def _cols(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.mat33:
    return wp.mat33(a[0], b[0], c[0], a[1], b[1], c[1], a[2], b[2], c[2])


@wp.func
def _frame(a: wp.vec3, n: wp.vec3) -> wp.mat33:
    return _cols(a, n, wp.cross(a, n))


@wp.func
def _rotvec(v: wp.vec3) -> wp.mat33:
    a = wp.length(v)
    out = wp.identity(n=3, dtype=float)
    if a >= 1.0e-9:
        k = v / a
        K = wp.mat33(0.0, -k[2], k[1], k[2], 0.0, -k[0], -k[1], k[0], 0.0)
        out = out + K * wp.sin(a) + (K * K) * (1.0 - wp.cos(a))
    return out


@wp.func
def _log(R: wp.mat33) -> wp.vec3:
    q = wp.normalize(wp.quat_from_matrix(R))
    w = q[3]
    v = wp.vec3(q[0], q[1], q[2])
    if w < 0.0:
        w = -w
        v = -v
    s = wp.length(v)
    out = wp.vec3(0.0, 0.0, 0.0)
    if s >= 1.0e-9:
        out = v / s * (2.0 * wp.atan2(s, w))
    return out


@wp.func
def _slerp(R: wp.mat33, t: float) -> wp.mat33:
    return _rotvec(_log(R) * t)


@wp.func
def _rot_z(a: float) -> wp.mat33:
    c = wp.cos(a)
    s = wp.sin(a)
    return wp.mat33(c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)


@wp.func
def _swing(a_: wp.vec3, b_: wp.vec3) -> wp.mat33:
    a = _unit(a_)
    b = _unit(b_)
    c = wp.cross(a, b)
    s = wp.length(c)
    d = wp.dot(a, b)
    out = wp.identity(n=3, dtype=float)
    if s < 1.0e-9:
        if d <= 0.0:
            ref = wp.vec3(0.0, 1.0, 0.0)
            if wp.abs(a[0]) < 0.9:
                ref = wp.vec3(1.0, 0.0, 0.0)
            out = _rotvec(_unit(wp.cross(a, ref)) * wp.pi)
    else:
        out = _rotvec(c / s * wp.atan2(s, d))
    return out


@wp.func
def _glob(j: int, parents: wp.array(dtype=wp.int32), local: wp.array(dtype=wp.mat44)) -> wp.mat44:
    G = local[j]
    k = parents[j]
    while k >= 0:
        G = local[k] * G
        k = parents[k]
    return G


@wp.func
def _pos(j: int, C: wp.mat33, parents: wp.array(dtype=wp.int32), local: wp.array(dtype=wp.mat44)) -> wp.vec3:
    G = _glob(j, parents, local)
    return C * wp.vec3(G[0, 3], G[1, 3], G[2, 3])


@wp.func
def _set_rot(j: int, R_body: wp.mat33, C: wp.mat33, parents: wp.array(dtype=wp.int32),
             local: wp.array(dtype=wp.mat44)):
    p = parents[j]
    Rp = wp.identity(n=3, dtype=float)
    if p >= 0:
        Rp = _rot(_glob(p, parents, local))
    Rm = wp.transpose(C) * R_body * C
    L = local[j]
    local[j] = _compose(wp.inverse(Rp) * Rm, wp.vec3(L[0, 3], L[1, 3], L[2, 3]))


@wp.func
def _set_delta(j: int, D: wp.mat33, C: wp.mat33, parents: wp.array(dtype=wp.int32),
               local: wp.array(dtype=wp.mat44), restrot: wp.array(dtype=wp.mat33)):
    _set_rot(j, D * restrot[j], C, parents, local)


@wp.func
def _set_pos(j: int, p_body: wp.vec3, C: wp.mat33, parents: wp.array(dtype=wp.int32),
             local: wp.array(dtype=wp.mat44)):
    p = parents[j]
    Gp = wp.identity(n=4, dtype=float)
    if p >= 0:
        Gp = _glob(p, parents, local)
    pm = wp.transpose(C) * p_body
    t = wp.inverse(_rot(Gp)) * (pm - wp.vec3(Gp[0, 3], Gp[1, 3], Gp[2, 3]))
    local[j] = _compose(_rot(local[j]), t)


@wp.func
def _set_len(j: int, factor: float, rest_local: wp.array(dtype=wp.mat44), local: wp.array(dtype=wp.mat44)):
    R = rest_local[j]
    local[j] = _compose(_rot(local[j]), wp.vec3(R[0, 3], R[1, 3], R[2, 3]) * factor)


@wp.kernel
def ik_kernel(
    n: int,
    parents: wp.array(dtype=wp.int32),
    rest_local: wp.array(dtype=wp.mat44),
    local: wp.array(dtype=wp.mat44),  # out: model-space locals
    restrot: wp.array(dtype=wp.mat33),  # rest global rotation of each joint, body frame
    C: wp.mat33,  # model -> body
    bones: wp.array(dtype=wp.int32),
    kv: wp.array(dtype=wp.vec3),
    kf: wp.array(dtype=float),
    fj: wp.array(dtype=wp.int32),  # finger segments, parent before child, both hands
    faxis: wp.array(dtype=wp.vec3),
    fbase: wp.array(dtype=float),
    fgain: wp.array(dtype=float),
    fsrc: wp.array(dtype=wp.int32),  # 0 trigger, 1 grip
    ffirst: wp.array(dtype=wp.int32),  # first segment of a finger (the curl sum restarts)
    frest: wp.array(dtype=wp.mat33),
    frange: wp.array(dtype=wp.int32),  # [start, end) per side
    fv: wp.array(dtype=wp.vec3),
    fm: wp.array(dtype=wp.mat33),
    ff: wp.array(dtype=float),
    fi: wp.array(dtype=wp.int32),
):
    for j in range(n):
        local[j] = rest_local[j]
    CT = wp.transpose(C)
    hips = bones[0]
    neck = bones[1]
    head = bones[2]
    nsp = bones[6]
    ex = wp.vec3(0.0, 0.0, -1.0)  # palm normal at rest (body frame)

    # hips under the neck, spine towards the neck target, neck / head orientation
    _set_pos(hips, fv[0], C, parents, local)
    N_b = fv[1]
    hips_b = _pos(hips, C, parents, local)
    D = _swing(kv[0], N_b - hips_b) * _rot_z(ff[0])
    for i in range(nsp):
        _set_delta(bones[3 + i], _slerp(D, float(i + 1) / float(nsp)), C, parents, local, restrot)
    cj = hips
    if nsp > 0:
        last = bones[3 + nsp - 1]
        lp = _pos(last, C, parents, local)
        corr = _swing(_pos(neck, C, parents, local) - lp, N_b - lp)
        R_last = C * _rot(_glob(last, parents, local)) * CT
        _set_rot(last, corr * R_last, C, parents, local)
        cj = last
    chest_D = C * _rot(_glob(cj, parents, local)) * CT * wp.transpose(restrot[cj])
    H_b = fm[0]
    _set_delta(neck, chest_D * _slerp(wp.transpose(chest_D) * H_b, 0.5), C, parents, local, restrot)
    _set_delta(head, H_b, C, parents, local, restrot)

    # arms: two-bone IK to the wrist target (may stretch), forearm takes half the wrist twist, hand, fingers
    for side in range(2):
        up = bones[7 + 3 * side]
        lo = bones[8 + 3 * side]
        ha = bones[9 + 3 * side]
        if up >= 0 and lo >= 0 and ha >= 0:
            sign = 1.0 - 2.0 * float(side)
            u0 = kv[1 + 4 * side]
            f0 = kv[2 + 4 * side]
            n0 = kv[3 + 4 * side]
            S = _pos(up, C, parents, local)
            l1 = kf[2 * side]
            l2 = kf[2 * side + 1]
            has = fi[side]
            W = fv[2 + side]
            if has == 0:
                W = S + kv[4 + 4 * side]
            hand_D = fm[1 + side]
            dvec = W - S
            dist = wp.length(dvec)
            stretch = float(1.0)
            if dist > (l1 + l2) * 0.999:
                stretch = wp.min(dist / ((l1 + l2) * 0.999), ff[7])
            L1 = l1 * stretch
            L2 = l2 * stretch
            dist = wp.clamp(dist, 1.0e-4, (L1 + L2) * 0.999)
            u = _unit(dvec)
            pole = _unit(wp.vec3(-0.35, sign * 0.5, -1.0))
            v = _unit(pole - u * wp.dot(pole, u))
            if wp.length(v) < 1.0e-6:
                v = _unit(wp.cross(u, wp.vec3(0.0, 0.0, 1.0)))
            cos_a = (L1 * L1 + dist * dist - L2 * L2) / (2.0 * L1 * dist)
            a_ang = wp.acos(wp.clamp(cos_a, -1.0, 1.0))
            E = S + (u * wp.cos(a_ang) + v * wp.sin(a_ang)) * L1
            Wc = S + u * dist
            nn = _unit(wp.cross(E - S, Wc - E))
            if wp.length(nn) < 1.0e-6:
                nn = n0
            up_dir = _unit(E - S)
            lo_dir = _unit(Wc - E)
            _set_delta(up, _frame(up_dir, nn) * wp.transpose(_frame(u0, n0)), C, parents, local, restrot)
            if stretch > 1.0:
                _set_len(lo, stretch, rest_local, local)
                _set_len(ha, stretch, rest_local, local)
            D_lo = _frame(lo_dir, nn) * wp.transpose(_frame(f0, n0))
            if has != 0:
                palm_now = D_lo * ex
                palm_tgt = hand_D * ex
                p1 = _unit(palm_now - lo_dir * wp.dot(palm_now, lo_dir))
                p2 = _unit(palm_tgt - lo_dir * wp.dot(palm_tgt, lo_dir))
                ang = wp.atan2(wp.dot(wp.cross(p1, p2), lo_dir), wp.dot(p1, p2))
                D_lo = _rotvec(lo_dir * (ang * 0.5)) * D_lo
            _set_delta(lo, D_lo, C, parents, local, restrot)
            if has == 0:
                hand_D = D_lo
            _set_delta(ha, hand_D, C, parents, local, restrot)
            trig = ff[1 + 2 * side]
            grip = ff[2 + 2 * side]
            total = float(0.0)
            for i in range(frange[2 * side], frange[2 * side + 1]):
                if ffirst[i] != 0:
                    total = 0.0
                val = grip
                if fsrc[i] == 0:
                    val = trig
                total = total + fbase[i] + fgain[i] * val
                _set_rot(fj[i], hand_D * _rotvec(faxis[i] * total) * frest[i], C, parents, local)

    # legs: two-bone IK to the planted feet, foot yaw
    for side in range(2):
        up = bones[13 + 3 * side]
        lo = bones[14 + 3 * side]
        ft = bones[15 + 3 * side]
        if up >= 0 and lo >= 0 and ft >= 0:
            sign = 1.0 - 2.0 * float(side)
            F = fv[4 + side]
            S = _pos(up, C, parents, local)
            l1 = kf[4 + 2 * side]
            l2 = kf[5 + 2 * side]
            u0 = kv[9 + 3 * side]
            f0 = kv[10 + 3 * side]
            n0 = kv[11 + 3 * side]
            dvec = F - S
            dist = wp.clamp(wp.length(dvec), 1.0e-4, (l1 + l2) * 0.999)
            u = _unit(dvec)
            pole = _unit(wp.vec3(1.0, sign * 0.15, 0.0))
            v = _unit(pole - u * wp.dot(pole, u))
            cos_a = (l1 * l1 + dist * dist - l2 * l2) / (2.0 * l1 * dist)
            ang = wp.acos(wp.clamp(cos_a, -1.0, 1.0))
            K = S + (u * wp.cos(ang) + v * wp.sin(ang)) * l1
            Fc = S + u * dist
            nn = _unit(wp.cross(K - S, Fc - K))
            if wp.length(nn) < 1.0e-6:
                nn = n0
            a1 = _unit(K - S)
            a2 = _unit(Fc - K)
            _set_delta(up, _frame(a1, nn) * wp.transpose(_frame(u0, n0)), C, parents, local, restrot)
            _set_delta(lo, _frame(a2, nn) * wp.transpose(_frame(f0, n0)), C, parents, local, restrot)
            _set_delta(ft, _rot_z(ff[5 + side]), C, parents, local, restrot)


def _np_unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


class IkSolver:
    """Static skeleton data for ``ik_kernel`` + per-frame target buffers (zero-copy numpy views)."""

    def __init__(self, av, local: wp.array) -> None:
        h, p0, n = av.h, av.p0, av.n

        def j(b: str) -> int:
            return int(h[b]) if b in h else -1

        f = np.float32
        bones = np.full(19, -1, dtype=np.int32)
        bones[0], bones[1], bones[2] = j("hips"), j("neck"), j("head")
        if min(bones[:3]) < 0:
            raise ValueError("avatar without hips / neck / head")
        for i, b in enumerate(av.spine_chain):
            bones[3 + i] = h[b]
        bones[6] = len(av.spine_chain)
        kv = np.zeros((15, 3))
        kf = np.zeros(8)
        kv[0] = p0["neck"] - p0["hips"]
        for si, (side, sign) in enumerate((("left", 1.0), ("right", -1.0))):
            up, lo, ha = f"{side}UpperArm", f"{side}LowerArm", f"{side}Hand"
            if all(b in h for b in (up, lo, ha)):
                bones[7 + 3 * si: 10 + 3 * si] = (h[up], h[lo], h[ha])
                u0 = _np_unit(p0[lo] - p0[up])
                f0 = _np_unit(p0[ha] - p0[lo])
                kv[1 + 4 * si], kv[2 + 4 * si] = u0, f0
                kv[3 + 4 * si] = _np_unit(np.cross(u0, [1.0, 0.0, 0.0]))
                l1, l2 = av.arm_len[side]
                kf[2 * si], kf[2 * si + 1] = l1, l2
                kv[4 + 4 * si] = (0.02, sign * 0.04, -(l1 + l2) * 0.98)
            up, lo, ft = f"{side}UpperLeg", f"{side}LowerLeg", f"{side}Foot"
            if all(b in h for b in (up, lo, ft)):
                bones[13 + 3 * si: 16 + 3 * si] = (h[up], h[lo], h[ft])
                u0 = _np_unit(p0[lo] - p0[up])
                kv[9 + 3 * si] = u0
                kv[10 + 3 * si] = _np_unit(p0[ft] - p0[lo])
                kv[11 + 3 * si] = _np_unit(np.cross(u0, [-1.0, 0.0, 0.0]))
                l1, l2 = av.leg_len[side]
                kf[4 + 2 * si], kf[5 + 2 * si] = l1, l2
        # fingers (same tables as the batched numpy path)
        if getattr(av, "_fingers", None) is None:
            av._init_fingers()
        fj, fax, fb, fg, fs, ffst, frs, frange = [], [], [], [], [], [], [], []
        for side in ("left", "right"):
            fd = av._fingers.get(side)
            frange.append(len(fj))
            if fd is not None:
                prev = -1
                for i in range(len(fd["joints"])):
                    fj.append(int(fd["joints"][i]))
                    fax.append(fd["axes"][i])
                    fb.append(fd["base"][i])
                    fg.append(fd["gain"][i])
                    fs.append(int(fd["src"][i]))
                    ffst.append(int(fd["finger"][i] != prev))
                    prev = fd["finger"][i]
                    frs.append(fd["rest"][i])
            frange.append(len(fj))
        if not fj:  # keep the arrays non-empty
            fj, fax, fb, fg, fs, ffst, frs = [0], [np.zeros(3)], [0.0], [0.0], [0], [1], [np.eye(3)]
        C = av.C
        restrot = np.array([C @ av.G0[k][:3, :3] @ C.T for k in range(n)])

        def arr(a, dtype):
            return wp.array(np.asarray(a), dtype=dtype, device=DEVICE)

        self.n = n
        self.local = local
        self.parents = arr(av.parents.astype(np.int32), wp.int32)
        self.rest_local = arr(av.rest_local.astype(f), wp.mat44)
        self.restrot = arr(restrot.astype(f), wp.mat33)
        self.C = wp.mat33(*C.astype(f).reshape(-1))
        self.bones = arr(bones, wp.int32)
        self.kv = arr(kv.astype(f), wp.vec3)
        self.kf = arr(kf.astype(f), float)
        self.fj = arr(np.array(fj, dtype=np.int32), wp.int32)
        self.faxis = arr(np.array(fax, dtype=f), wp.vec3)
        self.fbase = arr(np.array(fb, dtype=f), float)
        self.fgain = arr(np.array(fg, dtype=f), float)
        self.fsrc = arr(np.array(fs, dtype=np.int32), wp.int32)
        self.ffirst = arr(np.array(ffst, dtype=np.int32), wp.int32)
        self.frest = arr(np.array(frs, dtype=f), wp.mat33)
        self.frange = arr(np.array(frange, dtype=np.int32), wp.int32)
        self.fv = wp.zeros(6, dtype=wp.vec3, device=DEVICE)
        self.fm = wp.zeros(3, dtype=wp.mat33, device=DEVICE)
        self.ff = wp.zeros(8, dtype=float, device=DEVICE)
        self.fi = wp.zeros(2, dtype=wp.int32, device=DEVICE)
        self.fv_np = self.fv.numpy().reshape(6, 3)
        self.fm_np = self.fm.numpy().reshape(3, 3, 3)
        self.ff_np = self.ff.numpy()
        self.fi_np = self.fi.numpy()
        self.has_leg = [bones[13 + 3 * si] >= 0 for si in range(2)]
        self.has_arm = [bones[7 + 3 * si] >= 0 for si in range(2)]
        # load the compiled module now: the first launch would otherwise stall the sim loop for a moment
        wp.load_module(module=sys.modules[__name__], device=DEVICE)

    def run(self) -> None:
        wp.launch(
            ik_kernel,
            dim=1,
            inputs=[
                self.n, self.parents, self.rest_local, self.local, self.restrot, self.C, self.bones, self.kv, self.kf,
                self.fj, self.faxis, self.fbase, self.fgain, self.fsrc, self.ffirst, self.frest, self.frange,
                self.fv, self.fm, self.ff, self.fi,
            ],
            device=DEVICE,
        )
