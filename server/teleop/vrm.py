"""VRM (0.x / 1.0) -> USD converter (UsdSkel skinned meshes + UsdPreviewSurface materials).

VRM is glTF 2.0 (.glb) with humanoid metadata. The converter writes

    <out>/<name>.usdc          /Avatar (Xform, default prim)
                                 Root (SkelRoot, Y-up glTF -> Z-up, model faces +X)
                                   Skeleton      every node that is a joint of a skin (or an ancestor of one)
                                   Anim          SkelAnimation over the humanoid bones (driven at runtime)
                                   Meshes/...    one Mesh per glTF primitive, bound to the skeleton
                                 Looks/...
    <out>/textures/*.png
    <out>/<name>.avatar.json   skeleton + humanoid map used by the runtime IK (teleop/avatar.py)

Not converted: blend shapes (expressions), spring bones, MToon shading (approximated: lit base color plus a
little of it as emission so the toon look survives without strong lights), outlines.
Needs pxr (run inside Isaac Sim).
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdSkel, Vt

COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
NORM_DIV = {np.int8: 127.0, np.uint8: 255.0, np.int16: 32767.0, np.uint16: 65535.0}
CONVERTER_VERSION = 7
# MToon (toon) approximation with UsdPreviewSurface, calibrated in the teleop scene's lighting (RTX renders a
# lit 0.5 grey almost white, and an emission of ~0.6 shows on screen at about its own value):
#   diffuse  = main texture * TOON_LIT         (shading from the scene lights: lit side brighter)
#   emission = shade texture * TOON_SHADE      (the MToon shadow color, so shadows keep the skin warm)
#   glow     = HDR _EmissionColor * TOON_GLOW  (halo and other emissive parts)
TOON_LIT = 0.2
TOON_SHADE = 0.45
TOON_GLOW = 0.8


class Glb:
    def __init__(self, path: Path) -> None:
        data = Path(path).read_bytes()
        magic, _ver, length = struct.unpack_from("<III", data, 0)
        if magic != 0x46546C67:
            raise ValueError(f"{path}: not a binary glTF / VRM file")
        off, self.j, self.bin = 12, None, b""
        while off < length:
            clen, ctype = struct.unpack_from("<II", data, off)
            off += 8
            chunk = data[off : off + clen]
            off += clen
            if ctype == 0x4E4F534A:
                self.j = json.loads(chunk)
            elif ctype == 0x004E4942:
                self.bin = chunk
        if self.j is None:
            raise ValueError(f"{path}: no JSON chunk")

    def view_bytes(self, i: int) -> tuple[bytes, int | None]:
        bv = self.j["bufferViews"][i]
        o = bv.get("byteOffset", 0)
        return self.bin[o : o + bv["byteLength"]], bv.get("byteStride")

    def accessor(self, i: int) -> np.ndarray:
        a = self.j["accessors"][i]
        dt = np.dtype(COMP[a["componentType"]])
        n, cnt = NCOMP[a["type"]], a["count"]
        if "bufferView" in a:
            buf, stride = self.view_bytes(a["bufferView"])
            off = a.get("byteOffset", 0)
            if stride and stride != dt.itemsize * n:
                out = np.ndarray((cnt, n), dt, buffer=buf, offset=off, strides=(stride, dt.itemsize)).copy()
            else:
                out = np.frombuffer(buf, dt, cnt * n, off).reshape(cnt, n).copy()
        else:
            out = np.zeros((cnt, n), dt)
        sp = a.get("sparse")
        if sp:
            idx_dt = np.dtype(COMP[sp["indices"]["componentType"]])
            ib, _ = self.view_bytes(sp["indices"]["bufferView"])
            idx = np.frombuffer(ib, idx_dt, sp["count"], sp["indices"].get("byteOffset", 0))
            vb, _ = self.view_bytes(sp["values"]["bufferView"])
            vals = np.frombuffer(vb, dt, sp["count"] * n, sp["values"].get("byteOffset", 0)).reshape(-1, n)
            out = out.copy()
            out[idx] = vals
        if a.get("normalized") and dt.type in NORM_DIV:
            out = out.astype(np.float32) / NORM_DIV[dt.type]
        return out


def _local_matrix(node: dict) -> np.ndarray:
    """Column-vector 4x4 of a glTF node."""
    if "matrix" in node:
        return np.array(node["matrix"], dtype=float).reshape(4, 4).T
    t = node.get("translation", [0, 0, 0])
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    s = node.get("scale", [1, 1, 1])
    R = np.array(Gf.Matrix3d(Gf.Rotation(Gf.Quatd(w, x, y, z)))).T  # Gf is row-vector -> transpose
    M = np.eye(4)
    M[:3, :3] = R @ np.diag(s)
    M[:3, 3] = t
    return M


def _gf_matrix(M: np.ndarray) -> Gf.Matrix4d:
    return Gf.Matrix4d(*[float(v) for v in M.T.reshape(-1)])


def _clean(name: str, used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_]", "_", name or "node") or "node"
    if base[0].isdigit():
        base = "_" + base
    out, k = base, 1
    while out in used:
        k += 1
        out = f"{base}_{k}"
    used.add(out)
    return out


def humanoid_map(g: Glb) -> tuple[int, dict[str, int]]:
    ext = g.j.get("extensions", {})
    if "VRMC_vrm" in ext:
        hb = ext["VRMC_vrm"]["humanoid"]["humanBones"]
        return 1, {k: v["node"] for k, v in hb.items()}
    if "VRM" in ext:
        return 0, {b["bone"]: b["node"] for b in ext["VRM"]["humanoid"]["humanBones"]}
    raise ValueError("not a VRM file (no VRM / VRMC_vrm extension)")


def _spring_raw(g: Glb) -> tuple[list[dict], list[dict], bool]:
    """Spring bone groups and sphere colliders (model space) of VRM 0.x / VRM 1.0."""
    ext = g.j.get("extensions", {})
    groups, cols = [], []
    if "VRMC_springBone" in ext:
        sb = ext["VRMC_springBone"]
        cmap: list[list[int]] = []
        for c in sb.get("colliders", []):
            shape = c.get("shape", {})
            ids = []
            if "sphere" in shape:
                sp = shape["sphere"]
                ids.append(len(cols))
                cols.append({"node": c["node"], "offset": list(sp.get("offset", [0, 0, 0])), "radius": sp.get("radius", 0)})
            elif "capsule" in shape:
                cp = shape["capsule"]
                o, t = np.array(cp.get("offset", [0, 0, 0])), np.array(cp.get("tail", [0, 0, 0]))
                for k in range(4):  # capsule -> spheres along its axis
                    ids.append(len(cols))
                    cols.append({"node": c["node"], "offset": list(o + (t - o) * k / 3), "radius": cp.get("radius", 0)})
            cmap.append(ids)
        gmap = [[i for c in cg.get("colliders", []) for i in cmap[c]] for cg in sb.get("colliderGroups", [])]
        for spring in sb.get("springs", []):
            per = {}
            for jt in spring.get("joints", []):
                per[jt["node"]] = {
                    "stiffness": jt.get("stiffness", 1.0),
                    "gravityPower": jt.get("gravityPower", 0.0),
                    "gravityDir": jt.get("gravityDir", [0, -1, 0]),
                    "drag": jt.get("dragForce", 0.5),
                    "hitRadius": jt.get("hitRadius", 0.0),
                }
            groups.append({
                "nodes": [jt["node"] for jt in spring.get("joints", [])],
                "per_joint": per,
                "colliders": [i for k in spring.get("colliderGroups", []) for i in gmap[k]],
            })
        return groups, cols, True
    sa = ext.get("VRM", {}).get("secondaryAnimation")
    if not sa:
        return [], [], False
    gmap = []
    for cg in sa.get("colliderGroups", []):
        ids = []
        for c in cg.get("colliders", []):
            o = c.get("offset", {})
            ids.append(len(cols))
            # VRM 0.x stores Unity-space offsets: z is mirrored relative to glTF
            cols.append({"node": cg["node"], "offset": [o.get("x", 0), o.get("y", 0), -o.get("z", 0)], "radius": c.get("radius", 0)})
        gmap.append(ids)
    for bg in sa.get("boneGroups", []):
        gd = bg.get("gravityDir", {"x": 0, "y": -1, "z": 0})
        groups.append({
            "nodes": list(bg.get("bones", [])),
            "stiffness": bg.get("stiffiness", 1.0),  # (sic) VRM 0.x spelling
            "gravityPower": bg.get("gravityPower", 0.0),
            "gravityDir": [gd.get("x", 0), gd.get("y", -1), -gd.get("z", 0)],
            "drag": bg.get("dragForce", 0.4),
            "hitRadius": bg.get("hitRadius", 0.02),
            "colliders": [i for k in bg.get("colliderGroups", []) for i in gmap[k]],
        })
    return groups, cols, False


def convert(vrm_path: Path, out_dir: Path) -> Path:
    """Convert and return the path of the .usdc file. Skips the work if the output is up to date."""
    vrm_path, out_dir = Path(vrm_path), Path(out_dir)
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", vrm_path.stem)
    usd_path = out_dir / f"{stem}.usdc"
    meta_path = out_dir / f"{stem}.avatar.json"
    if usd_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("converter") == CONVERTER_VERSION and meta.get("source_mtime") == vrm_path.stat().st_mtime:
                return usd_path
        except (OSError, ValueError):
            pass
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "textures").mkdir(exist_ok=True)

    g = Glb(vrm_path)
    j = g.j
    nodes = j["nodes"]
    version, human = humanoid_map(g)
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    roots = j["scenes"][j.get("scene", 0)]["nodes"]

    # global rest matrices
    G: dict[int, np.ndarray] = {}

    def walk(i: int, P: np.ndarray) -> None:
        G[i] = P @ _local_matrix(nodes[i])
        for c in nodes[i].get("children", []):
            walk(c, G[i])

    for r in roots:
        walk(r, np.eye(4))

    spring_roots, spring_colliders, spring_v1 = _spring_raw(g)
    # skeleton = skin joints + humanoid bones + spring bones + parents of rigid meshes, closed under ancestors
    wanted: set[int] = set(human.values())
    for grp in spring_roots:
        wanted.update(grp["nodes"])
    wanted.update(c["node"] for c in spring_colliders)
    for s in j.get("skins", []):
        wanted.update(s["joints"])
    for i, n in enumerate(nodes):
        if "mesh" in n and "skin" not in n and i in parent:
            wanted.add(parent[i])
    closed = set()
    for i in wanted:
        while i is not None and i not in closed:
            closed.add(i)
            i = parent.get(i)
    order: list[int] = []
    names: dict[int, str] = {}

    def dfs(i: int, prefix: str, used: set[str]) -> None:
        if i not in closed:
            return
        nm = _clean(nodes[i].get("name", f"node{i}"), used)
        names[i] = f"{prefix}/{nm}" if prefix else nm
        order.append(i)
        child_used: set[str] = set()
        for c in nodes[i].get("children", []):
            dfs(c, names[i], child_used)

    root_used: set[str] = set()
    for r in roots:
        dfs(r, "", root_used)
    jidx = {n: k for k, n in enumerate(order)}

    # model orientation: up = +Y, forward from the foot -> toes direction (VRM 0: -Z, VRM 1: +Z)
    up = np.array([0.0, 1.0, 0.0])
    fwd = np.array([0.0, 0.0, -1.0 if version == 0 else 1.0])
    for side in ("left", "right"):
        if f"{side}Foot" in human and f"{side}Toes" in human:
            d = G[human[f"{side}Toes"]][:3, 3] - G[human[f"{side}Foot"]][:3, 3]
            d[1] = 0
            if np.linalg.norm(d) > 1e-4:
                fwd = np.array([0.0, 0.0, np.sign(d[2]) or -1.0])
            break
    left = np.cross(up, fwd)
    C = np.stack([fwd, left, up])  # model -> body frame (x forward, y left, z up)

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    avatar = UsdGeom.Xform.Define(stage, "/Avatar")
    stage.SetDefaultPrim(avatar.GetPrim())
    root = UsdSkel.Root.Define(stage, "/Avatar/Root")
    C4 = np.eye(4)
    C4[:3, :3] = C
    UsdGeom.Xformable(root).AddTransformOp().Set(_gf_matrix(C4))

    skel = UsdSkel.Skeleton.Define(stage, "/Avatar/Root/Skeleton")
    joint_tokens = [names[i] for i in order]
    skel.CreateJointsAttr(Vt.TokenArray(joint_tokens))
    skel.CreateJointNamesAttr(Vt.TokenArray([t.split("/")[-1] for t in joint_tokens]))
    skel.CreateBindTransformsAttr(Vt.Matrix4dArray([_gf_matrix(G[i]) for i in order]))
    skel.CreateRestTransformsAttr(Vt.Matrix4dArray([_gf_matrix(_local_matrix(nodes[i])) for i in order]))

    # springs: every simulated joint with its tail joint (VRM 0: first child, VRM 1: next joint of the chain)
    children = {i: [c for c in nodes[i].get("children", []) if c in jidx] for i in order}
    spring_entries: dict[int, dict] = {}
    for grp in spring_roots:
        if spring_v1:
            chain = [n for n in grp["nodes"] if n in jidx]
            pairs = [(a, b) for a, b in zip(chain, chain[1:])]
            params = grp["per_joint"]
        else:
            pairs = []
            stack = [n for n in grp["nodes"] if n in jidx]
            while stack:
                n = stack.pop()
                kids = children.get(n, [])
                pairs.append((n, kids[0] if kids else -1))
                stack.extend(kids)
            params = {}
        for a, b in pairs:
            if a in spring_entries or a in set(human.values()):
                continue
            pr = params.get(a, grp)
            spring_entries[a] = {
                "joint": jidx[a],
                "child": jidx[b] if b >= 0 else -1,
                "stiffness": float(pr["stiffness"]),
                "gravityPower": float(pr["gravityPower"]),
                "gravityDir": [float(v) for v in pr["gravityDir"]],
                "drag": float(pr["drag"]),
                "hitRadius": float(pr["hitRadius"]),
                "colliders": list(grp["colliders"]),
            }
    springs = sorted(spring_entries.values(), key=lambda e: e["joint"])
    colliders = [
        {"joint": jidx[c["node"]], "offset": c["offset"], "radius": c["radius"]} for c in spring_colliders if c["node"] in jidx
    ]

    # animation over the humanoid bones and the spring bones
    anim_set = set(human.values()) | {order[e["joint"]] for e in springs}
    anim_nodes = [i for i in order if i in anim_set]
    anim = UsdSkel.Animation.Define(stage, "/Avatar/Root/Skeleton/Anim")
    anim.CreateJointsAttr(Vt.TokenArray([names[i] for i in anim_nodes]))
    rest_t, rest_r, rest_s = [], [], []
    for i in anim_nodes:
        n = nodes[i]
        t = n.get("translation", [0, 0, 0])
        x, y, z, w = n.get("rotation", [0, 0, 0, 1])
        rest_t.append(Gf.Vec3f(*t))
        rest_r.append(Gf.Quatf(w, x, y, z))
        rest_s.append(Gf.Vec3h(*n.get("scale", [1, 1, 1])))
    anim.CreateTranslationsAttr(Vt.Vec3fArray(rest_t))
    anim.CreateRotationsAttr(Vt.QuatfArray(rest_r))
    anim.CreateScalesAttr(Vt.Vec3hArray(rest_s))
    UsdSkel.BindingAPI.Apply(skel.GetPrim()).CreateAnimationSourceRel().SetTargets([anim.GetPath()])

    # textures / materials
    img_files: dict[int, str] = {}
    used_img: set[str] = set()
    for k, im in enumerate(j.get("images", [])):
        ext = ".jpg" if im.get("mimeType") == "image/jpeg" else ".png"
        fname = _clean(im.get("name", f"image{k}"), used_img) + ext
        if "bufferView" in im:
            data, _ = g.view_bytes(im["bufferView"])
            (out_dir / "textures" / fname).write_bytes(data)
            img_files[k] = f"./textures/{fname}"
    tex_image = {k: t.get("source") for k, t in enumerate(j.get("textures", []))}
    looks = UsdGeom.Scope.Define(stage, "/Avatar/Looks")
    mtoon = _mtoon_params(g)
    mat_paths: list[Sdf.Path] = []
    mat_double: list[bool] = []
    used_mat: set[str] = set()
    for k, m in enumerate(j.get("materials", [])):
        mp = looks.GetPath().AppendChild(_clean(m.get("name", f"mat{k}"), used_mat))
        mat_paths.append(mp)
        mat_double.append(bool(m.get("doubleSided", False)))
        _make_material(stage, mp, m, tex_image, img_files, mtoon.get(k))

    # meshes
    meshes_scope = UsdGeom.Scope.Define(stage, "/Avatar/Root/Meshes")
    used_mesh: set[str] = set()
    n_verts = 0
    samples_p: list[np.ndarray] = []  # rest-pose vertices (model space) and their dominant joint, for colliders
    samples_j: list[np.ndarray] = []
    for ni in [i for i in range(len(nodes)) if i in G and "mesh" in nodes[i]]:
        node = nodes[ni]
        mesh = j["meshes"][node["mesh"]]
        skin = j["skins"][node["skin"]] if "skin" in node else None
        if skin is not None:
            sj = skin["joints"]
            if "inverseBindMatrices" in skin:
                ibm = g.accessor(skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
            else:
                ibm = np.repeat(np.eye(4)[None], len(sj), 0)
            gb = [G[jn] @ ibm[k] for k, jn in enumerate(sj)]
            geom_bind = gb[0]
            if max(np.abs(x - geom_bind).max() for x in gb) > 1e-3:
                print(f"[vrm] warning: {node.get('name')}: inconsistent inverse bind matrices", flush=True)
        for pk, prim in enumerate(mesh["primitives"]):
            if prim.get("mode", 4) != 4:
                continue
            at = prim["attributes"]
            P = g.accessor(at["POSITION"]).astype(np.float32)
            idx = g.accessor(prim["indices"]).reshape(-1) if "indices" in prim else np.arange(len(P))
            used_v, remap = np.unique(idx, return_inverse=True)
            P = P[used_v]
            name = _clean(f"{node.get('name', 'mesh')}_{pk}" if len(mesh["primitives"]) > 1 else node.get("name", "mesh"), used_mesh)
            um = UsdGeom.Mesh.Define(stage, meshes_scope.GetPath().AppendChild(name))
            um.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(P))
            um.CreateFaceVertexCountsAttr(Vt.IntArray([3] * (len(remap) // 3)))
            um.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(remap.astype(np.int32)))
            um.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            lo, hi = P.min(0), P.max(0)
            um.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))
            if "NORMAL" in at:
                Nn = g.accessor(at["NORMAL"]).astype(np.float32)[used_v]
                um.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(Nn))
                um.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            if "TEXCOORD_0" in at:
                uv = g.accessor(at["TEXCOORD_0"]).astype(np.float32)[used_v]
                uv[:, 1] = 1.0 - uv[:, 1]
                pv = UsdGeom.PrimvarsAPI(um).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
                pv.Set(Vt.Vec2fArray.FromNumpy(uv))
            mat = prim.get("material")
            if mat is not None and mat < len(mat_paths):
                um.CreateDoubleSidedAttr(mat_double[mat])
                UsdShade.MaterialBindingAPI.Apply(um.GetPrim()).Bind(UsdShade.Material.Get(stage, mat_paths[mat]))
            binding = UsdSkel.BindingAPI.Apply(um.GetPrim())
            binding.CreateSkeletonRel().SetTargets([skel.GetPath()])
            if skin is not None and "JOINTS_0" in at:
                J = g.accessor(at["JOINTS_0"]).astype(np.int32)[used_v]
                W = g.accessor(at["WEIGHTS_0"]).astype(np.float32)[used_v]
                if "JOINTS_1" in at and "WEIGHTS_1" in at:
                    J = np.hstack([J, g.accessor(at["JOINTS_1"]).astype(np.int32)[used_v]])
                    W = np.hstack([W, g.accessor(at["WEIGHTS_1"]).astype(np.float32)[used_v]])
                ws = W.sum(1, keepdims=True)
                W = np.where(ws > 1e-6, W / np.maximum(ws, 1e-6), 0)
                W[ws[:, 0] <= 1e-6, 0] = 1.0
                binding.CreateJointsAttr(Vt.TokenArray([names[x] for x in sj]))
                binding.CreateJointIndicesPrimvar(False, J.shape[1]).Set(Vt.IntArray.FromNumpy(J.reshape(-1)))
                binding.CreateJointWeightsPrimvar(False, W.shape[1]).Set(Vt.FloatArray.FromNumpy(W.reshape(-1)))
                binding.CreateGeomBindTransformAttr(_gf_matrix(geom_bind))
                sj_idx = np.array([jidx[x] for x in sj])
                samples_p.append(P[::2] @ geom_bind[:3, :3].T + geom_bind[:3, 3])
                samples_j.append(sj_idx[J[np.arange(len(J)), np.argmax(W, 1)]][::2])
            else:
                # rigid mesh: follows its nearest joint ancestor
                a = ni if ni in jidx else parent.get(ni)
                while a is not None and a not in jidx:
                    a = parent.get(a)
                if a is None:
                    UsdGeom.Xformable(um).AddTransformOp().Set(_gf_matrix(G[ni]))
                    continue
                binding.CreateJointsAttr(Vt.TokenArray([names[a]]))
                binding.CreateJointIndicesPrimvar(True, 1).Set(Vt.IntArray([0]))
                binding.CreateJointWeightsPrimvar(True, 1).Set(Vt.FloatArray([1.0]))
                binding.CreateGeomBindTransformAttr(_gf_matrix(G[ni]))
                samples_p.append(P[::2] @ G[ni][:3, :3].T + G[ni][:3, 3])
                samples_j.append(np.full(len(P[::2]), jidx[a]))
            n_verts += len(P)
    stage.GetRootLayer().Save()

    # runtime metadata (body frame = x forward, y left, z up; model units)
    def bpos(i: int) -> list[float]:
        return [float(v) for v in C @ G[i][:3, 3]]

    eyes = [human[k] for k in ("leftEye", "rightEye") if k in human]
    eye = np.mean([C @ G[i][:3, 3] for i in eyes], 0) if eyes else C @ G[human["head"]][:3, 3] + [0.06, 0, 0.06]
    meta = {
        "converter": CONVERTER_VERSION,
        "source": vrm_path.name,
        "source_mtime": vrm_path.stat().st_mtime,
        "vrm_version": version,
        "usd": usd_path.name,
        "model_to_body": C.tolist(),
        "joints": joint_tokens,
        "parents": [jidx.get(parent.get(i), -1) if parent.get(i) in jidx else -1 for i in order],
        "anim_joints": [jidx[i] for i in anim_nodes],
        "rest_local": [_local_matrix(nodes[i]).tolist() for i in order],
        "humanoid": {k: jidx[v] for k, v in human.items() if v in jidx},
        "rest_pos_body": {k: bpos(v) for k, v in human.items() if v in G},
        "eye_body": [float(v) for v in eye],
        "springs": springs,
        "colliders": colliders,
        "bodies": _body_shapes(
            C, G, order, jidx, parent, human, {order[e["joint"]] for e in springs},
            np.concatenate(samples_p) if samples_p else np.zeros((0, 3)),
            np.concatenate(samples_j) if samples_j else np.zeros(0, dtype=int),
        ),
        "vertices": int(n_verts),
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    print(f"[vrm] {vrm_path.name}: {len(order)} joints, {n_verts} vertices -> {usd_path}", flush=True)
    return usd_path


# collision bodies of the avatar: name -> (humanoid bones whose vertices it covers, segment start, segment end)
# segment end None = sphere around the vertices
BODY_SEGMENTS = {
    "hips": (("hips", "spine"), "hips", "chest"),
    "chest": (("chest", "upperChest", "leftShoulder", "rightShoulder"), "chest", "neck"),
    "head": (("neck", "head", "leftEye", "rightEye", "jaw"), None, None),
    "leftUpperArm": (("leftUpperArm",), "leftUpperArm", "leftLowerArm"),
    "rightUpperArm": (("rightUpperArm",), "rightUpperArm", "rightLowerArm"),
    "leftLowerArm": (("leftLowerArm",), "leftLowerArm", "leftHand"),
    "rightLowerArm": (("rightLowerArm",), "rightLowerArm", "rightHand"),
    "leftHand": (("leftHand",), None, None),
    "rightHand": (("rightHand",), None, None),
    "leftUpperLeg": (("leftUpperLeg",), "leftUpperLeg", "leftLowerLeg"),
    "rightUpperLeg": (("rightUpperLeg",), "rightUpperLeg", "rightLowerLeg"),
    "leftLowerLeg": (("leftLowerLeg",), "leftLowerLeg", "leftFoot"),
    "rightLowerLeg": (("rightLowerLeg",), "rightLowerLeg", "rightFoot"),
    "leftFoot": (("leftFoot", "leftToes"), None, None),
    "rightFoot": (("rightFoot", "rightToes"), None, None),
}


def _body_shapes(C, G, order, jidx, parent, human, spring_nodes, P, J) -> list[dict]:
    """Capsules / spheres fitted to the rest-pose skin. Vertices of spring bones (hair, skirt...) are ignored.

    Each body is attached to a bone: center/axis are stored in that bone's rest frame (model space), so the
    runtime only needs the bone's current global transform."""
    bone_of = {v: k for k, v in human.items()}
    # every skeleton joint -> humanoid bone it belongs to (nearest humanoid ancestor; None below a spring)
    owner = np.full(len(order), -1, dtype=int)
    names_h = list(human)
    for k, node in enumerate(order):
        n, ok = node, True
        while n is not None and n not in bone_of:
            if n in spring_nodes:
                ok = False
                break
            n = parent.get(n)
        if ok and n is not None:
            owner[k] = names_h.index(bone_of[n])
    vb = owner[J] if len(J) else np.zeros(0, dtype=int)
    out = []
    for name, (bones, a, b) in BODY_SEGMENTS.items():
        ids = [names_h.index(x) for x in bones if x in human]
        attach = a or bones[0]
        if not ids or attach not in human:
            continue
        pts = P[np.isin(vb, ids)]
        Ga = G[human[attach]]
        if a and b and a in human and b in human:
            p0, p1 = G[human[a]][:3, 3], G[human[b]][:3, 3]
            axis = p1 - p0
            length = float(np.linalg.norm(axis))
            if length < 1e-4:
                continue
            u = axis / length
            if len(pts) >= 20:
                t = (pts - p0) @ u
                inside = (t > 0) & (t < length)
                perp = np.linalg.norm((pts - p0) - np.outer(t, u), axis=1)[inside]
                radius = float(np.percentile(perp, 70)) if len(perp) >= 10 else length * 0.25
            else:
                radius = length * 0.25
            radius = float(np.clip(radius, length * 0.12, length * 0.8))
            center = (p0 + p1) / 2
            half = max(0.0, length / 2 - radius * 0.5)
            kind = "capsule"
        else:
            if len(pts) < 20:
                continue
            center = pts.mean(0)
            radius = float(np.percentile(np.linalg.norm(pts - center, axis=1), 70))
            u = np.array([0.0, 1.0, 0.0])
            half = 0.0
            kind = "sphere"
        inv = np.linalg.inv(Ga)
        out.append({
            "name": name,
            "bone": attach,
            "kind": kind,
            "center": [float(v) for v in inv[:3, :3] @ center + inv[:3, 3]],  # in the bone's frame
            "axis": [float(v) for v in inv[:3, :3] @ u],
            "radius": radius,
            "half": half,
        })
    return out


def _mtoon_params(g: Glb) -> dict[int, dict]:
    """material index -> MToon parameters (VRM 0.x materialProperties / VRM 1.0 VRMC_materials_mtoon)."""
    out: dict[int, dict] = {}
    ext = g.j.get("extensions", {})
    if "VRM" in ext:
        for k, mp in enumerate(ext["VRM"].get("materialProperties", [])):
            if k >= len(g.j.get("materials", [])) or "MToon" not in mp.get("shader", ""):
                continue
            v, t = mp.get("vectorProperties", {}), mp.get("textureProperties", {})
            out[k] = {
                "shadeColor": v.get("_ShadeColor", [1, 1, 1, 1])[:3],
                "shadeTexture": t.get("_ShadeTexture"),
                "emissionColor": v.get("_EmissionColor", [0, 0, 0, 1])[:3],
                "emissionTexture": t.get("_EmissionMap"),
            }
    for k, m in enumerate(g.j.get("materials", [])):
        mt = m.get("extensions", {}).get("VRMC_materials_mtoon")
        if mt:
            st = mt.get("shadeMultiplyTexture") or {}
            out[k] = {
                "shadeColor": mt.get("shadeColorFactor", [1, 1, 1]),
                "shadeTexture": st.get("index"),
                "emissionColor": m.get("emissiveFactor", [0, 0, 0]),
                "emissionTexture": (m.get("emissiveTexture") or {}).get("index"),
            }
    return out


def _make_material(
    stage: Usd.Stage, path: Sdf.Path, m: dict, tex_image: dict, img_files: dict, mtoon: dict | None = None
) -> None:
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path.AppendChild("Surface"))
    sh.CreateIdAttr("UsdPreviewSurface")
    pbr = m.get("pbrMetallicRoughness", {})
    factor = [float(x) for x in pbr.get("baseColorFactor", [1, 1, 1, 1])]
    toon = mtoon is not None or "KHR_materials_unlit" in m.get("extensions", {})
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9 if toon else float(pbr.get("roughnessFactor", 1.0)))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0 if toon else float(pbr.get("metallicFactor", 0.0)))
    alpha = m.get("alphaMode", "OPAQUE")
    reader = None

    def tex_file(index):
        return img_files.get(tex_image.get(index)) if index is not None else None

    def texture(name: str, file: str, scale) -> UsdShade.Shader:
        nonlocal reader
        if reader is None:
            reader = UsdShade.Shader.Define(stage, path.AppendChild("st"))
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.String).Set("st")
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        t = UsdShade.Shader.Define(stage, path.AppendChild(name))
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(file)
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        t.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(*[float(x) for x in scale]))
        return t

    def connect(inp: str, shader: UsdShade.Shader, out: str = "rgb") -> None:
        tn = Sdf.ValueTypeNames.Float if out == "a" else Sdf.ValueTypeNames.Float3
        it = Sdf.ValueTypeNames.Float if out == "a" else Sdf.ValueTypeNames.Color3f
        sh.CreateInput(inp, it).ConnectToSource(shader.CreateOutput(out, tn))

    base_tex = (pbr.get("baseColorTexture") or {}).get("index")
    base_file = tex_file(base_tex)
    lit = TOON_LIT if toon else 1.0
    if base_file:
        base = texture("BaseColor", base_file, [factor[0] * lit, factor[1] * lit, factor[2] * lit, factor[3]])
        connect("diffuseColor", base)
    else:
        base = None
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[c * lit for c in factor[:3]]))
    if toon:
        mt = mtoon or {}
        glow = [float(x) for x in mt.get("emissionColor", m.get("emissiveFactor", [0, 0, 0]))]
        if max(glow) > 1.0 and not mt.get("emissionTexture"):
            # HDR emission color (halo...): glows on its own
            sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[c * TOON_GLOW for c in glow]))
        else:
            sc = [float(x) for x in mt.get("shadeColor", [1, 1, 1])]
            shade_file = tex_file(mt.get("shadeTexture")) or base_file
            k = [factor[i] * sc[i] * TOON_SHADE for i in range(3)]
            if shade_file:
                connect("emissiveColor", texture("Shade", shade_file, [*k, 1.0]))
            else:
                sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*k))
    if alpha in ("MASK", "BLEND") and base is not None:
        connect("opacity", base, "a")
        sh.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(
            float(m.get("alphaCutoff", 0.5)) if alpha == "MASK" else 0.05
        )
    mat.CreateSurfaceOutput().ConnectToSource(sh.CreateOutput("surface", Sdf.ValueTypeNames.Token))
