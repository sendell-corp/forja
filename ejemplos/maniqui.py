"""A simple mannequin to try Forja without any game model (Un maniquí para probar Forja sin modelos del juego).

python -m ejemplos.maniqui SALIDA.glb

Writes a skinned glTF 2.0 binary (GLB) made of boxes, one per bone, with the skeleton that `forja.gesto` expects:
root > pelvis > spine > chest > neck > head, chest > upper_arm_{L,R} > forearm_{L,R} > hand_{L,R} and
pelvis > thigh_{L,R} > shin_{L,R} > foot_{L,R}. Y up, metres, facing +Z, left = +X. Each bone's local Y axis points
to its child, as Blender exports it. Every vertex belongs to one bone (rigid skinning).

It is generated from scratch by this script: no third-party asset, same licence as the rest of the repository.
To use your own character instead, export a GLB with the same bone names (see README).
"""
import json
import pathlib
import struct
import sys

import numpy as np

# head/tail of every bone in glTF world coordinates (Y up, facing +Z, character's left = +X)
HUESOS = [
    # name,        parent,       head,                  tail
    ("root",        None,        (0, 0, 0),             (0, 0.1, 0)),
    ("pelvis",      "root",      (0, 0.95, 0),          (0, 1.05, 0)),
    ("spine",       "pelvis",    (0, 1.05, 0),          (0, 1.25, 0)),
    ("chest",       "spine",     (0, 1.25, 0),          (0, 1.48, 0)),
    ("neck",        "chest",     (0, 1.48, 0),          (0, 1.58, 0)),
    ("head",        "neck",      (0, 1.58, 0),          (0, 1.80, 0)),
]
for s, sx in (("L", 1), ("R", -1)):
    HUESOS += [
        (f"upper_arm_{s}", "chest",          (0.18 * sx, 1.43, 0),    (0.29 * sx, 1.16, 0)),
        (f"forearm_{s}",   f"upper_arm_{s}", (0.29 * sx, 1.16, 0),    (0.37 * sx, 0.91, 0.02)),
        (f"hand_{s}",      f"forearm_{s}",   (0.37 * sx, 0.91, 0.02), (0.40 * sx, 0.73, 0.02)),
        (f"thigh_{s}",     "pelvis",         (0.10 * sx, 0.92, 0),    (0.10 * sx, 0.50, 0.015)),
        (f"shin_{s}",      f"thigh_{s}",     (0.10 * sx, 0.50, 0.015), (0.10 * sx, 0.09, -0.01)),
        (f"foot_{s}",      f"shin_{s}",      (0.10 * sx, 0.09, -0.01), (0.10 * sx, 0.02, 0.15)),
    ]

# half widths (local X, local Z) of each bone's box
ANCHO = {"pelvis": (0.15, 0.10), "spine": (0.14, 0.09), "chest": (0.17, 0.10), "neck": (0.05, 0.05),
         "head": (0.09, 0.10), "upper_arm": (0.05, 0.05), "forearm": (0.04, 0.04), "hand": (0.015, 0.045),
         "thigh": (0.07, 0.07), "shin": (0.05, 0.05)}


def marco(y):
    y = np.asarray(y, float)
    y /= np.linalg.norm(y)
    zref = np.array([0, 0, 1.0]) if abs(y[2]) < 0.9 else np.array([0, -1.0, 0])
    x = np.cross(y, zref)
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    return np.stack([x, y, z], 1)


def quat(R):
    """Rotation matrix → (x, y, z, w)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0] * 4
        q[i] = 0.25 * s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
    q = np.array(q)
    return q / np.linalg.norm(q)


def caja(lo, hi):
    """24 vertices with flat normals and 36 indices of an axis-aligned box in the frame where it is defined."""
    pos, nor, idx = [], [], []
    for eje in range(3):
        for sg in (-1, 1):
            n = np.zeros(3)
            n[eje] = sg
            u, v = [e for e in range(3) if e != eje]
            b = len(pos)
            for a, c in ((0, 0), (1, 0), (1, 1), (0, 1)):
                p = np.zeros(3)
                p[eje] = hi[eje] if sg > 0 else lo[eje]
                p[u] = hi[u] if a else lo[u]
                p[v] = hi[v] if c else lo[v]
                pos.append(p), nor.append(n)
            tri = [0, 1, 2, 0, 2, 3]
            if np.dot(np.cross(np.eye(3)[u], np.eye(3)[v]), n) < 0:
                tri = [0, 2, 1, 0, 3, 2]
            idx += [b + t for t in tri]
    return np.array(pos), np.array(nor), idx


def construir():
    nombres = [h[0] for h in HUESOS]
    W = {}
    for nm, _, head, tail in HUESOS:
        M = np.eye(4)
        M[:3, :3] = marco(np.subtract(tail, head))
        M[:3, 3] = head
        W[nm] = M
    nodes = []
    for nm, par, _, _ in HUESOS:
        L = W[nm] if par is None else np.linalg.inv(W[par]) @ W[nm]
        nodes.append({"name": nm, "translation": [float(v) for v in L[:3, 3]],
                      "rotation": [float(v) for v in quat(L[:3, :3])]})
    for k, (nm, par, _, _) in enumerate(HUESOS):
        if par is not None:
            nodes[nombres.index(par)].setdefault("children", []).append(k)

    P, N, J, I = [], [], [], []
    for k, (nm, _, head, tail) in enumerate(HUESOS):
        if nm == "root":
            continue
        papel = nm.rsplit("_", 1)[0] if nm.endswith(("_L", "_R")) else nm
        if papel == "foot":
            # world-aligned shoe: sole on the ground, heel behind the ankle, toes in front
            x0 = head[0]
            p, n, idx = caja(np.array([x0 - 0.05, 0.0, -0.07]), np.array([x0 + 0.05, 0.08, 0.17]))
        else:
            L = np.linalg.norm(np.subtract(tail, head))
            wx, wz = ANCHO[papel]
            p, n, idx = caja(np.array([-wx, 0, -wz]), np.array([wx, L, wz]))
            R, t = W[nm][:3, :3], W[nm][:3, 3]
            p, n = p @ R.T + t, n @ R.T
        I += [len(P) + i for i in idx]
        P += list(p)
        N += list(n)
        J += [k] * len(p)
    P, N = np.array(P, np.float32), np.array(N, np.float32)
    joints = np.zeros((len(P), 4), np.uint16)
    joints[:, 0] = J
    weights = np.zeros((len(P), 4), np.float32)
    weights[:, 0] = 1
    ibm = np.stack([np.linalg.inv(W[nm]).T.reshape(-1) for nm in nombres]).astype(np.float32)  # column-major
    indices = np.array(I, np.uint16)

    bin_ = bytearray()
    views, accs = [], []

    def add(arr, typ, comp, target=None, minmax=False):
        while len(bin_) % 4:
            bin_.append(0)
        views.append({"buffer": 0, "byteOffset": len(bin_), "byteLength": arr.nbytes, **({"target": target} if target else {})})
        bin_.extend(arr.tobytes())
        a = {"bufferView": len(views) - 1, "componentType": comp, "count": len(arr), "type": typ}
        if minmax:
            a["min"], a["max"] = [float(v) for v in arr.min(0)], [float(v) for v in arr.max(0)]
        accs.append(a)
        return len(accs) - 1

    pa = add(P, "VEC3", 5126, 34962, True)
    na = add(N, "VEC3", 5126, 34962)
    ja = add(joints, "VEC4", 5123, 34962)
    wa = add(weights, "VEC4", 5126, 34962)
    ia = add(indices, "SCALAR", 5123, 34963)
    ba = add(ibm, "MAT4", 5126)

    n_huesos = len(nodes)
    nodes.append({"name": "maniqui_surface", "mesh": 0, "skin": 0})
    nodes.append({"name": "maniqui_rig", "children": [n_huesos, 0]})
    js = {
        "asset": {"version": "2.0", "generator": "forja ejemplos/maniqui.py"},
        "scene": 0, "scenes": [{"nodes": [n_huesos + 1]}], "nodes": nodes,
        "meshes": [{"name": "maniqui_surface", "primitives": [{"attributes": {"POSITION": pa, "NORMAL": na,
                    "JOINTS_0": ja, "WEIGHTS_0": wa}, "indices": ia, "material": 0}]}],
        "materials": [{"name": "maniqui", "pbrMetallicRoughness": {"baseColorFactor": [0.8, 0.8, 0.78, 1],
                       "metallicFactor": 0, "roughnessFactor": 0.7}}],
        "skins": [{"name": "maniqui_skin", "joints": list(range(n_huesos)), "inverseBindMatrices": ba,
                   "skeleton": 0}],
        "accessors": accs, "bufferViews": views, "buffers": [{"byteLength": len(bin_)}],
    }
    return js, bin_


def escribir(js, bin_, ruta):
    bin_ = bytes(bin_) + b"\0" * (-len(bin_) % 4)
    js["buffers"][0]["byteLength"] = len(bin_)
    j = json.dumps(js, separators=(",", ":")).encode()
    j += b" " * (-len(j) % 4)
    total = 12 + 8 + len(j) + 8 + len(bin_)
    pathlib.Path(ruta).write_bytes(struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(j), 0x4E4F534A)
                                   + j + struct.pack("<II", len(bin_), 0x004E4942) + bin_)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    js, b = construir()
    escribir(js, b, sys.argv[1])
    print("maniquí:", sys.argv[1], len(js["nodes"]) - 2, "huesos")
