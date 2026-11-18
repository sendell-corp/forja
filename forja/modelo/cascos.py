"""Un casco convexo por hueso, sacado de la malla: lo que Jolt usa para saber si algo atraviesa el suelo u otra pata.

blender --factory-startup -b -P forja/modelo/cascos.py -- rigged.blend rig_info.json SALIDA.json

Por hueso, los vértices de reposo con peso ≥ 0,5 a ese hueso (coordenadas de anymove, metros), partidos en piezas
con k-medias (una por cada ~120 vértices, hasta 8: el cuerpo y el abdomen son cóncavos y un solo casco convexo los
haría mucho más grandes de lo que son), cada pieza reducida a 48 puntos por muestreo del punto más lejano; y cuántos
vértices tenía cada pieza (para repartir la masa al estimar el centro de masas).
"""
import json
import sys

import bpy
import numpy as np

args = sys.argv[sys.argv.index("--") + 1:]
blend, info_path, out = args[:3]
N_MAX = 64
bpy.ops.wm.open_mainfile(filepath=blend)
info = json.load(open(info_path))
body = bpy.data.objects["body"]
mw = np.array(body.matrix_world)
co = np.array([mw[:3, :3] @ np.array(v.co) + mw[:3, 3] for v in body.data.vertices])
groups = {g.index: g.name for g in body.vertex_groups}
por_hueso = {}
for v in body.data.vertices:
    for g in v.groups:
        if g.weight >= 0.5:
            por_hueso.setdefault(groups[g.group], []).append(v.index)
N_MAX = 48


def fps(P, n):
    sel = [int(np.argmin(P[:, 2]))]
    d = np.linalg.norm(P - P[sel[0]], axis=1)
    while len(sel) < min(n, len(P)):
        k = int(np.argmax(d))
        sel.append(k)
        d = np.minimum(d, np.linalg.norm(P - P[k], axis=1))
    return P[sel]


def kmedias(P, k, iters=30):
    rng = np.random.default_rng(0)
    C = fps(P, k)
    for _ in range(iters):
        lab = np.argmin(((P[:, None] - C[None]) ** 2).sum(-1), 1)
        C = np.array([P[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    return lab


res = {}
for b in info["bones"]:
    idx = por_hueso.get(b["name"], [])
    if len(idx) < 4:
        continue
    P = co[idx]
    k = int(min(8, max(1, round(len(idx) / 120))))
    lab = kmedias(P, k) if k > 1 else np.zeros(len(P), int)
    piezas = []
    for j in range(k):
        Q = P[lab == j]
        if len(Q) >= 4:
            piezas.append({"puntos": np.round(fps(Q, N_MAX), 5).tolist(), "vertices": int(len(Q))})
    res[b["name"]] = piezas
    print(f"{b['name']}: {len(idx)} vértices en {len(piezas)} piezas", flush=True)
json.dump(res, open(out, "w"), indent=1)
print("cascos", out)
