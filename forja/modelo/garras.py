"""La parte baja de cada pata tal como la deforma el juego: vértices de la malla y sus pesos sobre los huesos de la pata.

blender --factory-startup -b -P forja/modelo/garras.py -- rigged.blend rig_info.json creature.json SALIDA.json

La punta (articulación) no es lo que toca el suelo: la garra de la malla baja por debajo y, según la postura, más o
menos. Además, en nuestra araña las garras más bajas de L3, L4 y R4 pesan 0,85 a la tibia y 0,15 al tarso. Por eso se
guarda, por pata, una muestra de los vértices bajos (z de reposo < 15 cm) que solo dependen de los huesos de esa pata,
con sus pesos: la IK calcula su posición exacta con la piel lineal del rig (Σ w·M_hueso·B⁻¹·v) y les exige z ≥ 0.
Muestra: hasta 40 vértices por pata, por muestreo del punto más lejano empezando por el más bajo.
"""
import json
import sys

import bpy
import numpy as np

args = sys.argv[sys.argv.index("--") + 1:]
blend, info_path, cfg_path, out = args[:4]
Z_MAX, N_MAX = 0.15, 40
bpy.ops.wm.open_mainfile(filepath=blend)
info = json.load(open(info_path))
cfg = json.load(open(cfg_path, encoding="utf-8"))
body = bpy.data.objects["body"]
names = [b["name"] for b in info["bones"]]
parent = [b["parent"] for b in info["bones"]]
kp = {k["name"]: k for k in info["keypoints"]}
mw = np.array(body.matrix_world)
co = np.array([mw[:3, :3] @ np.array(v.co) + mw[:3, 3] for v in body.data.vertices])
groups = {g.index: g.name for g in body.vertex_groups}
res = {}
for c in cfg.get("contacts", []):
    last = kp[c]["bone"]
    chain = [last]
    while parent[chain[0]] > 0:  # hasta el hueso que cuelga de la raíz
        chain.insert(0, parent[chain[0]])
    cn = [names[b] for b in chain]
    idx, wts = [], []
    for v in body.data.vertices:
        if co[v.index, 2] >= Z_MAX:
            continue
        w = {groups[g.group]: g.weight for g in v.groups if g.weight > 1e-4}
        if not w or any(n not in cn for n in w):
            continue
        s = sum(w.values())
        idx.append(v.index)
        wts.append([w.get(n, 0.0) / s for n in cn])
    if not idx:
        continue
    P = co[idx]
    sel = [int(np.argmin(P[:, 2]))]
    d = np.linalg.norm(P - P[sel[0]], axis=1)
    while len(sel) < min(N_MAX, len(idx)):
        k = int(np.argmax(d))
        sel.append(k)
        d = np.minimum(d, np.linalg.norm(P - P[k], axis=1))
    res[c] = {"huesos": cn, "vertices": np.round(P[sel], 5).tolist(), "pesos": np.round(np.array(wts)[sel], 4).tolist(),
              "z_min_reposo": float(P[:, 2].min()), "candidatos": len(idx)}
    print(f"{c}: {len(idx)} vértices bajos de la pata, {len(sel)} elegidos; z mínima en reposo {P[:, 2].min() * 100:.2f} cm", flush=True)
json.dump(res, open(out, "w"), indent=1)
print("garras", out)
