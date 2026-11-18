"""Banderas por segundo de un gesto para el crítico de video (formato de TELEMETRIA.md de la skill critico-de-video).

python -m forja.gesto.telemetria MODELO.glb CARPETA_DEL_GESTO telemetria-segundos.json [--desde-segundo 0]

Banderas: `desliza` (un pie apoyado se mueve > 0,5 cm en un cuadro), `bajo-suelo` (la malla baja > 0,5 cm del suelo),
`atraviesa` (la malla del cuerpo, sin las manos, entra > 0,5 cm en la caja), `mano-al-aire` (con la caja agarrada, una
palma a más de 1 cm de su cara), `caja-al-aire` (la caja suelta y por encima del suelo más de 1 cm sin estar cayendo).
Con --desde-segundo se desplazan los segundos (la toma en primera persona va detrás de la de tercera en el video).
"""
import argparse
import json
import pathlib

import numpy as np

from forja.gesto.cuerpo import LADOS, Cuerpo
from forja.gesto.gestos import FPS
from forja.gesto.humanoide import C, Humanoide
from forja.rig import quat_to_matrix


def banderas(h, d, eventos):
    cu = Cuerpo(h)
    P, caja = d["P"], d["caja"]
    T = len(P)
    V = h.piel(h.mundos_gltf(P)) @ C
    dom = h.VJ[np.arange(len(h.V)), h.VW.argmax(1)]
    nodo = {n: h.skin_joints.index(h.rig_nodes[k]) for k, n in enumerate(cu.rig.names)}
    half = np.array([0.36, 0.26, 0.26]) / 2
    Rb = quat_to_matrix(caja[:, [6, 3, 4, 5]])
    loc = np.einsum("tji,tvj->tvi", Rb, V - caja[:, None, :3])
    manos = np.isin(dom, [nodo["hand_L"], nodo["hand_R"]])
    prof = np.where(manos[None], -1, np.min(half - np.abs(loc), -1)).max(1)
    J = cu.rig.joints_from(P)
    f = [set() for _ in range(T)]
    for t in range(T):
        if V[t, :, 2].min() < -0.005:
            f[t].add("bajo-suelo")
        if prof[t] > 0.005:
            f[t].add("atraviesa")
    for s in LADOS:  # un pie que se mueve poco en horizontal y está abajo cuenta como apoyado
        for jn in (f"foot_{s}", f"foot_{s}_tail"):
            p = J[:, cu.rig.kp_names.index(jn)]
            mov = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=-1)
            abajo = p[1:, 2] < p[:, 2].min() + 0.005
            for t in np.nonzero(abajo & (mov > 0.005) & (mov < 0.02))[0]:
                f[t + 1].add("desliza")
    a, b = eventos["agarre"], eventos["suelta"] or T
    for s, sg in (("L", 1), ("R", -1)):
        sel = dom == nodo[f"hand_{s}"]
        dist = (sg * loc[:, sel, 0] - half[0]).min(1)
        for t in range(a, b):
            if dist[t] > 0.01:
                f[t].add("mano-al-aire")
    vz = np.gradient(caja[:, 2]) * FPS
    for t in range(T):
        if not (a <= t < b) and caja[t, 2] - half[2] > 0.01 and vz[t] > -0.05:
            f[t].add("caja-al-aire")
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modelo"), ap.add_argument("carpeta"), ap.add_argument("salida")
    ap.add_argument("--tomas", default="0:-1@0", help="tramos desde:hasta@segundo_inicial, separados por comas")
    a = ap.parse_args()
    h = Humanoide(a.modelo)
    d = np.load(pathlib.Path(a.carpeta) / "movimiento.npz")
    ev = json.loads(str(d["eventos"]))
    f = banderas(h, d, ev)
    seg = {}
    for toma in a.tomas.split(","):
        rango, s0 = toma.split("@")
        i0, i1 = (int(x) for x in rango.split(":"))
        i1 = len(f) if i1 < 0 else i1
        for k, t in enumerate(range(i0, i1)):
            s = int(float(s0) + k / FPS)
            seg.setdefault(s, set()).update(f[t])
    out = [{"s": s, "banderas": sorted(b)} for s, b in sorted(seg.items())]
    json.dump(out, open(a.salida, "w"), ensure_ascii=False)
    print("telemetría", a.salida, sum(len(x["banderas"]) for x in out), "banderas:",
          [x for x in out if x["banderas"]])


if __name__ == "__main__":
    main()
