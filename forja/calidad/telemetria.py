"""Telemetría de la vitrina para el crítico de video (skill critico-de-video, TELEMETRIA.md): banderas por segundo del
video montado, sacadas de los propios clips (lo que las hojas a 4 cuadros por segundo no ven).

python -m forja.calidad.telemetria SALIDA_FABRICAR rig_info.json creature.json telemetria-segundos.json [--vueltas 3]

Banderas: desliza:<clip> (punta apoyada que se mueve más de 0,5 cm en un cuadro, R3), atraviesa:<clip> (dos partes que
penetran más de 1 cm según Jolt, R4), bajo-suelo:<clip> (casco bajo el suelo más de 1 cm, R4), marcha-atras:<clip>
(el root motion va hacia atrás de donde mira la criatura a más de 0,2 m/s, R3; en «retroceder herida» es lo pedido).
La línea de tiempo es la de vitrina_video.py: 1 s de cartela y el clip (los bucles, `vueltas` veces).
"""
import argparse
import json
import pathlib

import numpy as np

from forja.fisica.limpieza import cargar_cascos, diagnostico, pares_limpios
from forja.paquete.vitrina_video import TITULOS
from forja.rig import Rig, load_motion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("salida"), ap.add_argument("info"), ap.add_argument("cfg"), ap.add_argument("out")
    ap.add_argument("--vueltas", type=int, default=3)
    ap.add_argument("--fps", type=float, default=24)
    a = ap.parse_args()
    rig = Rig(a.info)
    cascos = cargar_cascos(a.cfg)
    pares, _ = pares_limpios(rig, cascos)
    base = pathlib.Path(a.salida)
    anot = json.load(open(base / "forja_obrera.anotaciones.json", encoding="utf-8"))
    por_seg = {}
    t0 = 0.0
    for clip in TITULOS:
        if clip not in anot["clips"]:
            continue
        c = clip.replace("forja_", "")
        meta = json.load(open(base / "clips" / c / "meta.json", encoding="utf-8"))
        m = load_motion(base / "clips" / c / "limpio.json")
        d = diagnostico(rig, m, meta, cascos, pares)
        T = len(m["trans"])
        flags = [set() for _ in range(T)]
        for t, _, _, dep, _, _ in d["_jolt"]["pares"]:
            if dep > 0.01:
                flags[t].add(f"atraviesa:{c}")
        for t, _, dep, _ in d["_jolt"]["suelo"]:
            if dep > 0.01:
                flags[t].add(f"bajo-suelo:{c}")
        J, _ = rig.forward(m["trans"], m["root_R"], m["euler"])
        st = np.array(meta["stance"], bool)
        for k, n in enumerate(meta["tips"]):
            j = rig.kp_names.index(n)
            dd = np.linalg.norm(np.diff(J[:, j, :2], axis=0), axis=-1)
            for t in np.where(st[1:, k] & st[:-1, k] & (dd > 0.005))[0]:
                flags[t + 1].add(f"desliza:{c}")
        rm = anot["clips"][clip]["root_motion"]
        x, z, yaw = (np.array(rm[k]) for k in ("x", "z", "yaw"))
        vx, vz = np.gradient(x) * a.fps, np.gradient(z) * a.fps
        fx, fz = np.sin(yaw), np.cos(yaw)  # la criatura mira a +Z (girada yaw)
        v = np.hypot(vx, vz)
        atras = (v > 0.2) & ((vx * fx + vz * fz) < -0.3 * v)
        for t in np.where(atras)[0]:
            flags[t].add(f"marcha-atras:{c}")
        reps = a.vueltas if anot["clips"][clip]["bucle"] else 1
        t0 += 1.0  # cartela
        for i in range(T * reps):
            s = int(t0 + i / a.fps)
            por_seg.setdefault(s, set()).update(flags[i % T])
        t0 += T * reps / a.fps
    out = [{"s": s, "banderas": sorted(f)} for s, f in sorted(por_seg.items()) if f]
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"telemetría: {len(out)} segundos con banderas de {int(t0) + 1}", out)


if __name__ == "__main__":
    main()
