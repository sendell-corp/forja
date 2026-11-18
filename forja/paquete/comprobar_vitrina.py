"""Compara las cabezas de hueso que midió Blender al importar el GLB (blender_vitrina.py → comprobacion.json) con la
cinemática de Forja en el mismo cuadro (clip en el sitio). Si coinciden, el GLB se reproduce igual fuera de Forja.

python -m forja.paquete.comprobar_vitrina comprobacion.json FORJA.glb SALIDA_FABRICAR rig_info.json forja_map.json
"""
import json
import pathlib
import sys

import numpy as np

from forja.paquete.glb import convertir, fk_juego, leer
from forja.rig import Rig, load_motion


def main():
    comp_path, glb, salida, info, mapa_path = sys.argv[1:6]
    comp = json.load(open(comp_path))
    js, _ = leer(glb)
    rig = Rig(info)
    mapa = json.load(open(mapa_path, encoding="utf-8"))
    peor = 0.0
    for clip, cuadros in comp.items():
        m = load_motion(pathlib.Path(salida) / "clips" / clip.replace("forja_", "") / "limpio.json")
        pistas, _ = convertir(rig, m, mapa, js, en_sitio=True)
        err = []
        for f, huesos in cuadros.items():
            pos = fk_juego(js, pistas, int(f))
            for h, pb in huesos.items():
                if h in pos:
                    g = pos[h]
                    err.append(np.linalg.norm(np.array(pb) - np.array([g[0], -g[2], g[1]])))
        e = max(err) * 1000
        peor = max(peor, e)
        print(f"{clip}: {len(err)} cabezas de hueso, error máximo {e:.3f} mm")
    print(f"GLB en Blender frente a Forja: {peor:.3f} mm como mucho")


if __name__ == "__main__":
    main()
