"""Las cámaras de la vista previa (coordenadas de Blender = las de Forja: Z arriba, el personaje mira a −Y).

python -m forja.gesto.camaras CARPETA_DEL_GESTO CAMARAS.json [--primera DESDE:HASTA]

- tercera persona: quieta, de 3/4 por delante y a la izquierda del personaje, encuadrando todo el recorrido;
- primera persona: a la altura de los ojos y 3 cm por delante de la cara (hueso de la cabeza: Y arriba, Z adelante), mirando entre la cabeza (20° más abajo) y la caja (60 %),
  con un suavizado gaussiano de 2 cuadros (sin el temblor de una cámara pegada al hueso).
"""
import argparse
import json
import pathlib

import numpy as np

from forja.gesto.__main__ import suave


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carpeta"), ap.add_argument("salida")
    ap.add_argument("--primera", default=None)
    a = ap.parse_args()
    d = np.load(pathlib.Path(a.carpeta) / "movimiento.npz")
    P, nombres = d["P"], list(d["nombres"])
    T = len(P)
    pel = P[:, 0, :3, 3]
    centro = np.array([0.0, (pel[:, 1].max() + d["caja"][:, 1].min()) / 2, 0.72])
    out = {"fps": 30, "cuadros": T,
           "tercera": {"pos": (centro + np.array([2.75, -2.55, 0.55])).tolist(), "mira": centro.tolist(), "lente": 35}}
    if a.primera:
        i0, i1 = (int(x) for x in a.primera.split(":"))
        Hm = P[:, nombres.index("head")]
        Y, Z, o = Hm[:, :3, 1], Hm[:, :3, 2], Hm[:, :3, 3]
        ojo = suave(o + 0.09 * Y + 0.19 * Z, 2.0)  # delante de la cara y las gafas (la malla llega a Z = 0,16)
        ang = np.radians(20)
        cabeza = Z * np.cos(ang) - Y * np.sin(ang)
        a_caja = d["caja"][:, :3] - ojo
        a_caja /= np.linalg.norm(a_caja, axis=-1, keepdims=True)
        mira = suave(ojo + 0.4 * cabeza + 0.6 * a_caja, 2.0)  # los ojos siguen la caja (con la cabeza se perdía de vista)
        out["primera"] = {"desde": i0, "hasta": i1, "pos": ojo.round(5).tolist(), "mira": mira.round(5).tolist(), "lente": 20}
    json.dump(out, open(a.salida, "w"))
    print("cámaras", a.salida, out["tercera"])


if __name__ == "__main__":
    main()
