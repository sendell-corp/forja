"""Error de la lectura (el lector, antes de ajustar) contra la verdad proyectada, por vista, en píxeles y en cm.

python -m forja.calidad.lectura DET.npz VERDAD/truth_forja.json creature.json [--json SALIDA.json]
"""
import argparse
import json

import numpy as np

from forja.lector.leer import cargar
from forja.vistas.layout import Layout


def medir(det, truth_joints, lay, names):
    J = np.asarray(truth_joints)
    T = min(len(J), det["uv"].shape[1])
    out = {}
    tips = [j for j, n in enumerate(names) if n.startswith("leg") and n.endswith("_3")]
    for vi, v in enumerate(det["vistas"]):
        e = np.linalg.norm(det["uv"][vi, :T] - lay[v].project(J[:T]), axis=-1)
        out[v] = {"media_px": float(e.mean()), "mediana_px": float(np.median(e)), "p95_px": float(np.percentile(e, 95)),
                  "media_cm": float(e.mean() * lay.M * 100), "puntas_px": float(e[:, tips].mean()),
                  "mas_de_20px_pct": float((e > 20).mean() * 100), "conf_media": float(det["conf"][vi, :T].mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("det"), ap.add_argument("truth"), ap.add_argument("cfg")
    ap.add_argument("--json")
    a = ap.parse_args()
    det = cargar(a.det)
    tr = json.load(open(a.truth, encoding="utf-8"))
    r = medir(det, tr["joints"], Layout(a.cfg), tr["names"])
    for v, x in r.items():
        print(f"{v}: {x['media_px']:.2f} px ({x['media_cm']:.2f} cm), mediana {x['mediana_px']:.2f}, p95 {x['p95_px']:.2f}, "
              f"puntas {x['puntas_px']:.2f} px, >20 px {x['mas_de_20px_pct']:.1f} %, conf {x['conf_media']:.2f}")
    if a.json:
        json.dump(r, open(a.json, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
