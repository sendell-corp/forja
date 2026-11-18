"""Hoja de muestras del dataset de N vistas: cada muestra compuesta en su cuadro, con las articulaciones verdaderas
dibujadas (deben caer sobre las articulaciones en las tres vistas).

python -m forja.lector.muestras DATASET creature.json SALIDA.jpg [n=6] [--vistas front,side,top]
"""
import argparse
import json
import pathlib

import cv2
import numpy as np

from forja.vistas.layout import Layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"), ap.add_argument("cfg"), ap.add_argument("out")
    ap.add_argument("n", nargs="?", type=int, default=6)
    ap.add_argument("--vistas", default="front,side,top")
    a = ap.parse_args()
    lay = Layout(a.cfg, a.vistas.split(","))
    metas = sorted(pathlib.Path(a.data).glob("*_meta.json"))[:a.n]
    rows = []
    for m in metas:
        meta = json.loads(m.read_text())
        s = meta["scale"]
        jw = np.array(meta["joints_world"])
        renders = {}
        for v in lay.views:
            img = cv2.imread(str(m).replace("_meta.json", f"_{v.name}.png"), cv2.IMREAD_UNCHANGED)[..., [2, 1, 0, 3]]
            renders[v.name] = cv2.resize(img, (v.w, lay.H), interpolation=cv2.INTER_CUBIC)
        fr = np.ascontiguousarray(lay.compose(renders)[..., ::-1])
        for v, x0 in zip(lay.views, lay.x0):
            for u, vv in v.project(jw):
                cv2.circle(fr, (int(u + x0), int(vv)), 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(fr, (int(u + x0), int(vv)), 3, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.putText(fr, m.name[:5], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        rows.append(cv2.resize(fr, (lay.W // 2, lay.H // 2), interpolation=cv2.INTER_AREA))
    cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("hoja", a.out, len(rows), "muestras")


if __name__ == "__main__":
    main()
