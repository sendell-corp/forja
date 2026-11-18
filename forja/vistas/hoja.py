"""Hoja de cuadros de un clip: algunos cuadros del video de N vistas, apilados (para el informe).

python -m forja.vistas.hoja VIDEO.mp4 SALIDA.jpg [cuadros=0,30,60,90] [--escala 0.5] [--puntos DET.npz]
Con --puntos dibuja encima las lecturas del lector (blanco con borde negro) en cada vista.
"""
import argparse

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video"), ap.add_argument("out")
    ap.add_argument("cuadros", nargs="?", default="0,30,60,90")
    ap.add_argument("--escala", type=float, default=0.5)
    ap.add_argument("--puntos")
    a = ap.parse_args()
    want = [int(x) for x in a.cuadros.split(",")]
    cap = cv2.VideoCapture(a.video)
    frames, t = {}, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if t in want:
            frames[t] = f
        t += 1
    det = np.load(a.puntos) if a.puntos else None
    rows = []
    for t in want:
        f = frames[t].copy()
        if det is not None:
            uv = det["uv"] if "uv" in det.files else np.stack([det["front_uv"], det["side_uv"]])
            w = f.shape[1] // uv.shape[0]
            for vi in range(uv.shape[0]):
                for u, v in uv[vi, t]:
                    cv2.circle(f, (int(u + vi * w), int(v)), 4, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(f, (int(u + vi * w), int(v)), 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.putText(f, f"cuadro {t}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(cv2.resize(f, None, fx=a.escala, fy=a.escala, interpolation=cv2.INTER_AREA))
    cv2.imwrite(a.out, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 82])
    print("hoja", a.out, len(rows), "cuadros")


if __name__ == "__main__":
    main()
