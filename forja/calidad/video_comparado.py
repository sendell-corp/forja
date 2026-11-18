"""Video comparado: varios renders del mismo clip (verdad y ajustes) uno al lado de otro, con su rótulo.

python -m forja.calidad.video_comparado SALIDA.mp4 DIR1:rótulo1 DIR2:rótulo2 … [--hoja SALIDA.jpg --cuadros 0,40,80]
Cada DIR es una salida de apply_motion.py con --views persp (DIR/persp/f_*.png).
"""
import argparse
import pathlib

import cv2
import numpy as np

BG = (252, 252, 251)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("pares", nargs="+")
    ap.add_argument("--hoja")
    ap.add_argument("--cuadros", default="0,40,80")
    ap.add_argument("--fps", type=float, default=24)
    a = ap.parse_args()
    cols = []
    for p in a.pares:
        d, lab = p.rsplit(":", 1)
        cols.append((sorted((pathlib.Path(d) / "persp").glob("*.png")), lab))
    n = min(len(f) for f, _ in cols)
    h, w = cv2.imread(str(cols[0][0][0])).shape[:2]
    W = w * len(cols)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, h + 40))
    want = [int(x) for x in a.cuadros.split(",")]
    sheet = []
    for t in range(n):
        row = np.full((h + 40, W, 3), BG[::-1], np.uint8)
        for k, (files, lab) in enumerate(cols):
            img = cv2.imread(str(files[t]), cv2.IMREAD_UNCHANGED)
            if img.shape[2] == 4:  # fondo transparente sobre el gris claro
                al = img[..., 3:] / 255.0
                img = (img[..., :3] * al + np.array(BG[::-1]) * (1 - al)).astype(np.uint8)
            row[40:, k * w:(k + 1) * w] = img
            cv2.putText(row, lab, (k * w + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(row, f"{t}", (W - 60, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1, cv2.LINE_AA)
        vw.write(row)
        if t in want:
            sheet.append(row)
    vw.release()
    if a.hoja and sheet:
        cv2.imwrite(a.hoja, np.vstack(sheet), [cv2.IMWRITE_JPEG_QUALITY, 82])
    print("video", a.out, n, "cuadros", W, "x", h + 40)


if __name__ == "__main__":
    main()
