"""Monta la vista previa: la toma en tercera persona y, detrás, la de primera persona, con un rótulo discreto abajo.

python -m forja.gesto.video RENDER_DIR SALIDA.mp4 --frase "pick up box from ground" [--fps 30] [--ingles]
"""
import argparse
import pathlib

import cv2

FONDO = (34, 34, 38)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("render"), ap.add_argument("out")
    ap.add_argument("--frase", default="")
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--ingles", action="store_true", help="rótulos en inglés (third person / first person)")
    a = ap.parse_args()
    t3, t1 = ("third person", "first person") if a.ingles else ("tercera persona", "primera persona")
    d = pathlib.Path(a.render)
    tomas = [(c, r) for c, r in (("tercera", f'"{a.frase}"  ·  {t3}'), ("primera", f'"{a.frase}"  ·  {t1}'))
             if (d / c).is_dir()]
    first = cv2.imread(str(sorted((d / tomas[0][0]).glob("*.png"))[0]))
    H, W = first.shape[:2]
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    for c, rotulo in tomas:
        for f in sorted((d / c).glob("*.png")):
            img = cv2.imread(str(f))
            cv2.rectangle(img, (0, H - 64), (W, H), FONDO, -1)
            cv2.putText(img, rotulo, (28, H - 22), cv2.FONT_HERSHEY_DUPLEX, 1.05, (240, 240, 240), 2, cv2.LINE_AA)
            vw.write(img)
    vw.release()
    print("video", a.out, [c for c, _ in tomas])


if __name__ == "__main__":
    main()
