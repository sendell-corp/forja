"""Monta la vitrina: una cartela de 1 s y luego el clip, para cada clip renderizado por blender_vitrina.py.

python -m forja.paquete.vitrina_video RENDER_DIR SALIDA.mp4 [--fps 24]
"""
import argparse
import pathlib

import cv2
import numpy as np

TITULOS = {
    "forja_caminar": ("Caminar", "ciclo de 1 s, 8 patas en tetrápodos alternos"),
    "forja_trotar": ("Trotar", "ciclo de 0,67 s, más rápido y más alto"),
    "forja_girar": ("Girar en el sitio", "20 grados por segundo"),
    "forja_alzarse": ("Alzarse amenazando", "sobre las patas de atrás, chasquido de colmillos"),
    "forja_mordisco": ("Mordisco", "carga, lanza la cabeza y cierra los colmillos"),
    "forja_abalanzarse": ("Abalanzarse", "se agacha, salta y aterriza con las delanteras"),
    "forja_retroceder_herida": ("Retroceder herida", "golpe de frente: sale hacia atrás y se recompone"),
}
FONDO = (40, 40, 44)


def texto(img, s, y, escala, grosor, color=(245, 245, 245)):
    (w, h), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_DUPLEX, escala, grosor)
    cv2.putText(img, s, ((img.shape[1] - w) // 2, y), cv2.FONT_HERSHEY_DUPLEX, escala, color, grosor, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("render"), ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=24)
    a = ap.parse_args()
    d = pathlib.Path(a.render)
    clips = [c for c in TITULOS if (d / c).is_dir()]
    first = cv2.imread(str(sorted((d / clips[0]).glob("*.png"))[0]))
    H, W = first.shape[:2]
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    for c in clips:
        t, sub = TITULOS[c]
        card = np.full((H, W, 3), FONDO, np.uint8)
        texto(card, t, H // 2 - 10, 1.8, 3)
        texto(card, sub, H // 2 + 45, 0.85, 2, (200, 200, 205))
        for _ in range(int(a.fps)):
            vw.write(card)
        for f in sorted((d / c).glob("*.png")):
            img = cv2.imread(str(f))
            cv2.rectangle(img, (0, H - 54), (W, H), FONDO, -1)
            cv2.putText(img, t, (20, H - 16), cv2.FONT_HERSHEY_DUPLEX, 1.1, (245, 245, 245), 2, cv2.LINE_AA)
            vw.write(img)
    vw.release()
    print("vitrina", a.out, clips)


if __name__ == "__main__":
    main()
