"""Lee las articulaciones de un clip de N vistas con un lector entrenado, y compone ese clip desde los renders.

python -m forja.lector.leer componer RENDER_DIR creature.json SALIDA.mp4 [--vistas front,side,top] [--fps 24]
    renders de apply_motion.py (RENDER_DIR/front/*.png, side/, top/) → un video del cuadro, como el examen de anymove
python -m forja.lector.leer leer VIDEO.mp4 lector.pt creature.json SALIDA.npz [--vistas front,side,top]
    → SALIDA.npz con uv (V, T, J, 2) en píxeles de cada vista, conf (V, T, J), vistas y nombres

La lectura usa el modelo y las funciones de `mocap/detect.py` de anymove (Apache-2.0): cada vista se rellena a un
cuadrado con su gris, se reduce a 448 px y el pico de cada mapa de calor da el punto y la confianza.
"""
import argparse
import pathlib

import cv2
import numpy as np

from forja.vistas.layout import Layout


def componer(render_dir, lay, out, fps=24):
    d = pathlib.Path(render_dir)
    files = {v: sorted((d / v).glob("*.png")) for v in lay.names}
    n = min(len(f) for f in files.values())
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (lay.W, lay.H))
    for t in range(n):
        renders = {}
        for v in lay.views:
            img = cv2.imread(str(files[v.name][t]), cv2.IMREAD_UNCHANGED)[..., [2, 1, 0, 3]]
            if img.shape[0] != lay.H:
                img = cv2.resize(img, (v.w, lay.H), interpolation=cv2.INTER_CUBIC)
            renders[v.name] = img
        vw.write(np.ascontiguousarray(lay.compose(renders)[..., ::-1]))
    vw.release()
    print("video", out, n, "cuadros", lay.W, "x", lay.H, flush=True)


def leer(video, reader, lay, out):
    import torch  # noqa: F401  (solo aquí: componer no necesita torch)
    from mocap.detect import load, read_frames, run
    frames, fps = read_frames(str(video))
    if frames.shape[1] != lay.H or frames.shape[2] != lay.W:
        frames = np.array([cv2.resize(f, (lay.W, lay.H), interpolation=cv2.INTER_AREA) for f in frames])
    model, names = load(str(reader))
    uv, conf = [], []
    for v, x0 in zip(lay.views, lay.x0):
        u, c = run(model, [f[:, x0:x0 + v.w] for f in frames])
        uv.append(u)
        conf.append(c)
    uv, conf = np.array(uv), np.array(conf)
    np.savez(out, uv=uv, conf=conf, vistas=np.array(lay.names), names=np.array(names), fps=fps)
    print(f"{len(frames)} cuadros a {fps:.2f} fps; confianza media " +
          ", ".join(f"{n} {c.mean():.2f}" for n, c in zip(lay.names, conf)), flush=True)


def desde_anymove(npz):
    """det.npz de mocap/detect.py (dos vistas) al formato de Forja."""
    d = np.load(npz)
    return {"uv": np.stack([d["front_uv"], d["side_uv"]]), "conf": np.stack([d["front_conf"], d["side_conf"]]),
            "vistas": ["front", "side"], "names": [str(n) for n in d["names"]], "fps": float(d["fps"])}


def cargar(npz):
    d = np.load(npz)
    if "front_uv" in d.files:
        return desde_anymove(npz)
    return {"uv": d["uv"], "conf": d["conf"], "vistas": [str(v) for v in d["vistas"]], "names": [str(n) for n in d["names"]],
            "fps": float(d["fps"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["componer", "leer"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--vistas", default="front,side,top")
    ap.add_argument("--fps", type=float, default=24)
    a = ap.parse_args()
    if a.cmd == "componer":
        render_dir, cfg, out = a.args
        componer(render_dir, Layout(cfg, a.vistas.split(",")), out, a.fps)
    else:
        video, reader, cfg, out = a.args
        leer(video, reader, Layout(cfg, a.vistas.split(",")), out)


if __name__ == "__main__":
    main()
