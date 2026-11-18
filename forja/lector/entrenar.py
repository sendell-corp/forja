"""Entrena el lector de una criatura con N vistas (frente, lado y arriba).

python -m forja.lector.entrenar DATASET rig_info.json creature.json SALIDA.pt [épocas=20] [lote=8] [--vistas front,side,top]
                               [--desde lector.pt]

Es `mocap/train.py` de anymove (Apache-2.0, © Patch) con dos cambios:
1. **N vistas.** Cada muestra da una imagen por vista; los píxeles verdaderos de cada vista salen de `joints_world` del
   meta con el layout de Forja (así la vista de arriba no necesita tocar `render_dataset.py`). Se comprueba que frente y
   lado coinciden con los que escribió Blender.
2. **8 GB.** Lote por defecto 8 (en Windows, con 14 o más desborda a la RAM compartida y va 15 veces más lento) y como
   mucho FORJA_WORKERS procesos de carga (4): en Windows cada uno carga torch entero.
Lo demás (fondos al azar, degradación como de video, pérdida, optimizador, programa de lr, 4 % apartado) es lo suyo.
Imprime el error de las muestras apartadas por vista en píxeles del cuadro.
"""
import json
import math
import os
import pathlib
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mocap.model import HM, SIZE, JointReader, decode, square
from mocap.train import backdrop, heatmaps, photometric

from forja.vistas.layout import BG, FLOOR, Layout

NW = int(os.environ.get("FORJA_WORKERS", "4"))


class Vistas(Dataset):
    def __init__(self, metas, train, lay):
        self.metas, self.train, self.lay = metas, train, lay
        self.nv = len(lay.views)

    def __len__(self):
        return self.nv * len(self.metas)

    def __getitem__(self, i):
        rng = np.random.default_rng() if self.train else np.random.default_rng(i)
        m = self.metas[i // self.nv]
        vi = i % self.nv
        view = self.lay.views[vi]
        meta = json.loads(m.read_text())
        s = meta["scale"]
        rgba = cv2.imread(str(m).replace("_meta.json", f"_{view.name}.png"), cv2.IMREAD_UNCHANGED)[..., [2, 1, 0, 3]].astype(np.float32)
        h, w = rgba.shape[:2]
        kp = (view.project(np.array(meta["joints_world"])) * s).astype(np.float32)
        ground = 0 if view.name == "top" else round(self.lay.GROUND_V * s)
        if self.train:
            bg = backdrop(h, w, ground, rng)
        else:
            bg = np.empty((h, w, 3), np.float32)
            bg[:] = BG
            bg[ground:] = FLOOR
        a = rgba[..., 3:] / 255
        img = rgba[..., :3] * a + bg * (1 - a)
        if self.train and rng.random() < 0.7:  # los divisores del cuadro en los bordes compartidos
            if vi < self.nv - 1:
                img[:, -2:] = 34
            if vi > 0:
                img[:, :2] = 34
        _, ox, oy, side = square(img, bg[0, 0])
        k = SIZE / side
        A = np.array([[k, 0, ox * k], [0, k, oy * k]], np.float32)
        if self.train:  # encuadre: escala, giro pequeño y desplazamiento
            sc, rot = rng.uniform(0.85, 1.15), math.radians(rng.uniform(-4, 4))
            c = SIZE / 2
            R = cv2.getRotationMatrix2D((c, c), math.degrees(rot), sc)
            R[:, 2] += rng.uniform(-0.05, 0.05, 2) * SIZE
            A = R @ np.vstack([A, [0, 0, 1]])
        out = cv2.warpAffine(img, A.astype(np.float32), (SIZE, SIZE), flags=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=tuple(float(x) for x in bg[0, 0]))
        kp = kp @ A[:, :2].T + A[:, 2]
        if self.train:
            out = photometric(out, rng)
        x = torch.from_numpy(np.ascontiguousarray(out)).permute(2, 0, 1).float() / 255
        return x, torch.from_numpy(kp.astype(np.float32)), torch.tensor([ox, oy, side, s], dtype=torch.float32), vi


def check_layout(metas, lay):
    """Frente y lado: los píxeles de Forja contra los que escribió Blender (deben coincidir al milipíxel)."""
    worst = 0.0
    for m in metas[:20]:
        meta = json.loads(m.read_text())
        jw = np.array(meta["joints_world"])
        for v in lay.views:
            if f"{v.name}_uv" in meta:
                worst = max(worst, float(np.abs(v.project(jw) - np.array(meta[f"{v.name}_uv"])).max()))
    print(f"layout de Forja contra Blender: diferencia máxima {worst:.4f} px", flush=True)
    if worst > 0.05:
        raise SystemExit("el layout no coincide con el del dataset")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("data"), ap.add_argument("info"), ap.add_argument("cfg"), ap.add_argument("out")
    ap.add_argument("epochs", nargs="?", type=int, default=20)
    ap.add_argument("bs", nargs="?", type=int, default=8)
    ap.add_argument("--vistas", default="front,side,top")
    ap.add_argument("--desde")
    a = ap.parse_args()
    data, info_path, out, epochs, bs = a.data, a.info, a.out, a.epochs, a.bs
    opt = lambda k, d=None: {"--desde": a.desde}.get(k) or d
    lay = Layout(a.cfg, a.vistas.split(","))
    info = json.load(open(info_path))
    names = [k["name"] for k in info["keypoints"]]

    def split():
        metas = sorted(pathlib.Path(data).glob("*_meta.json"))
        val = [m for m in metas if int(m.name[:5]) % 25 == 0]
        return [m for m in metas if int(m.name[:5]) % 25 != 0], val

    train, val = split()
    check_layout(train, lay)
    print(f"{len(train)} de entrenamiento / {len(val)} apartadas, {len(names)} articulaciones, vistas {lay.names}, lote {bs}, "
          f"{epochs} épocas, {NW} procesos de carga", flush=True)
    vl = DataLoader(Vistas(val, False, lay), batch_size=bs, num_workers=min(4, NW))
    dev = "cuda"
    model = JointReader(len(names)).to(dev)
    if opt("--desde"):
        ck = torch.load(opt("--desde"), map_location="cpu", weights_only=True)
        model.load_state_dict(ck["state_dict"])
        print("desde", opt("--desde"), flush=True)
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    optim = torch.optim.AdamW([{"params": model.backbone.parameters(), "lr": 2e-5}, {"params": head, "lr": 1e-3}], weight_decay=0.05)
    base = [g["lr"] for g in optim.param_groups]
    step, t0, warm = 0, time.time(), 300
    hist = []
    for ep in range(epochs):
        train, _ = split()
        dl = DataLoader(Vistas(train, True, lay), batch_size=bs, shuffle=True, num_workers=NW, drop_last=True,
                        persistent_workers=False)
        model.train()
        for bi_, (x, kp, _, _) in enumerate(dl):
            f = 0.5 * (1 + math.cos(math.pi * (ep + bi_ / len(dl)) / epochs)) * min(1, (step + 1) / warm)
            for g, b0 in zip(optim.param_groups, base):
                g["lr"] = b0 * f
            x, kp = x.to(dev, non_blocking=True), kp.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x)
            loss = F.mse_loss(pred.float(), heatmaps(kp)) * 1000
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            step += 1
            if step % 100 == 0:
                print(f"ep {ep} paso {step} pérdida {loss.item():.4f} lr {f:.3f} {time.time() - t0:.0f}s", flush=True)
        model.eval()
        errs = {n: [] for n in lay.names}
        with torch.no_grad():
            for x, kp, geo, vi in vl:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(x.to(dev)).float()
                uv, _ = decode(pred)
                uv = uv.cpu() * SIZE / HM
                lay_px = lambda p: (p * geo[:, None, 2:3] / SIZE - geo[:, None, :2]) / geo[:, None, 3:4]
                e = (lay_px(uv) - lay_px(kp)).norm(dim=-1)
                for r in range(len(x)):
                    errs[lay.names[int(vi[r])]].append(e[r].numpy())
        msg = {v: float(np.mean(errs[v])) for v in errs}
        med = {v: float(np.median(errs[v])) for v in errs}
        hist.append(msg)
        print(f"época {ep} error apartado medio (px del cuadro) " + " ".join(f"{v} {msg[v]:.2f}" for v in msg)
              + " | mediana " + " ".join(f"{v} {med[v]:.2f}" for v in med) + f" | {time.time() - t0:.0f}s", flush=True)
        torch.save({"state_dict": model.state_dict(), "names": names, "epoch": ep, "val": msg, "vistas": lay.names}, out)
        np.save(str(out) + ".val_ultimo.npy", {v: np.array(errs[v]) for v in errs}, allow_pickle=True)
    print("guardado", out, f"FORJA pico_asignado_MiB {torch.cuda.max_memory_allocated() >> 20} pico_reservado_MiB "
          f"{torch.cuda.max_memory_reserved() >> 20} segundos {time.time() - t0:.0f}", flush=True)


if __name__ == "__main__":
    main()
