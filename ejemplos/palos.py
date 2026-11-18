"""Side view (stick figure) of a `forja.gesto` result, a few frames in one image.

python -m ejemplos.palos CARPETA_DEL_GESTO SALIDA.png [--cuadros 0,60,95,110,150,209]

Reads CARPETA/movimiento.npz (bone world matrices in Forja: Z up, the character faces −Y) and draws each bone from
its head to its child's head, plus the box centre and its size from informe.json. Needs matplotlib.
"""
import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PADRES = {"spine": "pelvis", "chest": "spine", "neck": "chest", "head": "neck"}
for s in "LR":
    PADRES |= {f"upper_arm_{s}": "chest", f"forearm_{s}": f"upper_arm_{s}", f"hand_{s}": f"forearm_{s}",
               f"thigh_{s}": "pelvis", f"shin_{s}": f"thigh_{s}", f"foot_{s}": f"shin_{s}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carpeta")
    ap.add_argument("salida")
    ap.add_argument("--cuadros", default="0,60,95,110,150,209")
    a = ap.parse_args()
    d = np.load(pathlib.Path(a.carpeta) / "movimiento.npz")
    P, caja, nombres = d["P"], d["caja"], [str(n) for n in d["nombres"]]
    inf = json.loads((pathlib.Path(a.carpeta) / "informe.json").read_text())
    med = inf.get("caja_medidas") or {}
    fondo, alto = med.get("fondo", 0.3), med.get("alto", 0.3)
    cuadros = [min(int(c), len(P) - 1) for c in a.cuadros.split(",")]
    fig, axs = plt.subplots(1, len(cuadros), figsize=(2.4 * len(cuadros), 3.6), sharey=True)
    for ax, c in zip(np.atleast_1d(axs), cuadros):
        pos = P[c, :, :3, 3]
        for hijo, padre in PADRES.items():
            if hijo in nombres and padre in nombres:
                p, h = pos[nombres.index(padre)], pos[nombres.index(hijo)]
                lado = "#2a6fdb" if hijo.endswith("_L") else "#e8743b" if hijo.endswith("_R") else "#222"
                ax.plot([-p[1], -h[1]], [p[2], h[2]], color=lado, lw=2.2, solid_capstyle="round")
        b = caja[c, :3]
        ax.add_patch(plt.Rectangle((-b[1] - fondo / 2, b[2] - alto / 2), fondo, alto, fc="#c9a26b", ec="#7a5a2e"))
        ax.axhline(0, color="#999", lw=1)
        ax.set_title(f"{c / 30:.1f} s", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(-0.6, 1.0)
        ax.set_ylim(-0.05, 1.9)
        ax.set_xticks([])
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
    fig.suptitle('"pick up box from ground" on the box mannequin (side view; blue = left, orange = right)', fontsize=10)
    fig.tight_layout()
    fig.savefig(a.salida, dpi=110)
    print("imagen:", a.salida)


if __name__ == "__main__":
    main()
