"""Comprueba un render de apply_motion.py contra Forja: las cámaras de Blender (cam_<vista>.json) proyectan igual que el
layout, y las articulaciones de Blender (joints.json) coinciden con la cinemática directa de forja/rig.py.

python -m forja.vistas.comprobar RENDER_DIR creature.json [rig_info.json motion.json]
"""
import json
import pathlib
import sys

import numpy as np

from forja.vistas.layout import Layout


def main():
    d = pathlib.Path(sys.argv[1])
    lay = Layout(sys.argv[2])
    J = np.array(json.load(open(d / "joints.json"))["joints"])
    worst = {}
    for v in lay.views:
        f = d / f"cam_{v.name}.json"
        if not f.exists():
            continue
        c = json.load(open(f))
        R, t = np.array(c["R"]), np.array(c["t"])
        pc = J @ R.T + t
        uv = np.stack([c["cx"] + c["f"] * pc[..., 0], c["cy"] - c["f"] * pc[..., 1]], -1)
        k = c["res"][1] / lay.H  # el render puede ir a otra escala
        worst[v.name] = float(np.abs(uv / k - v.project(J)).max())
    print("cámaras de Blender contra el layout de Forja (px, máx):", {k: round(x, 4) for k, x in worst.items()})
    if len(sys.argv) > 4:
        from forja.rig import Rig, load_motion
        rig = Rig(sys.argv[3])
        m = load_motion(sys.argv[4])
        F, _ = rig.forward(m["trans"], m["root_R"], m["euler"])
        print(f"articulaciones de Blender contra forja/rig.py: {np.abs(F - J).max() * 1000:.4f} mm máx")
    if any(x > 0.5 for x in worst.values()):
        raise SystemExit("una cámara no coincide con el layout")


if __name__ == "__main__":
    main()
