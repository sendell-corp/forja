"""Corre un script de Blender de anymove con la tercera cámara (arriba) añadida a su cuadro.

blender --factory-startup -b -P forja/vistas/blender/con_vistas.py -- ANYMOVE/blender/SCRIPT.py [sus argumentos…]

anymove (Apache-2.0) pone sus dos cámaras ortográficas en `blender/common.py::cameras`. Aquí se envuelve esa función
para que devuelva también `cam_top`: ortográfica, en (0, TOP_Y_CENTER, 20), mirando hacia abajo (−Z), con +X a la
derecha y +Y arriba de la imagen, a la misma escala que las otras (ortho_scale = M_PER_PX · max(W_TOP, H)).
Así `render_dataset.py` escribe también `NNNNN_top.png` y `apply_motion.py --views front,side,top` renderiza las tres,
sin tocar su código. W_TOP y TOP_Y_CENTER salen de la clave "forja" de creature.json (MOCAP_CREATURE).
"""
import pathlib
import runpy
import sys

args = sys.argv[sys.argv.index("--") + 1:]
target = pathlib.Path(args[0]).resolve()
sys.path.insert(0, str(target.parent))

import bpy  # noqa: E402
import common  # noqa: E402  (el de anymove: lee MOCAP_CREATURE al importarse)

_cfg = getattr(common, "_c", {}) or {}
_fj = _cfg.get("forja", {})
W_TOP = int(_fj.get("W_TOP", 640))
TOP_Y = float(_fj.get("TOP_Y_CENTER", 0.0))
_original = common.cameras


def cameras(scene):
    cams = _original(scene)
    cam = bpy.data.objects.get("cam_top")
    if cam is None:
        cam = bpy.data.objects.new("cam_top", bpy.data.cameras.new("cam_top"))
        scene.collection.objects.link(cam)
    cam.location, cam.rotation_euler = (0.0, TOP_Y, 20.0), (0.0, 0.0, 0.0)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = common.M_PER_PX * max(W_TOP, common.H)
    cam.data.clip_end = 100
    cams["cam_top"] = (cam, W_TOP)
    return cams


common.cameras = cameras
common.LAYOUT["W_TOP"], common.LAYOUT["TOP_Y_CENTER"] = W_TOP, TOP_Y
print(f"forja: cámara de arriba añadida (W_TOP={W_TOP}, TOP_Y_CENTER={TOP_Y}, M_PER_PX={common.M_PER_PX})", flush=True)
sys.argv = sys.argv[:sys.argv.index("--") + 1] + args[1:]
runpy.run_path(str(target), run_name="__main__")
