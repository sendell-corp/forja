"""El punto más bajo de la malla deformada en cada cuadro de un movimiento (penetración del suelo de verdad, con la
garra y no solo con las articulaciones).

blender --factory-startup -b -P forja/calidad/blender_suelo.py -- ANYMOVE/blender/apply_motion.py rigged.blend \
    rig_info.json motion.json SALIDA_DIR
Corre apply_motion.py de anymove (sin vistas: solo pone las claves y escribe joints.json) y, con la escena ya animada,
evalúa la malla `body` en cada cuadro. Escribe SALIDA_DIR/suelo.json: z mínima por cuadro (m) y el vértice.
"""
import json
import pathlib
import runpy
import sys

args = sys.argv[sys.argv.index("--") + 1:]
target = pathlib.Path(args[0]).resolve()
sys.path.insert(0, str(target.parent))
sys.argv = sys.argv[:sys.argv.index("--") + 1] + args[1:]
runpy.run_path(str(target), run_name="__main__")

import bpy  # noqa: E402
import numpy as np  # noqa: E402

scene = bpy.context.scene
body = bpy.data.objects["body"]
out = pathlib.Path(args[4])
zs, idx = [], []
for t in range(scene.frame_start, scene.frame_end + 1):
    scene.frame_set(t)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = body.evaluated_get(dg)
    me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    ev.to_mesh_clear()
    co = co.reshape(-1, 3) @ np.array(body.matrix_world)[:3, :3].T + np.array(body.matrix_world)[:3, 3]
    k = int(co[:, 2].argmin())
    zs.append(float(co[k, 2]))
    idx.append(k)
json.dump({"min_z": zs, "vertex": idx}, open(out / "suelo.json", "w"))
print(f"suelo: z mínima de la malla {min(zs) * 100:.2f} cm (cuadro {int(np.argmin(zs))})", flush=True)
