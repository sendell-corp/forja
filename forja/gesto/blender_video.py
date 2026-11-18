"""Vista previa del gesto en Blender: el GLB tal cual lo cargaría un motor, con cámara quieta en tercera persona y,
si se pide, una toma en primera persona (cámara en los ojos, calculada fuera de Blender desde el hueso de la cabeza).

blender --factory-startup -b -P forja/gesto/blender_video.py -- GESTO.glb CAMARAS.json SALIDA_DIR [--res 1920x1080]
    [--samples N] [--motor eevee|cycles]

CAMARAS.json (forja/gesto/camaras.py): {"fps", "cuadros", "tercera": {"pos", "mira", "lente"},
"primera": {"desde", "hasta", "pos": [[x,y,z]…], "mira": [[x,y,z]…], "lente"}} en coordenadas de Blender.
Escribe SALIDA_DIR/tercera/f_0000.png… y SALIDA_DIR/primera/f_0000.png…, y comprobacion.json (la caja y la mano por cuadro).
"""
import json
import math
import pathlib
import sys

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
glb, cams_path, out = args[:3]
opts = args[3:]
opt = lambda k, d=None: opts[opts.index(k) + 1] if k in opts else d
out = pathlib.Path(out)
out.mkdir(parents=True, exist_ok=True)
cams = json.load(open(cams_path, encoding="utf-8"))
W, H = (int(x) for x in opt("--res", "1920x1080").split("x"))

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.fps = cams["fps"]  # antes de importar: el importador pasa los segundos de glTF a cuadros con este valor
bpy.ops.import_scene.gltf(filepath=glb)
arm = next(o for o in scene.objects if o.type == "ARMATURE")
caja = next(o for o in scene.objects if o.name.startswith("caja"))
acts = [a.name for a in bpy.data.actions]
print("forja: importado", glb, "armadura", arm.name, "acciones", acts, flush=True)
for o in (arm, caja):
    ad = o.animation_data
    print("forja: animación de", o.name, ad.action.name if ad and ad.action else None, flush=True)

# comprobación: la caja y la mano izquierda en algunos cuadros (coordenadas de Blender)
comp = {}
T = cams["cuadros"]
for f in (0, T // 2, T - 1):
    scene.frame_set(f)
    bpy.context.view_layer.update()
    pb = arm.pose.bones.get("hand_L")
    comp[f] = {"caja": list(caja.matrix_world.translation),
               "mano_L": list(arm.matrix_world @ pb.head) if pb else None}
json.dump(comp, open(out / "comprobacion.json", "w"))
print("forja: comprobación", comp, flush=True)

# escenario
bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, 0))
suelo = bpy.context.object
mat = bpy.data.materials.new("suelo")
mat.use_nodes = True
nt = mat.node_tree
bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
chk = nt.nodes.new("ShaderNodeTexChecker")
chk.inputs["Scale"].default_value = 80
chk.inputs["Color1"].default_value = (0.40, 0.40, 0.40, 1)
chk.inputs["Color2"].default_value = (0.30, 0.30, 0.31, 1)
nt.links.new(chk.outputs["Color"], bsdf.inputs["Base Color"])
bsdf.inputs["Roughness"].default_value = 0.9
suelo.data.materials.append(mat)
sol = bpy.data.objects.new("sol", bpy.data.lights.new("sol", "SUN"))
scene.collection.objects.link(sol)
sol.rotation_euler = (math.radians(45), 0, math.radians(30))
sol.data.energy = 3.2
sol.data.angle = math.radians(8)
world = bpy.data.worlds.new("mundo")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.57, 0.60, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.9
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
scene.collection.objects.link(cam)
scene.camera = cam
if opt("--motor", "eevee") == "cycles":
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "CUDA"
    scene.cycles.device = "GPU"
    scene.cycles.samples = int(opt("--samples", 16))
    scene.cycles.use_denoising = True
else:  # Eevee: ~10 veces más rápido que Cycles en la 3070 para una vista previa
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = int(opt("--samples", 32))
    sol.data.shadow_soft_size = 0.02
scene.render.resolution_x, scene.render.resolution_y = W, H
scene.render.fps = cams["fps"]
scene.render.image_settings.file_format = "PNG"
scene.view_settings.view_transform = "Standard"


def apuntar(pos, mira):
    cam.location = Vector(pos)
    cam.rotation_euler = (Vector(mira) - cam.location).to_track_quat("-Z", "Y").to_euler()


def render(carpeta, cuadros, pose):
    d = out / carpeta
    d.mkdir(exist_ok=True)
    for i, f in enumerate(cuadros):
        scene.frame_set(f)
        pose(i, f)
        bpy.context.view_layer.update()
        scene.render.filepath = str(d / f"f_{i:04d}.png")
        bpy.ops.render.render(write_still=True)
    print("forja: renderizado", carpeta, len(cuadros), "cuadros", flush=True)


c3 = cams["tercera"]
cam.data.lens = c3["lente"]
cam.data.clip_start = 0.05
apuntar(c3["pos"], c3["mira"])  # quieta: se pone una vez
render("tercera", range(T), lambda i, f: None)
if "primera" in cams:
    c1 = cams["primera"]
    cam.data.lens = c1["lente"]
    cam.data.clip_start = 0.02
    render("primera", range(c1["desde"], c1["hasta"]),
           lambda i, f: apuntar(c1["pos"][f], c1["mira"][f]))
