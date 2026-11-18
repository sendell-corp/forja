"""La vitrina: el GLB exportado, importado en Blender tal cual lo cargaría el juego, y cada clip renderizado con su root
motion, sobre un suelo con textura y con una cámara que sigue a la criatura. De paso comprueba el GLB.

blender --factory-startup -b -P forja/paquete/blender_vitrina.py -- FORJA.glb ANOTACIONES.json SALIDA_DIR
    [--clips forja_caminar,…] [--res 960x540] [--samples 16] [--vueltas 3]

- Comprobación: con el objeto en el origen, la cabeza de cada hueso en los cuadros 0 y T/2 de cada clip (coordenadas de
  Blender) → SALIDA_DIR/comprobacion.json, para compararla con forja/paquete/glb.py (fk_juego).
- Render: SALIDA_DIR/<clip>/f_0000.png… Los bucles se repiten `vueltas` veces encadenando su root motion.
"""
import json
import math
import pathlib
import sys

import bpy
from mathutils import Euler, Vector

args = sys.argv[sys.argv.index("--") + 1:]
glb, anot_path, out = args[:3]
opts = args[3:]
opt = lambda k, d=None: opts[opts.index(k) + 1] if k in opts else d
out = pathlib.Path(out)
out.mkdir(parents=True, exist_ok=True)
anot = json.load(open(anot_path, encoding="utf-8"))
fps = anot["fps"]
clips = opt("--clips", ",".join(anot["clips"])).split(",")
W, H = (int(x) for x in opt("--res", "960x540").split("x"))
VUELTAS = int(opt("--vueltas", 3))

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
bpy.ops.import_scene.gltf(filepath=glb)
arm = next(o for o in scene.objects if o.type == "ARMATURE")
top = arm
while top.parent is not None:
    top = top.parent
top.rotation_mode = "XYZ"  # el importador de glTF lo deja en cuaterniones: rotation_euler se ignoraría (giro perdido)
print("forja: importado", glb, "armadura", arm.name, "raíz", top.name, "acciones", [a.name for a in bpy.data.actions], flush=True)


def poner(nombre):
    act = next((a for a in bpy.data.actions if a.name == nombre or a.name.startswith(nombre + "_") or a.name.startswith(nombre + ".")), None)
    if act is None:
        raise SystemExit(f"no está la acción {nombre}: {[a.name for a in bpy.data.actions]}")
    arm.animation_data_create()
    arm.animation_data.action = act
    if hasattr(arm.animation_data, "action_slot") and getattr(act, "slots", None):
        arm.animation_data.action_slot = act.slots[0]
    return act


# 1. comprobación
comp = {}
for c in clips:
    poner(c)
    T = anot["clips"][c]["cuadros"]
    top.location, top.rotation_euler = (0, 0, 0), (0, 0, 0)
    comp[c] = {}
    for f in (0, T // 2):
        scene.frame_set(f + 1 if scene.frame_start == 1 else f)
        scene.frame_set(f)
        bpy.context.view_layer.update()
        comp[c][str(f)] = {pb.name: list(arm.matrix_world @ pb.head) for pb in arm.pose.bones}
json.dump(comp, open(out / "comprobacion.json", "w"))
print("forja: comprobación escrita", flush=True)

# 2. escenario: suelo a cuadros, sol, cielo
bpy.ops.mesh.primitive_plane_add(size=60, location=(0, 0, 0))
suelo = bpy.context.object
mat = bpy.data.materials.new("suelo")
mat.use_nodes = True
nt = mat.node_tree
bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
chk = nt.nodes.new("ShaderNodeTexChecker")
chk.inputs["Scale"].default_value = 60
chk.inputs["Color1"].default_value = (0.42, 0.40, 0.37, 1)
chk.inputs["Color2"].default_value = (0.30, 0.29, 0.27, 1)
nt.links.new(chk.outputs["Color"], bsdf.inputs["Base Color"])
bsdf.inputs["Roughness"].default_value = 0.85
suelo.data.materials.append(mat)
sol = bpy.data.objects.new("sol", bpy.data.lights.new("sol", "SUN"))
scene.collection.objects.link(sol)
sol.rotation_euler = (math.radians(50), 0, math.radians(-40))
sol.data.energy = 3.0
sol.data.angle = math.radians(6)
world = bpy.data.worlds.new("mundo")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.58, 0.62, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.8
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
scene.collection.objects.link(cam)
cam.data.lens = 32
scene.camera = cam
scene.render.engine = "CYCLES"
prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "CUDA"
prefs.get_devices()
for d in prefs.devices:
    d.use = d.type == "CUDA"
scene.cycles.device = "GPU"
scene.cycles.samples = int(opt("--samples", 16))
scene.cycles.use_denoising = True
scene.render.resolution_x, scene.render.resolution_y = W, H
scene.render.fps = fps
scene.render.image_settings.file_format = "PNG"
scene.view_settings.view_transform = "Standard"
OFF = Vector((2.5, -2.9, 1.35))  # cámara a 3/4 por delante y a su izquierda (la criatura mira a −Y en Blender)
# los bucles se siguen con la cámara (viajan); los gestos, con la cámara quieta, para que se vea su desplazamiento;
# el mordisco, más de frente y más alto, para que se vea la cabeza lanzarse por encima de las patas
OFF_CLIP = {"forja_mordisco": Vector((1.3, -3.0, 1.75))}


def pose_raiz(rm, f, n, T):
    """Root motion del cuadro f en la vuelta n: (x, y) de Blender y giro, encadenando vueltas."""
    x, z, yaw = rm["x"], rm["z"], rm["yaw"]
    dx, dz, dy = (x[-1] - x[-2], z[-1] - z[-2], yaw[-1] - yaw[-2]) if T > 1 else (0, 0, 0)
    ex, ez, ey = x[-1] + dx, z[-1] + dz, yaw[-1] + dy  # fin de una vuelta
    px, pz, py = 0.0, 0.0, 0.0
    for _ in range(n):
        c, s = math.cos(py), math.sin(py)
        px, pz = px + c * ex + s * ez, pz - s * ex + c * ez
        py += ey
    c, s = math.cos(py), math.sin(py)
    gx, gz = px + c * x[f] + s * z[f], pz - s * x[f] + c * z[f]
    return Vector((gx, -gz, 0.0)), py + yaw[f]


for c in clips:
    a = anot["clips"][c]
    T = a["cuadros"]
    poner(c)
    reps = VUELTAS if a["bucle"] else 1
    d = out / c
    d.mkdir(exist_ok=True)
    foco = None
    off = OFF_CLIP.get(c, OFF)
    fijo = None
    if not a["bucle"]:
        pts = [pose_raiz(a["root_motion"], f, 0, T)[0] for f in range(T)]
        fijo = sum(pts, Vector()) / len(pts) + Vector((0, 0, 0.45))
    for i in range(T * reps):
        f, n = i % T, i // T
        loc, yaw = pose_raiz(a["root_motion"], f, n, T)
        top.location = loc
        top.rotation_euler = Euler((0, 0, yaw))
        scene.frame_set(f)
        bpy.context.view_layer.update()
        objetivo = top.location + Vector((0, 0, 0.45))
        foco = fijo if fijo is not None else (objetivo if foco is None else foco.lerp(objetivo, 0.15))
        cam.location = foco + off
        cam.rotation_euler = (foco - cam.location).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(d / f"f_{i:04d}.png")
        bpy.ops.render.render(write_still=True)
    print("forja: renderizado", c, T * reps, "cuadros", flush=True)
