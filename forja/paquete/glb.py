"""El GLB para el juego: el modelo original de la criatura con clips nuevos, en sus propios nombres de hueso.

python -m forja.paquete.glb MODELO.glb rig_info.json forja_map.json SALIDA.glb clip1.json:nombre1 [clip2.json:nombre2 …]
    [--en-sitio]

Los clips son motion.json del rig de anymove/Forja (trans, root_quat, euler). Se añaden como animaciones nuevas a una
copia del GLB del juego (se conservan su malla, su piel y sus animaciones), así el juego lo carga igual que hoy y
`getObjectByName('body')` sigue encontrando el hueso.

**El conversor de vuelta** (lo que faltaba desde F1, §3.2 de F0-F1):
- ejes: p_glTF = C·(p_anymove − desplazamiento), con C: (x, y, z) → (x, z, −y) y el desplazamiento de prep_mesh.py
  (forja_map.json);
- rotación: en el juego los huesos no tienen rotación de reposo, y en anymove cada hueso tiene la suya (B, con su
  roll). La rotación de mundo del hueso del juego es G = C·(R_anim·R_reposo⁻¹)·Cᵀ y la local sale con el padre:
  q_local = G_padre⁻¹·G;
- traslación: solo `body` (la raíz libre) se mueve; el resto conserva las traslaciones de reposo del juego (las
  longitudes son las mismas). `root` queda quieto.
Con --en-sitio, el avance horizontal y el giro de la raíz se quitan del clip (van al root motion de las anotaciones).
"""
import argparse
import json
import pathlib
import struct

import numpy as np

from forja.rig import Rig, load_motion, matrix_to_quat, quat_to_matrix

C = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])  # anymove → glTF


def leer(path):
    b = pathlib.Path(path).read_bytes()
    assert b[:4] == b"glTF"
    off, js, bin_ = 12, None, b""
    while off < len(b):
        n, typ = struct.unpack("<II", b[off:off + 8])
        chunk = b[off + 8:off + 8 + n]
        if typ == 0x4E4F534A:
            js = json.loads(chunk)
        elif typ == 0x004E4942:
            bin_ = bytes(chunk)
        off += 8 + n
    return js, bytearray(bin_)


def escribir(js, bin_, path):
    j = json.dumps(js, separators=(",", ":")).encode()
    j += b" " * (-len(j) % 4)
    bin_ = bytes(bin_) + b"\0" * (-len(bin_) % 4)
    js["buffers"][0]["byteLength"] = len(bin_)
    j = json.dumps(js, separators=(",", ":")).encode()
    j += b" " * (-len(j) % 4)
    total = 12 + 8 + len(j) + 8 + len(bin_)
    out = struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(j), 0x4E4F534A) + j + \
        struct.pack("<II", len(bin_), 0x004E4942) + bin_
    pathlib.Path(path).write_bytes(out)


def _accessor(js, bin_, arr, typ, minmax=False):
    arr = np.ascontiguousarray(arr, np.float32)
    while len(bin_) % 4:
        bin_.append(0)
    off = len(bin_)
    bin_ += arr.tobytes()
    js["bufferViews"].append({"buffer": 0, "byteOffset": off, "byteLength": arr.nbytes})
    acc = {"bufferView": len(js["bufferViews"]) - 1, "componentType": 5126, "count": int(arr.shape[0]), "type": typ}
    if minmax:
        acc["min"] = arr.min(0).reshape(-1).tolist()
        acc["max"] = arr.max(0).reshape(-1).tolist()
    js["accessors"].append(acc)
    return len(js["accessors"]) - 1


def anadir_animacion(js, bin_, nombre, fps, pistas):
    """pistas: {nombre de nodo: {"rotation": (T, 4) xyzw, "translation": (T, 3)}} → una animación nueva (LINEAR)."""
    nodes = {n.get("name"): i for i, n in enumerate(js["nodes"])}
    T = len(next(iter(next(iter(pistas.values())).values())))
    t = _accessor(js, bin_, (np.arange(T) / fps)[:, None], "SCALAR", minmax=True)
    anim = {"name": nombre, "samplers": [], "channels": []}
    for node, ch in pistas.items():
        for path, val in ch.items():
            a = _accessor(js, bin_, val, "VEC4" if path == "rotation" else "VEC3")
            anim["samplers"].append({"input": t, "output": a, "interpolation": "LINEAR"})
            anim["channels"].append({"sampler": len(anim["samplers"]) - 1, "target": {"node": nodes[node], "path": path}})
    js.setdefault("animations", [])
    js["animations"] = [a for a in js["animations"] if a.get("name") != nombre] + [anim]


def _padres(js):
    p = {}
    for i, n in enumerate(js["nodes"]):
        for c in n.get("children", []):
            p[c] = i
    return p


def convertir(rig, motion, mapa, js, en_sitio=False):
    """motion (load_motion) → pistas glTF para los nodos del juego, y el root motion quitado (si en_sitio)."""
    off = np.array(mapa["offset_m"], np.float64)
    _, P = rig.forward(motion["trans"], motion["root_R"], motion["euler"])  # (T, nb, 4, 4)
    T = len(P)
    names = {n.get("name"): i for i, n in enumerate(js["nodes"])}
    par = _padres(js)
    R0 = rig.B[:, :3, :3]
    G = {}
    for b, bn in enumerate(rig.names):
        G[bn] = np.einsum("ij,tjk,lk,ml->tim", C, P[:, b, :3, :3], R0[b], C)  # C·(R·R0ᵀ)·Cᵀ
    root_motion = None
    body = rig.names[0]
    pos_body = (P[:, 0, :3, 3] - off) @ C.T  # glTF, mundo
    if en_sitio:
        # quitar el avance horizontal (x, z de glTF) y el giro sobre la vertical del cuerpo; quedan como root motion
        fwd = np.einsum("tij,j->ti", G[body], np.array([0, 0, 1.0]))
        yaw = np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 2]))
        root_motion = {"x": pos_body[:, 0] - pos_body[0, 0], "z": pos_body[:, 2] - pos_body[0, 2], "yaw": yaw - yaw[0]}
        Ry = np.zeros((T, 3, 3))
        c, s = np.cos(-root_motion["yaw"]), np.sin(-root_motion["yaw"])
        Ry[:, 0, 0], Ry[:, 0, 2], Ry[:, 1, 1], Ry[:, 2, 0], Ry[:, 2, 2] = c, s, 1, -s, c
        for bn in G:
            G[bn] = Ry @ G[bn]
        pos_body = pos_body.copy()
        pos_body[:, 0] -= root_motion["x"]
        pos_body[:, 2] -= root_motion["z"]
    pistas = {}
    for bn, Gb in G.items():
        if bn not in names:
            continue
        pi = par.get(names[bn])
        pn = js["nodes"][pi].get("name") if pi is not None else None
        Gp = G.get(pn) if pn in G else np.broadcast_to(np.eye(3), (T, 3, 3))
        Rl = np.einsum("tji,tjk->tik", Gp, Gb)
        q = matrix_to_quat(Rl)  # w, x, y, z
        pistas[bn] = {"rotation": q[:, [1, 2, 3, 0]]}
    # body: traslación local respecto a su padre (root, sin rotación)
    root_t = np.array(js["nodes"][par[names[body]]].get("translation", [0, 0, 0]), np.float64)
    pistas[body]["translation"] = pos_body - root_t
    return pistas, root_motion


def fk_juego(js, pistas, t):
    """Cinemática directa de los nodos del juego en el cuadro t con las pistas (para comprobar el conversor)."""
    par = _padres(js)
    W = {}

    def mundo(i):
        if i in W:
            return W[i]
        n = js["nodes"][i]
        nm = n.get("name")
        tr = np.array(n.get("translation", [0, 0, 0]), np.float64)
        R = np.eye(3)
        if nm in pistas:
            q = pistas[nm]["rotation"][t]
            R = quat_to_matrix(np.array([q[3], q[0], q[1], q[2]]))
            if "translation" in pistas[nm]:
                tr = pistas[nm]["translation"][t]
        if i in par:
            Rp, pp = mundo(par[i])
            W[i] = (Rp @ R, pp + Rp @ tr)
        else:
            W[i] = (R, tr)
        return W[i]

    return {n.get("name"): mundo(i)[1] for i, n in enumerate(js["nodes"])}


def comprobar(rig, motion, mapa, js, pistas):
    """Error máximo (mm) entre las articulaciones del rig (pasadas a glTF) y los nodos del juego animados."""
    off = np.array(mapa["offset_m"], np.float64)
    J, _ = rig.forward(motion["trans"], motion["root_R"], motion["euler"])
    worst = 0.0
    for t in range(0, len(J), max(1, len(J) // 12)):
        pos = fk_juego(js, pistas, t)
        for j, n in enumerate(rig.kp_names):
            if n in pos:
                worst = max(worst, float(np.linalg.norm(pos[n] - C @ (J[t, j] - off))))
    return worst * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modelo"), ap.add_argument("info"), ap.add_argument("mapa"), ap.add_argument("out")
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--en-sitio", action="store_true")
    a = ap.parse_args()
    rig = Rig(a.info)
    mapa = json.load(open(a.mapa, encoding="utf-8"))
    js, bin_ = leer(a.modelo)
    rm = {}
    for c in a.clips:
        path, nombre = c.rsplit(":", 1)
        m = load_motion(path)
        pistas, root_motion = convertir(rig, m, mapa, js, a.en_sitio)
        if not a.en_sitio:
            print(f"{nombre}: conversor, error máximo {comprobar(rig, m, mapa, js, pistas):.4f} mm")
        anadir_animacion(js, bin_, nombre, m["fps"], pistas)
        if root_motion:
            rm[nombre] = {k: np.round(v, 6).tolist() for k, v in root_motion.items()}
    escribir(js, bin_, a.out)
    if rm:
        json.dump(rm, open(pathlib.Path(a.out).with_suffix(".root_motion.json"), "w"))
    print("glb", a.out, [x.get("name") for x in js["animations"]])


if __name__ == "__main__":
    main()
