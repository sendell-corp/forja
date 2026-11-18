"""Gesto desde texto (F4 mínima): una frase → un clip esquelético en el rig del personaje, con la caja como cuerpo físico.

python -m forja.gesto "pick up box from ground" --personaje MODELO.glb --salida CARPETA [--nombre gesto]

CARPETA/
  <nombre>.glb               el personaje (malla, piel, rig) con un solo clip nuevo y la caja animada como nodo aparte
  <nombre>.anotaciones.json  eventos (pasos, agarre, suelta), apoyos de los pies, contacto de las manos
  informe.json               tiempos de cada paso, poses clave buscadas y medidas (deslizamiento, suelo, manos, caja)
  movimiento.npz             las matrices de mundo por cuadro (para la vista previa y las comprobaciones)

Pasos: frase → plantilla (frase.py) → metas por cuadro (gestos.py, con las poses clave buscadas) → IK de Forja con
límites (cuerpo.py) → Jolt (fisica_caja.mjs: la caja sigue a las manos como cuerpo físico; colisiones del cuerpo con
la caja y el suelo) → medida con la malla deformada → corrección (manos al ras de la caja, caja más lejos si algo la
atraviesa) → GLB.
"""
import argparse
import json
import pathlib
import subprocess
import tempfile
import time

import numpy as np

from forja.gesto.cuerpo import LADOS, Cuerpo
from forja.gesto.frase import interpretar
from forja.gesto.gestos import FPS, Plan
from forja.gesto.humanoide import C, C4, Humanoide
from forja.paquete import glb as G
from forja.rig import matrix_to_quat, quat_to_matrix

AQUI = pathlib.Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent
TOL_MANO = 0.005   # m: hueco o penetración de la palma con la cara de la caja que se corrige


def suave(x, sigma=1.5):
    """Suavizado gaussiano en el tiempo (eje 0), con bordes replicados."""
    k = np.arange(-int(3 * sigma), int(3 * sigma) + 1)
    w = np.exp(-0.5 * (k / sigma) ** 2)
    w /= w.sum()
    xp = np.concatenate([np.repeat(x[:1], len(k) // 2, 0), x, np.repeat(x[-1:], len(k) // 2, 0)])
    return sum(wi * xp[i:i + len(x)] for i, wi in enumerate(w))


def resolver_todo(cu, pl, ajuste_munecas=None):
    mun = pl["munecas"] if ajuste_munecas is None else {s: pl["munecas"][s] + ajuste_munecas[s] for s in LADOS}
    a = (pl["pel"], pl["lean"], pl["cabeza"], pl["tobillos"], mun, pl["dedos"], pl["rodillas"], pl["codos"])
    trans, root_R, eul, err = cu.resolver(*a)
    # segunda pasada: con la curva suavizada de la primera como previo (sin saltos de una solución a otra)
    trans, root_R, eul, err = cu.resolver(*a, prior=suave(eul), wp=4.0)
    eul = cu.orientar(trans, root_R, eul, pl["R_pies"], pl["R_manos"])
    return trans, root_R, eul, err


def pose7(p, R):
    q = matrix_to_quat(R)  # w, x, y, z
    return np.concatenate([p, q[..., [1, 2, 3, 0]]], -1)


def cascos(h, cu, max_puntos=300):
    """Por hueso, los vértices que domina (en reposo, Forja), submuestreados: Jolt hace el casco convexo."""
    dom = h.VJ[np.arange(len(h.V)), h.VW.argmax(1)]
    out = {}
    rng = np.random.default_rng(0)
    for k, n in enumerate(cu.rig.names):
        sk = h.skin_joints.index(h.rig_nodes[k])
        pts = h.V_forja[(dom == sk) & (h.VW.max(1) >= 0.5)]
        if len(pts) < 8:
            continue
        if len(pts) > max_puntos:
            pts = pts[rng.choice(len(pts), max_puntos, replace=False)]
        out[n] = pts.round(5).tolist()
    return out


def fisica(h, cu, pl, P, eventos):
    """Jolt: la caja y las colisiones del cuerpo. Devuelve las poses de la caja (T, 7) y lo que encontró."""
    r = cu.rig
    T = len(P)
    # el agarre: entre los puntos de contacto de las dos palmas, con X de la palma derecha a la izquierda
    pc = {}
    for s in LADOS:
        M = P[:, cu.ix[f"hand_{s}"]]
        pc[s] = M[:, :3, 3] + np.einsum("tij,j->ti", M[:, :3, :3], cu.palma[s]["punto"])
    x = pc["L"] - pc["R"]
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    z = np.broadcast_to([0, 0, 1.0], x.shape) - x[:, 2:3] * x
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    Rg = np.stack([x, np.cross(z, x), z], -1)
    mano = pose7((pc["L"] + pc["R"]) / 2, Rg)
    X = P @ np.linalg.inv(r.B)[None]
    cs = cascos(h, cu)
    cuadros = [{n: np.concatenate([X[t, cu.ix[n], :3, :3].ravel(), X[t, cu.ix[n], :3, 3]]).round(6).tolist() for n in cs}
               for t in range(T)]
    m = pl["caja_medidas"]
    c0 = pl["caja"][0]
    entrada = {"caja": {"medidas": [m["ancho"], m["fondo"], m["alto"]], "masa": m["masa_kg"],
                        "pose0": [*c0.tolist(), 0, 0, 0, 1]},
               "fps": FPS, "sub": 8, "agarre": [eventos["agarre"], eventos["suelta"]],
               "mano": mano.round(6).tolist(), "cascos": cs, "cuadros": cuadros}
    with tempfile.TemporaryDirectory() as d:
        fi, fo = pathlib.Path(d) / "in.json", pathlib.Path(d) / "out.json"
        fi.write_text(json.dumps(entrada))
        res = subprocess.run(["node", str(AQUI / "fisica_caja.mjs"), str(fi), str(fo)], cwd=RAIZ, capture_output=True,
                             text=True)
        if res.returncode:
            raise SystemExit("jolt falló:\n" + res.stderr[-2000:])
        out = json.loads(fo.read_text())
    out["log"] = res.stdout.strip()
    return np.array(out["caja"]), out


def medir(h, cu, pl, P, caja, eventos):
    """Con la malla deformada: suelo, palmas contra las caras de la caja, cuerpo dentro de la caja, pies plantados."""
    m = pl["caja_medidas"]
    half = np.array([m["ancho"], m["fondo"], m["alto"]]) / 2
    Wg = h.mundos_gltf(P)
    Vg = h.piel(Wg)                                 # glTF
    V = Vg @ C                                      # → Forja (p_f = Cᵀ p_g; fila: p_g · C)
    T = len(V)
    Rb = quat_to_matrix(caja[:, [6, 3, 4, 5]])
    loc = np.einsum("tji,tvj->tvi", Rb, V - caja[:, None, :3])  # vértices en el marco de la caja
    dom = h.VJ[np.arange(len(h.V)), h.VW.argmax(1)]
    nodo = {n: h.skin_joints.index(h.rig_nodes[k]) for k, n in enumerate(cu.rig.names)}
    out = {"suelo_min_cm": float(V[..., 2].min() * 100)}
    out["suelo_min_cm_por_cuadro"] = (V[..., 2].min(1) * 100).round(2).tolist()
    # palmas: distancia con signo de los vértices de la mano a su cara lateral (> 0 fuera de la caja)
    agarrada = np.zeros(T, bool)
    a, b = eventos["agarre"], eventos["suelta"] if eventos["suelta"] is not None else T
    agarrada[a:b] = True
    hueco = {}
    for s in LADOS:
        sel = dom == nodo[f"hand_{s}"]
        signo = 1 if s == "L" else -1     # la cara izquierda es +X de la caja (el personaje la mira de frente)
        xl = loc[:, sel]
        # solo los vértices a la altura y fondo de la cara cuentan como contacto
        dentro_cara = (np.abs(xl[..., 1]) < half[1]) & (np.abs(xl[..., 2]) < half[2])
        dist = signo * xl[..., 0] - half[0]
        dist = np.where(dentro_cara, dist, np.inf)
        hueco[s] = dist.min(1)
    out["mano_hueco_cm"] = {s: [float(np.nanmax(np.where(agarrada, hueco[s], -np.inf)) * 100),
                                float(np.nanmin(np.where(agarrada, hueco[s], np.inf)) * 100)] for s in LADOS}
    # el resto del cuerpo dentro de la caja (profundidad: lo que le falta para salir por la cara más cercana)
    manos = np.isin(dom, [nodo["hand_L"], nodo["hand_R"]])
    prof = np.min(half - np.abs(loc), -1)           # > 0: dentro
    prof = np.where(manos[None], -1, prof)
    out["cuerpo_en_caja_cm_max"] = float(max(0.0, prof.max()) * 100)
    peor = np.unravel_index(prof.argmax(), prof.shape)
    inv = {v: k for k, v in nodo.items()}
    out["cuerpo_en_caja_peor"] = {"cuadro": int(peor[0]), "hueso": inv.get(int(dom[peor[1]]), "?")}
    # caja contra el suelo
    esq = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
    zc = (np.einsum("tij,kj->tki", Rb, esq) + caja[:, None, :3])[..., 2]
    out["caja_bajo_suelo_cm"] = float(max(0.0, -zc.min()) * 100)
    # pies plantados: deslizamiento del tobillo y de la punta mientras el pie está apoyado
    J = cu.rig.joints_from(P)
    desl = []
    for s in LADOS:
        apoyo = ~pl["vuelo"][s]
        for jn in (f"foot_{s}", f"foot_{s}_tail"):
            p = J[:, cu.rig.kp_names.index(jn), :2]
            mov = np.linalg.norm(np.diff(p, axis=0), axis=-1)
            desl.append(mov[apoyo[1:] & apoyo[:-1]])
    desl = np.concatenate(desl)
    out["pies_deslizamiento_cm_cuadro_max"] = float(desl.max() * 100)
    out["pies_deslizamiento_cm_total_max"] = float(sum(desl) * 100) / 4
    return out, hueco, agarrada, Vg


def exportar(h, cu, pl, P, caja, nombre, ruta):
    """El GLB: copia del personaje con un solo clip (todos los huesos del rig) y la caja animada como nodo aparte."""
    js = json.loads(json.dumps(h.js))
    bin_ = bytearray(h.bin)
    js["animations"] = []
    # nombres de nodo/malla/material sin el nombre del personaje
    base = pathlib.Path(h.js["nodes"][h.mesh_node].get("name", "")).name.split("_")[0]
    for coleccion in ("nodes", "meshes", "materials", "skins", "images", "textures"):
        for k, o in enumerate(js.get(coleccion, [])):
            if base and base in str(o.get("name", "")):
                o["name"] = o["name"].replace(base, "personaje")
    Wg = h.mundos_gltf(P)
    pistas = {}
    for j, W in Wg.items():
        if j not in h.rig_nodes:
            continue
        p = h.par[j]
        Wp = Wg[p] if p in Wg else np.broadcast_to(h.W[p], W.shape)
        L = np.linalg.inv(Wp) @ W
        q = matrix_to_quat(L[:, :3, :3])
        nm = js["nodes"][j]["name"]
        pistas[nm] = {"rotation": q[:, [1, 2, 3, 0]]}
        if j == h.node_idx["pelvis"]:
            pistas[nm]["translation"] = L[:, :3, 3]
    # la caja: malla de 24 vértices y su animación (en glTF)
    m = pl["caja_medidas"]
    hx, hy, hz = m["ancho"] / 2, m["fondo"] / 2, m["alto"] / 2
    ext = np.abs(C @ np.array([hx, hy, hz]))  # medio tamaño en ejes de glTF
    pos, nor, idx = [], [], []
    for eje in range(3):
        for sg in (-1, 1):
            n = np.zeros(3)
            n[eje] = sg
            u, v = [e for e in range(3) if e != eje]
            base_i = len(pos)
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3)
                p[eje], p[u], p[v] = sg * ext[eje], a * ext[u], b * ext[v]
                pos.append(p), nor.append(n)
            tri = [0, 1, 2, 0, 2, 3]
            if np.dot(np.cross(np.eye(3)[u], np.eye(3)[v]), n) < 0:
                tri = [0, 2, 1, 0, 3, 2]
            idx += [base_i + t for t in tri]
    pa = G._accessor(js, bin_, np.array(pos), "VEC3", minmax=True)
    na = G._accessor(js, bin_, np.array(nor), "VEC3")
    ia_data = np.array(idx, np.uint16)
    while len(bin_) % 4:
        bin_.append(0)
    off = len(bin_)
    bin_ += ia_data.tobytes()
    js["bufferViews"].append({"buffer": 0, "byteOffset": off, "byteLength": ia_data.nbytes})
    js["accessors"].append({"bufferView": len(js["bufferViews"]) - 1, "componentType": 5123, "count": len(idx), "type": "SCALAR"})
    ia = len(js["accessors"]) - 1
    js.setdefault("materials", []).append({"name": "carton", "pbrMetallicRoughness": {
        "baseColorFactor": [0.62, 0.45, 0.27, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.85}})
    js["meshes"].append({"name": "caja", "primitives": [{"attributes": {"POSITION": pa, "NORMAL": na}, "indices": ia,
                                                         "material": len(js["materials"]) - 1}]})
    js["nodes"].append({"name": "caja", "mesh": len(js["meshes"]) - 1})
    js["scenes"][js.get("scene", 0)]["nodes"].append(len(js["nodes"]) - 1)
    Rb = quat_to_matrix(caja[:, [6, 3, 4, 5]])
    Rg = np.einsum("ij,tjk,lk->til", C, Rb, C)
    pistas["caja"] = {"translation": caja[:, :3] @ C.T, "rotation": matrix_to_quat(Rg)[:, [1, 2, 3, 0]]}
    G.anadir_animacion(js, bin_, nombre, FPS, pistas)
    G.escribir(js, bin_, ruta)
    return Wg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("frase")
    ap.add_argument("--personaje", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--nombre", default=None)
    a = ap.parse_args()
    reloj = {}
    t0 = time.perf_counter()
    marca = lambda k: reloj.__setitem__(k, round(time.perf_counter() - t0, 3))
    intencion = interpretar(a.frase)
    marca("frase")
    nombre = a.nombre or {"recoger": "recoger_caja", "dejar": "dejar_caja"}[intencion["gesto"]]
    out = pathlib.Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)
    h = Humanoide(a.personaje)
    cu = Cuerpo(h)
    marca("personaje")
    separacion, carga, historial = 0.08, 0.30, []
    for vuelta in range(4):
        plan = Plan(cu, intencion["gesto"], separacion=separacion, carga=carga)
        pl = plan.construir()
        ev = pl["eventos"]
        marca(f"metas_{vuelta}")
        trans, root_R, eul, err = resolver_todo(cu, pl)
        _, P = cu.rig.forward(trans, root_R, eul)
        marca(f"ik_{vuelta}")
        caja, fis = fisica(h, cu, pl, P, ev)
        marca(f"jolt_{vuelta}")
        med, hueco, agarrada, Vg = medir(h, cu, pl, P, caja, ev)
        # suelo con la malla: si la piel de la pierna baja (talón, tobillo) entra en el suelo, ese pie sube lo que falta
        if med["suelo_min_cm"] < -0.1:
            dom = h.VJ[np.arange(len(h.V)), h.VW.argmax(1)]
            z = Vg[..., 1]  # glTF: Y arriba
            for s in LADOS:
                sel = np.isin(dom, [h.skin_joints.index(h.node_idx[f"shin_{s}"]), h.skin_joints.index(h.node_idx[f"foot_{s}"])])
                pen = np.maximum(0.0, -z[:, sel].min(1))
                pen = np.maximum(suave(pen, 1.5), pen)
                pl["tobillos"][s] = pl["tobillos"][s] + pen[:, None] * np.array([0, 0, 1.0])
            trans, root_R, eul, err = resolver_todo(cu, pl)
            _, P = cu.rig.forward(trans, root_R, eul)
            caja, fis = fisica(h, cu, pl, P, ev)
            med, hueco, agarrada, Vg = medir(h, cu, pl, P, caja, ev)
            marca(f"suelo_{vuelta}")
        # corrección de las palmas: el hueco (o la penetración) medido con la malla, quitado de la meta de la muñeca
        for it in range(2):
            if max(abs(v) for s in LADOS for v in med["mano_hueco_cm"][s]) <= TOL_MANO * 100:
                break
            aj = {}
            for s in LADOS:
                lado = plan.l if s == "L" else -plan.l
                d = np.where(agarrada & np.isfinite(hueco[s]), hueco[s], 0.0)
                d = suave(d, 2.0)
                aj[s] = (pl.get("_aj", {}).get(s, 0) if it else 0) - d[:, None] * lado
            pl["_aj"] = aj
            trans, root_R, eul, err = resolver_todo(cu, pl, aj)
            _, P = cu.rig.forward(trans, root_R, eul)
            caja, fis = fisica(h, cu, pl, P, ev)
            med, hueco, agarrada, Vg = medir(h, cu, pl, P, caja, ev)
            marca(f"palmas_{vuelta}_{it}")
        med["jolt"] = fis["log"]
        med["jolt_caja_cuerpo"] = sorted({(x[1]) for x in fis["caja_cuerpo"]})
        med["jolt_caja_cuerpo_prof_max_cm"] = max([x[2] for x in fis["caja_cuerpo"] if not x[1].startswith("hand")] or [0]) * 100
        med["jolt_suelo_prof_max_cm"] = max([x[2] for x in fis["suelo"]] or [0]) * 100
        med["muneca_err_max_mm_agarrada"] = float(max(err[f"muneca_{s}"][agarrada].max() for s in LADOS) * 1000)
        historial.append({"separacion_m": separacion, "carga_m": carga, **{k: v for k, v in med.items() if k != "suelo_min_cm_por_cuadro"}})
        # si el cuerpo atraviesa la caja (rodillas, muslos), se aleja la caja lo que falta y se repite
        # (en brazos: la caja más adelante; en el suelo o subiendo: más lejos de los pies)
        if med["cuerpo_en_caja_cm_max"] > 0.5 and vuelta < 3:
            extra = med["cuerpo_en_caja_cm_max"] / 100 + 0.015
            if pl["en_brazos"][med["cuerpo_en_caja_peor"]["cuadro"]]:
                carga += extra
            else:
                separacion += extra
            continue
        break
    Wg = exportar(h, cu, pl, P, caja, nombre, out / f"{nombre}.glb")
    marca("glb")
    total = reloj["glb"]
    # anotaciones e informe
    J = cu.rig.joints_from(P)
    apoyos = {}
    for s in LADOS:
        st = ~pl["vuelo"][s]
        tramos, t = [], 0
        while t < len(st):
            if st[t]:
                e = t
                while e < len(st) and st[e]:
                    e += 1
                tramos.append([t, e])
                t = e
            else:
                t += 1
        apoyos[f"foot_{s}"] = tramos
    anot = {"clip": nombre, "fps": FPS, "cuadros": len(P), "frase": a.frase, "intencion": intencion,
            "eventos": {k: v for k, v in ev.items()}, "apoyos": apoyos,
            "manos_en_caja": [int(ev["agarre"]), int(ev["suelta"]) if ev["suelta"] is not None else len(P)],
            "ejes": "glTF: Y arriba, el personaje mira a +Z; metros"}
    json.dump(anot, open(out / f"{nombre}.anotaciones.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    informe = {"reloj_s": reloj, "frase_a_glb_s": total, "claves": {k: {kk: (vv.tolist() if hasattr(vv, "tolist") else vv)
               for kk, vv in v.items()} for k, v in pl["claves"].items()}, "vueltas": historial, "medidas": med,
               "personaje_err_bind": h.err_bind}
    json.dump(informe, open(out / "informe.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    np.savez_compressed(out / "movimiento.npz", P=P, caja=caja, t=pl["t"],
                        nombres=np.array(cu.rig.names), eventos=json.dumps(ev))
    print(f"{nombre}: {len(P)} cuadros ({len(P) / FPS:.1f} s) en {total:.1f} s de reloj → {out / (nombre + '.glb')}")
    print(json.dumps({k: v for k, v in med.items() if k not in ("suelo_min_cm_por_cuadro",)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
