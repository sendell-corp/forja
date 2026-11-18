"""Limpieza física de un clip: suelo, partes que se atraviesan y equilibrio, con Jolt para las colisiones.

python -m forja.fisica.limpieza MOTION.json META.json rig_info.json creature.json SALIDA.json [--informe INFORME.json]

META.json es el truth_forja.json del generador (apoyos por pata, alturas de apoyo). La limpieza:
1. **Colisiones con Jolt** (jolt_colisiones.mjs): cada hueso es un casco convexo de su parte de la malla
   (modelo/cascos.py). Por cuadro, penetración contra el suelo y entre pares de partes que no deben tocarse (patas
   distintas, patas contra cuerpo y abdomen, colmillos contra patas). Los pares que ya se solapan en reposo no cuentan.
2. **Patas que se atraviesan:** la punta de la pata en vuelo (o la que menos apoyo tenga) se aparta del contacto en
   horizontal 2 × la profundidad + 1 cm (mover la punta mueve menos el hueso que choca), igual en todo el apoyo para que
   siga clavada y con un empuje suave en el tiempo en el vuelo, y se rehace su IK (con límites, suelo y garras).
3. **Suelo con los cascos:** si un casco de pata penetra el suelo, se añaden sus puntos a la restricción de suelo de
   esa pata (la de las garras ya está) y se rehace su IK.
4. **Equilibrio aproximado:** el centro de masas (masa de cada hueso ∝ sus vértices) frente al polígono de apoyo (las
   puntas apoyadas), en los cuadros lentos de los clips que no son bucle. Si cae a menos de 2 cm del borde o fuera, el
   cuerpo se desplaza en horizontal lo que falta, con una curva suave, y las patas apoyadas se rehacen clavadas en su
   sitio. En vuelo (menos de 3 apoyos) no se toca. Se mide además el punto de momento cero (ZMP = CdM − z/g·a).
Repite hasta 6 veces y mide antes y después.
"""
import argparse
import json
import pathlib
import subprocess
import tempfile

import numpy as np

from forja.rig import Rig, cargar_garras, load_motion, save_motion, matrix_to_quat
from forja.ajuste.ik import CadenaIK
from forja.ajuste.suavizado import suavizar

AQUI = pathlib.Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent


def cargar_cascos(cfg_path):
    """cascos.json → {pieza: (hueso, puntos (n, 3), vértices)}; pieza = «hueso#k»."""
    f = pathlib.Path(cfg_path).parent / "cascos.json"
    d = json.load(open(f, encoding="utf-8"))
    out = {}
    for h, piezas in d.items():
        for k, v in enumerate(piezas if isinstance(piezas, list) else [piezas]):
            out[f"{h}#{k}"] = (h, np.array(v["puntos"], np.float64), int(v["vertices"]))
    return out


def transformaciones(rig, m):
    """Reposo → pose de cada hueso (T, nb, 3, 4): P_b · B_b⁻¹."""
    _, P = rig.forward(m["trans"], m["root_R"], m["euler"])
    X = P @ np.linalg.inv(rig.B)[None]
    return X[..., :3, :]


def jolt(rig, m, cascos, pares, cuadros=None, crudo=False, separacion=0.0):
    X = transformaciones(rig, m)
    idx = {n: i for i, n in enumerate(rig.names)}
    T = len(X)
    sel = range(T) if cuadros is None else cuadros
    entrada = {"cascos": {p: c[1].tolist() for p, c in cascos.items()}, "pares": pares, "separacion": separacion,
               "cuadros": [{p: np.concatenate([X[t, idx[c[0]], :, :3].ravel(), X[t, idx[c[0]], :, 3]]).round(7).tolist()
                            for p, c in cascos.items()} for t in sel]}
    with tempfile.TemporaryDirectory() as d:
        fi, fo = pathlib.Path(d) / "e.json", pathlib.Path(d) / "s.json"
        fi.write_text(json.dumps(entrada))
        subprocess.run(["node", str(AQUI / "jolt_colisiones.mjs"), str(fi), str(fo)], check=True, cwd=RAIZ, capture_output=True)
        r = json.loads(fo.read_text())
    if crudo:
        return r
    # de piezas a huesos (la penetración de un par de huesos es la mayor de sus piezas)
    hb = lambda p: cascos[p][0]
    r["suelo"] = [[t, hb(p), d, x] for t, p, d, x in r["suelo"]]
    r["pares"] = [[t, *sorted((hb(a), hb(b))), d, ax, x] for t, a, b, d, ax, x in r["pares"]]
    return r


def cadena_de(rig):
    out = {}
    for k, (bones, _) in enumerate(rig.chains()):
        for b in bones:
            out[rig.names[b]] = k
    return out


def pares_por_defecto(rig, cascos):
    """Piezas que no deben tocarse: de huesos de cadenas distintas (no la raíz con el primer hueso de cada cadena)."""
    ch = cadena_de(rig)
    hs = [h for h in cascos]
    bone = lambda p: cascos[p][0]
    pares = []
    for i, a in enumerate(hs):
        for b in hs[i + 1:]:
            ba, bb = bone(a), bone(b)
            if ba == bb:
                continue
            ca, cb = ch.get(ba), ch.get(bb)
            if ca is not None and cb is not None and ca == cb:
                continue
            raiz = rig.names[0]
            primeros = {rig.names[bones[0]] for bones, _ in rig.chains()}
            if raiz in (ba, bb) and (ba in primeros or bb in primeros):
                continue
            pares.append([a, b])
    return pares


def pares_limpios(rig, cascos, vecinos=0.015):
    """Quita los pares que en reposo ya se tocan o están a menos de `vecinos` (rozan por diseño: el colmillo con la
    cabeza, el fémur con el cefalotórax)."""
    pares = pares_por_defecto(rig, cascos)
    nb = len(rig.names)
    reposo = {"trans": np.zeros((1, 3)), "root_R": np.eye(3)[None], "euler": np.zeros((1, nb, 3))}
    r = jolt(rig, reposo, cascos, pares, crudo=True, separacion=vecinos)
    malos = {tuple(sorted((a, b))) for _, a, b, d, _, _ in r["pares"]}
    return [p for p in pares if tuple(sorted(p)) not in malos], len(malos)


def centro_de_masas(rig, m, cascos):
    X = transformaciones(rig, m)
    idx = {n: i for i, n in enumerate(rig.names)}
    tot, acc = 0.0, 0.0
    for p, (h, pts, n) in cascos.items():
        c = pts.mean(0)
        acc = acc + n * (np.einsum("tij,j->ti", X[:, idx[h], :, :3], c) + X[:, idx[h], :, 3])
        tot += n
    return acc / tot


def _hull2d(P):
    P = sorted(map(tuple, P))
    if len(P) < 3:
        return np.array(P)
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in P:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(P):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return np.array(lo[:-1] + hi[:-1])


def zmp_de(com, fps=24.0, g=9.81, bucle=False):
    """Punto de momento cero aproximado (modelo de péndulo): ZMP = CdM_xy − z/g · a_xy, con a suavizada."""
    if False:  # (en los bucles la raíz avanza: la diferencia circular saltaría en la costura)
        pass
    else:
        a = np.zeros_like(com)
        a[1:-1] = (com[2:] - 2 * com[1:-1] + com[:-2]) * fps ** 2
        a[0], a[-1] = a[1], a[-2]
    a = suavizar(a, None, 1.0)
    return com[:, :2] - com[:, 2:3] / g * a[:, :2]


def margen_apoyo(com_xy, puntas_xy):
    """Distancia con signo del centro de masas al borde del polígono de apoyo (+ dentro) y el punto más cercano."""
    H = _hull2d(puntas_xy)
    if len(H) < 3:
        return None, None
    best, punto, dentro = np.inf, None, True
    for i in range(len(H)):
        a, b = H[i], H[(i + 1) % len(H)]
        ab = b - a
        u = np.clip(np.dot(com_xy - a, ab) / np.dot(ab, ab), 0, 1)
        q = a + u * ab
        d = np.linalg.norm(com_xy - q)
        if (ab[0] * (com_xy[1] - a[1]) - ab[1] * (com_xy[0] - a[0])) < 0:
            dentro = False
        if d < best:
            best, punto = d, q
    return (best if dentro else -best), punto


def diagnostico(rig, m, meta, cascos, pares):
    names = rig.kp_names
    tips = [names.index(n) for n in meta["tips"]]
    J, _ = rig.forward(m["trans"], m["root_R"], m["euler"])
    st = np.array(meta["stance"], bool)
    r = jolt(rig, m, cascos, pares)
    com = centro_de_masas(rig, m, cascos)
    zmp = zmp_de(com, bucle=bool(meta.get("bucle")))
    marg, margz = [], []
    for t in range(len(J)):
        on = np.where(st[t])[0]
        if len(on) >= 3:
            P = J[t, [tips[k] for k in on], :2]
            marg.append(margen_apoyo(com[t, :2], P)[0])
            margz.append(margen_apoyo(zmp[t], P)[0])
        else:
            marg.append(np.nan)
            margz.append(np.nan)
    marg, margz = np.array(marg, float), np.array(margz, float)
    suelo = max([d for _, _, d, _ in r["suelo"]], default=0.0)
    entre = max([x[3] for x in r["pares"]], default=0.0)
    return {"suelo_max_cm": suelo * 100, "suelo_cuadros": len({x[0] for x in r["suelo"] if x[2] > 0.002}),
            "partes_max_cm": entre * 100, "partes_cuadros": len({x[0] for x in r["pares"] if x[3] > 0.002}),
            "pares_tocando": sorted({f"{x[1]}–{x[2]}" for x in r["pares"] if x[3] > 0.002}),
            "equilibrio_margen_min_cm": float(np.nanmin(marg) * 100) if np.isfinite(marg).any() else None,
            "equilibrio_cuadros_fuera": int(np.nansum(marg < 0)), "equilibrio_cuadros_justos": int(np.nansum(marg < 0.02)),
            "zmp_margen_min_cm": float(np.nanmin(margz) * 100) if np.isfinite(margz).any() else None,
            "zmp_cuadros_fuera": int(np.nansum(margz < 0)),
            "_jolt": r, "_margen": marg, "_com": com, "_zmp": zmp}


def main():
    from forja.fuentes.procedural import Arana
    ap = argparse.ArgumentParser()
    ap.add_argument("motion"), ap.add_argument("meta"), ap.add_argument("info"), ap.add_argument("cfg")
    ap.add_argument("salida", nargs="?")
    ap.add_argument("--informe")
    a = ap.parse_args()
    rig = Rig(a.info)
    cfg = json.load(open(a.cfg, encoding="utf-8"))
    cascos = cargar_cascos(a.cfg)
    pares, fuera = pares_limpios(rig, cascos)
    m = load_motion(a.motion)
    meta = json.load(open(a.meta, encoding="utf-8"))
    if not a.salida:
        d = diagnostico(rig, m, meta, cascos, pares)
        print(f"{len(pares)} pares ({fuera} quitados por rozarse en reposo):",
              {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items() if not k.startswith("_")})
        return
    ar = Arana(rig, cfg, cfg.get("layout", {}).get("M_PER_PX", 0.0035), cargar_garras(a.cfg, rig))
    m2, inf = limpiar(rig, ar, m, meta, cascos, pares)
    save_motion(a.salida, rig.names, m2["trans"], m2["root_R"], m2["euler"], m.get("fps", 24))
    print("antes:", inf["antes"]); print("después:", inf["despues"]); print("desvío de puntas", round(inf["puntas_desvio_max_cm"], 2), "cm")
    if a.informe:
        json.dump(inf, open(a.informe, "w", encoding="utf-8"), indent=1, ensure_ascii=False)



# ------------------------------------------------------------------------------------------------ corrección

def _resolver(rig, ar, m, tips_tgt, prior_eul, legs=None):
    """Rehace la IK de las patas (todas o `legs`) hacia sus puntas objetivo, con la postura previa como previo."""
    T = len(m["trans"])
    Mroot = rig.root_matrix(m["trans"], m["root_R"])
    eul = m["euler"].copy()
    for i, (bones, joints) in enumerate(ar.legs):
        if legs is not None and i not in legs:
            continue
        ik = CadenaIK(rig, bones, ar.M, ar.garras.get(rig.kp_names[joints[-1]]))
        n = len(bones)
        tgt = np.zeros((T, n, 3))
        tgt[:, -1] = tips_tgt[:, i]
        Lam = np.zeros((T, n, 3, 3))
        Lam[:, -1] = np.eye(3) * 100.0 / ar.M ** 2
        prior = prior_eul[:, bones].reshape(T, -1)
        zmin = np.full(n, ar.zmin_otros)
        zmin[-1] = ar.h0[i] - 0.0005
        th, _, _ = ik.resolver(Mroot, tgt, Lam, prior[None], prior=prior, wp=50.0, wr=0.5, zmin=zmin, wg=400.0,
                               iters=40, sigmas=(1e6,))
        eul[:, bones] = th.reshape(T, n, 3)
    return dict(m, euler=eul)


def _tramos(mask):
    out, t = [], 0
    while t < len(mask):
        e = t
        while e < len(mask) and mask[e] == mask[t]:
            e += 1
        out.append((bool(mask[t]), t, e))
        t = e
    return out


def _suav(x, lam, bucle):
    """Suavizado (cuadrático) de señales (T, …); con bucle, circular (se repite el clip a los dos lados)."""
    if not bucle:
        return suavizar(x, None, lam)
    T = len(x)
    return suavizar(np.concatenate([x, x, x]), None, lam)[T:2 * T]


def _tramos_apoyo(st_k, bucle):
    """Tramos de apoyo [(a, b)] de una pata; con bucle, el que cruza la costura va como uno solo (índices módulo T)."""
    tr = [(a, b) for v, a, b in _tramos(st_k) if v]
    if bucle and len(tr) > 1 and tr[0][0] == 0 and tr[-1][1] == len(st_k):
        tr = [(tr[-1][0], tr[0][1] + len(st_k))] + tr[1:-1]
    return tr


def limpiar(rig, ar, m, meta, cascos, pares, iters=6, margen=0.02, ganancia=2.0, acel_lenta=1.0, log=print):
    """Corrige partes que se atraviesan y el equilibrio. Devuelve el movimiento limpio y el informe antes/después.
    Con meta["bucle"], todo es circular (la costura del ciclo sigue cerrando) y el equilibrio no se toca (la marcha en
    tetrápodos es estable; se mide). El equilibrio estático (centro de masas sobre el apoyo) solo se corrige en cuadros
    lentos; el punto de momento cero (ZMP) se mide, no se usa para corregir: realimentaba la corrección y la hacía
    estallar en el trote."""
    bucle = bool(meta.get("bucle"))
    fps = 24.0
    names = rig.kp_names
    tips = [names.index(n) for n in meta["tips"]]
    st = np.array(meta["stance"], bool)
    T = len(m["trans"])
    hip = [rig.kp_names.index(rig.head_joint[bones[0]]) for bones, _ in ar.legs]
    leg_de = {rig.names[b]: i for i, (bones, _) in enumerate(ar.legs) for b in bones}
    antes = diagnostico(rig, m, meta, cascos, pares)
    informe = {"antes": {k: v for k, v in antes.items() if not k.startswith("_")}}
    d = antes
    J0, _ = rig.forward(m["trans"], m["root_R"], m["euler"])
    tgt = J0[:, tips].copy()  # objetivos persistentes: cada vuelta les suma sus empujes (sin heredar errores de la IK)
    for it in range(iters):
        J, _ = rig.forward(m["trans"], m["root_R"], m["euler"])
        cambio = False
        # 1. equilibrio: el cuerpo hacia el polígono de apoyo (curva suave), las apoyadas quietas en el mundo
        marg, com = d["_margen"], d["_com"]
        # cuadros lentos: aceleración del centro de masas suavizado (las patas que ondean no cuentan) < 1 m/s²
        cs = suavizar(com, None, 50.0)
        acc = np.zeros(T)
        acc[1:-1] = np.linalg.norm(cs[2:] - 2 * cs[1:-1] + cs[:-2], axis=-1) * fps ** 2
        lento = acc < acel_lenta
        falta = np.zeros((T, 2))
        for t in range(T):
            if not bucle and lento[t] and np.isfinite(marg[t]) and marg[t] < margen:
                on = np.where(st[t])[0]
                H = _hull2d(J[t, [tips[k] for k in on], :2])
                c = H.mean(0)
                v = c - com[t, :2]
                falta[t] = v / max(np.linalg.norm(v), 1e-9) * (margen - marg[t])
        if np.abs(falta).max() > 1e-4:
            sh = _suav(falta, 20.0, bucle)
            m = dict(m, trans=m["trans"] + np.concatenate([sh, np.zeros((T, 1))], 1))
            for k in range(len(tips)):  # las que vuelan van con el cuerpo
                tgt[~st[:, k], k, :2] += sh[~st[:, k]]
            cambio = True
            log(f"  equilibrio: cuerpo desplazado hasta {np.linalg.norm(sh, axis=1).max() * 100:.1f} cm")
        # 2. partes que se atraviesan: apartar la punta de la pata que choca
        empuje = np.zeros((T, len(tips), 2))
        for t, a, b, dep, ax, p in d["_jolt"]["pares"]:
            if dep < 0.003:
                continue
            la, lb = leg_de.get(a), leg_de.get(b)
            if la is None and lb is None:
                continue
            if la is not None and lb is not None:  # entre patas: mueve la que vuela, o la de b
                k, otra = (la, lb) if not st[t, la] else (lb, la)
                dirv = J[t, tips[k], :2] - J[t, tips[otra], :2]
            else:
                k = la if la is not None else lb
                dirv = J[t, tips[k], :2] - J[t, hip[k], :2]  # hacia fuera de la cadera
            dirv = dirv / max(np.linalg.norm(dirv), 1e-9)
            v = dirv * (ganancia * dep + 0.01)  # la punta mueve el hueso que choca menos de lo que ella se mueve
            if np.linalg.norm(v) > np.linalg.norm(empuje[t, k]):
                empuje[t, k] = v
        if np.abs(empuje).max() > 0:
            for k in range(len(tips)):
                e = empuje[:, k].copy()
                if not np.abs(e).any():
                    continue
                const = {}
                for a0, a1 in _tramos_apoyo(st[:, k], bucle):  # todo el apoyo se mueve igual: sigue clavada
                    idx = np.arange(a0, a1) % T
                    big = e[idx][np.argmax(np.linalg.norm(e[idx], axis=1))]
                    e[idx] = big
                    const[(a0, a1)] = (idx, big)
                e = _suav(e, 2.0, bucle)  # el vuelo pasa suave de un apoyo al siguiente
                for idx, big in const.values():
                    e[idx] = big
                tgt[:, k, :2] += e
            cambio = True
            log(f"  partes: puntas apartadas hasta {np.linalg.norm(empuje, axis=-1).max() * 100:.1f} cm")
        if not cambio:
            break
        m = _resolver(rig, ar, m, tgt, m["euler"])
        d = diagnostico(rig, m, meta, cascos, pares)
        log(f"  vuelta {it + 1}: partes {d['partes_max_cm']:.1f} cm en {d['partes_cuadros']} cuadros; "
            f"equilibrio margen mín {d['equilibrio_margen_min_cm']}; suelo {d['suelo_max_cm']:.2f} cm")
    J, _ = rig.forward(m["trans"], m["root_R"], m["euler"])
    informe["despues"] = {k: v for k, v in d.items() if not k.startswith("_")}
    informe["puntas_desvio_max_cm"] = float(np.linalg.norm(J[:, tips] - tgt, axis=-1).max() * 100)
    return m, informe


if __name__ == "__main__":
    main()
