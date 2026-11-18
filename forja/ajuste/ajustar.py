"""El ajuste de Forja: de las lecturas de N vistas a un movimiento del rig, limpio y dentro de sus límites.

python -m forja.ajuste.ajustar DET.npz rig_info.json creature.json SALIDA.json [--vistas front,side] [--sin-suelo]

DET.npz es la salida de forja.lector.leer (N vistas) o de mocap/detect.py de anymove (frente y lado).
SALIDA.json es un motion.json de anymove (apply_motion.py lo pone en el rig de Blender y saca el GLB) con, además, las
articulaciones ajustadas, los apoyos por pata y el diagnóstico.

Pasos:
1. **Triangulación robusta** (triangulacion/robusta.py): por articulación y cuadro, qué lectura usar en cada vista
   (propia, de su gemela o ninguna), elegido por Viterbi con coherencia temporal.
2. **Raíz** (raiz.py): Kabsch robusto con los puntos rígidos del cuerpo, refinado con la información de cada punto.
3. **IK por cadena** (ik.py): cada pata, colmillo y el abdomen, con longitudes fijas y límites articulares.
   Pasada 1 con dos arranques (geométrico y reposo); pasada 2 con un previo hacia sus vecinos en el tiempo, que
   elige la rama coherente cuando los datos no deciden (p. ej. un eje que solo veía una vista que se descartó).
4. **Suavizado** de los ángulos con Huber sobre la aceleración (suavizado.py): quita el temblor, conserva los golpes.
5. **Suelo y apoyos** (contactos.py): ninguna articulación bajo el suelo (las puntas, bajo su altura de reposo); por
   pata, apoyo/vuelo por Viterbi; en cada apoyo la punta queda clavada en un punto del suelo y la IK se rehace.
"""
import argparse
import json

import numpy as np

from forja.ajuste.contactos import apoyos, plantar, runs
from forja.ajuste.ik import CadenaIK, obs_puntos, obs_unir, obs_vistas
from forja.ajuste.raiz import ajustar_raiz
from forja.ajuste.suavizado import suavizar
from forja.rig import Rig, cargar_garras, matrix_to_quat
from forja.triangulacion.robusta import gemelos, seleccionar, triangular
from forja.vistas.layout import Layout

DEF = {
    "z_min_otros": 0.03,      # m: ninguna articulación que no sea de contacto por debajo (radio de la pata)
    "w_suelo": 100.0,         # peso del suelo en la IK (px² por px de penetración)
    "w_vecinos": 20.0,        # previo de la pasada 2 hacia la media de sus vecinos (px² por rad²)
    "suave_lam": "auto", "suave_delta": 0.08,  # suavizado de ángulos: λ y escala de Huber (rad/cuadro²). "auto": 0,1 con 3
                              # vistas o más y 0,5 con 2 (clip de desarrollo: con λ = 2 el error subía de 0,58 a 0,85 cm
                              # con 3 vistas y el golpe del mordisco perdía el 40 % de su pico; con 2 vistas, λ = 0,1
                              # duplicaba el temblor)
    "suave_perdida": "huber",                # "huber" o "cauchy" (ver suavizado.py)
    "raiz_lam": 0.3,                         # suavizado de la raíz (λ; escala de Huber 4 mm/cuadro²)
    "w_plantado": 100.0,      # la punta en su apoyo: información = w · 1/M² (1 mm ≈ 3 px · √w)
    "w_final": 50.0,          # previo de la pasada final hacia la curva suavizada
    "seleccion": True,        # False: triangulación directa con todas las lecturas (sin Viterbi), para comparar
    "gemelas": False,         # candidatos de gemela (la pata que se superpone en una vista). Apagados: con lecturas reales
                              # empeoran el ajuste con 2 vistas (examen de F1: 1,68 frente a 1,07 cm) y con 3 (clip de
                              # desarrollo: 0,94 frente a 0,84 cm); solo ayudan en intercambios simulados
    "sel_kw": {"pen_descarte": 12.0, "tr_descarte": 4.0},  # la selección solo descarta lo muy claro; el resto lo hace
                              # el peso robusto por vista de la IK
    "datos": "vistas",        # "vistas": la IK persigue las lecturas 2D de cada vista con un peso robusto por vista;
                              # "puntos": persigue los puntos triangulados (un peso robusto por articulación)
}


def lecturas_elegidas(uv, conf, choice, wmin=0.05):
    """Las lecturas que eligió la triangulación por vista (propia o de la gemela) y su peso conf² (0 si se descartó)."""
    V, T, J = conf.shape
    src = np.where(choice >= 0, choice, 0)  # (T, J, V)
    ti = np.arange(T)[:, None]
    uvs = np.stack([uv[v][ti, src[..., v]] for v in range(V)])
    ws = np.stack([np.where(choice[..., v] >= 0, np.maximum(conf[v][ti, src[..., v]], wmin) ** 2, 0.0) for v in range(V)])
    return uvs, ws


def _relleno(P, Lam, Pall):
    """Para los arranques: donde la elección dejó un eje sin medir, el punto de la triangulación con todas las vistas."""
    ev = np.linalg.eigvalsh(Lam)[..., 0]
    bad = ev < 1e-6 * np.maximum(np.linalg.eigvalsh(Lam)[..., -1], 1e-12)
    return np.where(bad[..., None], Pall, P)


def ajustar(det, rig, lay, cfg, *, suelo=True, opciones=None, garras=None, log=print):
    """garras: {articulación de contacto: (vértices, pesos)} de rig.cargar_garras: con suelo, la parte baja de la malla
    de cada pata tampoco baja de z = 0."""
    garras = garras or {}
    o = dict(DEF, **(opciones or {}))
    vistas = list(det["vistas"])
    L = lay.sub(vistas)
    A, b = L.A()
    M = L.M
    uv, conf = np.asarray(det["uv"], np.float64), np.asarray(det["conf"], np.float64)
    V, T, J = conf.shape
    names = rig.kp_names
    assert list(det["names"]) == names, "las articulaciones del lector no son las del rig"
    contacts = [names.index(n) for n in cfg.get("contacts", cfg.get("feet", [])) if n in names]
    h0 = rig.rest[contacts, 2]
    zmin = np.full(J, o["z_min_otros"])
    zmin[contacts] = h0
    if o["suave_lam"] == "auto":
        o["suave_lam"] = 0.1 if V >= 3 else 0.5
    twins = gemelos(rig, L) if o["gemelas"] else {v: {} for v in vistas}
    log(f"vistas {vistas}; gemelas por vista: " + ", ".join(f"{v} {sum(len(x) for x in twins[v].values()) // 2}" for v in vistas))
    # 1. triangulación
    Pall, Lall, _ = triangular(uv, conf, A, b)
    if o["seleccion"]:
        P, Lam, choice = seleccionar(uv, conf, A, b, twins, vistas, suelo=(zmin - 0.01) if suelo else None, log=log, **o["sel_kw"])
    else:
        P, Lam, choice = Pall, Lall, np.broadcast_to(np.arange(J)[None, :, None], (T, J, V)).copy()
    Pf = _relleno(P, Lam, Pall)
    uvs, wsel = lecturas_elegidas(uv, conf, choice)
    por_vistas = o["datos"] == "vistas"
    obs_de = lambda js: obs_vistas(uvs, wsel, A, b, js) if por_vistas else obs_puntos(P[:, js], Lam[:, js])
    # 2. raíz
    trans, root_R, _ = ajustar_raiz(rig, P, Lam, M, vistas=(uvs, wsel, A, b) if por_vistas else None,
                                    suave=(o["raiz_lam"], 0.004), perdida=o["suave_perdida"], log=log)
    Mroot = rig.root_matrix(trans, root_R)
    # 3–4. IK por cadena, dos pasadas, y suavizado
    nb = len(rig.names)
    eul = np.zeros((T, nb, 3))
    diag = {}
    if suelo and garras:
        log(f"suelo: además, la malla baja de {len(garras)} patas (garras.json) no pasa de z = 0")
    malla_de = lambda js: garras.get(names[js[-1]]) if suelo else None
    for bones, joints in rig.chains():
        ik = CadenaIK(rig, bones, M, malla_de(joints))
        obs = obs_de(joints[1:])
        zm = zmin[joints[1:]] if suelo else None
        wg = o["w_suelo"] if suelo else 0.0
        st = np.stack([ik.inicio_geometrico(Mroot, Pf[:, joints[1:]]), np.zeros((T, ik.P))])
        th1, _, _ = ik.resolver_obs(Mroot, obs, st, zmin=zm, wg=wg)
        vec = th1.copy()
        vec[1:-1] = 0.5 * (th1[:-2] + th1[2:])
        th2, _, e2 = ik.resolver_obs(Mroot, obs, np.stack([th1, vec]), prior=vec, wp=o["w_vecinos"], zmin=zm, wg=wg)
        wq = 1.0 / (1.0 + (e2.mean(1) / 8.0) ** 2)  # cuadros que la IK no pudo encajar pesan menos en el suavizado
        th3 = np.clip(suavizar(th2, wq, o["suave_lam"], o["suave_delta"], perdida=o["suave_perdida"]), ik.lo, ik.hi)
        eul[:, bones] = th3.reshape(T, len(bones), 3)
        diag[rig.names[bones[0]]] = {"px_mediana": round(float(np.median(e2)), 2), "px_p95": round(float(np.percentile(e2, 95)), 2)}
    J3, _ = rig.forward(trans, root_R, eul)
    stance = np.zeros((T, len(contacts)), bool)
    # 5. apoyos y plantado
    if suelo and contacts:
        tips = J3[:, contacts]
        stance = apoyos(tips, h0)
        tplant = plantar(tips, stance, h0)
        for bones, joints in rig.chains():
            if joints[-1] not in contacts:
                continue
            k = contacts.index(joints[-1])
            ik = CadenaIK(rig, bones, M, malla_de(joints))
            n = len(bones)
            on = stance[:, k]
            tgt = np.zeros((T, n, 3))
            tgt[on, -1] = tplant[on, k]
            Lp = np.zeros((T, n, 3, 3))
            Lp[on, -1] = np.eye(3) * o["w_plantado"] / M ** 2
            obs = obs_unir(obs_de(joints[1:]), obs_puntos(tgt, Lp))
            th3 = eul[:, bones].reshape(T, -1)
            th4, _, _ = ik.resolver_obs(Mroot, obs, th3[None], prior=th3, wp=o["w_final"], zmin=zmin[joints[1:]],
                                       wg=o["w_suelo"], sigmas=(12.0, 6.0))
            eul[:, bones] = th4.reshape(T, len(bones), 3)
        n_ap = sum(sum(1 for v, a, b_ in runs(stance[:, k]) if v) for k in range(len(contacts)))
        log(f"apoyos: {n_ap} en {len(contacts)} patas; {stance.mean():.0%} de los cuadros de punta apoyados")
    Jf, _ = rig.forward(trans, root_R, eul)
    rep = np.stack([v.project(Jf) for v in L.views])
    err = np.linalg.norm(rep - uv, axis=-1)
    log("reproyección del ajuste (px, mediana / media): " + ", ".join(
        f"{v} {np.median(err[i]):.2f} / {err[i].mean():.2f}" for i, v in enumerate(vistas)))
    return {"trans": trans, "root_R": root_R, "euler": eul, "joints": Jf, "joints_previos": J3, "stance": stance, "contacts": [names[c] for c in contacts],
            "choice": choice, "P": P, "Lam": Lam, "diag": diag, "reproj_px": {v: float(err[i].mean()) for i, v in enumerate(vistas)}}


def guardar(res, rig, path, fps=24.0):
    out = {"fps": fps, "bones": rig.names, "trans": np.round(res["trans"], 6).tolist(),
           "root_quat": np.round(matrix_to_quat(res["root_R"]), 7).tolist(), "euler": np.round(res["euler"], 6).tolist(),
           "joints": np.round(res["joints"], 5).tolist(), "joint_names": rig.kp_names,
           "forja": {"contacts": res["contacts"], "stance": res["stance"].astype(int).tolist(), "diag": res["diag"],
                     "reproj_px": res["reproj_px"]}}
    json.dump(out, open(path, "w", encoding="utf-8"))


def main():
    from forja.lector.leer import cargar
    ap = argparse.ArgumentParser()
    ap.add_argument("det"), ap.add_argument("info"), ap.add_argument("cfg"), ap.add_argument("out")
    ap.add_argument("--vistas", default=None, help="usar solo estas vistas del det (p. ej. front,side)")
    ap.add_argument("--sin-suelo", action="store_true")
    a = ap.parse_args()
    det = cargar(a.det)
    if a.vistas:
        keep = [det["vistas"].index(v) for v in a.vistas.split(",")]
        det["uv"], det["conf"], det["vistas"] = det["uv"][keep], det["conf"][keep], [det["vistas"][i] for i in keep]
    rig = Rig(a.info)
    cfg = json.load(open(a.cfg, encoding="utf-8"))
    res = ajustar(det, rig, Layout(cfg), cfg, suelo=not a.sin_suelo, garras=cargar_garras(a.cfg, rig))
    guardar(res, rig, a.out, det.get("fps", 24.0))
    print("guardado", a.out)


if __name__ == "__main__":
    main()
