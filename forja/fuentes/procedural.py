"""Movimiento procedural por objetivos de IK: la marcha de N patas y gestos (alzarse, mordisco, abalanzarse).

python -m forja.fuentes.procedural rig_info.json creature.json CARPETA [--clips marcha,alzarse,mordisco,abalanzarse]
    → CARPETA/<clip>/truth.json (motion.json de anymove) y truth_forja.json (articulaciones, apoyos verdaderos)

En F2 sirve para fabricar los clips del examen: movimientos con verdad conocida que **respetan el suelo** (las puntas
apoyadas, en su altura de reposo y quietas; nada por debajo) y que se renderizan con las cámaras del cuadro.

Cómo se hace:
- el **cuerpo** (la raíz) sigue una trayectoria: posición de su cabeza b(t) y rotación de mundo R(t);
- cada **punta** tiene un objetivo por cuadro: en apoyo, un punto fijo del suelo (z = altura de reposo); en vuelo, un
  arco del punto donde despegó al punto donde aterriza, o una pose en el marco del cuerpo (patas alzadas);
- la **IK por pata** (ajuste/ik.py, la misma del ajuste) pone la punta en su objetivo con la postura más cercana a un
  «estilo» que varía despacio (rodillas más o menos altas), dentro de los límites y sin bajar del suelo;
- colmillos y abdomen siguen curvas de ángulos.
Las patas se reconocen solas: lado por el signo de X en reposo y orden de delante a atrás por Y.
"""
import argparse
import json
import math
import pathlib

import numpy as np

from forja.ajuste.ik import CadenaIK
from forja.rig import Rig, axis_angle_matrix, cargar_garras, matrix_to_quat

FPS = 24


def smooth01(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


def ease(t, t0, t1):
    """0 antes de t0, 1 después de t1, suave entre medias."""
    return smooth01((t - t0) / max(t1 - t0, 1e-9))


def rot_xyz(rx, ry, rz):
    return axis_angle_matrix(np.array([0, 0, rz])) @ axis_angle_matrix(np.array([0, ry, 0])) @ axis_angle_matrix(np.array([rx, 0, 0]))


class Arana:
    """Lo que el generador sabe de la criatura: cadenas, patas (lado y orden), puntas y alturas de reposo."""

    def __init__(self, rig, cfg, M=0.0035, garras=None):
        self.rig, self.cfg, self.M = rig, cfg, M
        self.garras = garras or {}
        names = rig.kp_names
        self.contacts = [names.index(n) for n in cfg.get("contacts", [])]
        self.legs = []  # (bones, joints) de las cadenas que acaban en un contacto
        self.otras = []
        for bones, joints in rig.chains():
            (self.legs if joints[-1] in self.contacts else self.otras).append((bones, joints))
        self.body = rig.joint(rig.head_joint[0])
        self.b0 = rig.rest[self.body].copy()
        tips = np.array([rig.rest[j[-1]] for _, j in self.legs])
        self.h0 = tips[:, 2].copy()
        self.n_local = tips - self.b0  # punta neutra respecto a la cabeza del cuerpo, en reposo
        self.left = tips[:, 0] > self.b0[0]
        order = {}
        for side in (True, False):
            idx = [i for i in range(len(self.legs)) if self.left[i] == side]
            for k, i in enumerate(sorted(idx, key=lambda i: tips[i, 1])):
                order[i] = k  # 0 = la de delante
        self.order = np.array([order[i] for i in range(len(self.legs))])
        self.zmin_otros = 0.03

    def nombre_pata(self, i):
        return ("L" if self.left[i] else "R") + str(self.order[i] + 1)


def raiz_desde(ar, b, Rw):
    """Cabeza del cuerpo b (T, 3) y rotación de mundo Rw (T, 3, 3) → trans y root_R del rig de anymove."""
    rig = ar.rig
    B0r = rig.B[0, :3, :3]
    return b - ar.b0, np.einsum("ji,tjk,kl->til", B0r, Rw, B0r)


def resolver_patas(ar, trans, root_R, tips_tgt, estilo, rng=None):
    """IK de cada pata hacia su objetivo de punta con la postura más cercana a su estilo. Devuelve eul (T, nb, 3)."""
    rig = ar.rig
    T = len(trans)
    eul = np.zeros((T, len(rig.names), 3))
    Mroot = rig.root_matrix(trans, root_R)
    err = []
    for i, (bones, joints) in enumerate(ar.legs):
        ik = CadenaIK(rig, bones, ar.M, ar.garras.get(rig.kp_names[joints[-1]]))
        n = len(bones)
        tgt = np.zeros((T, n, 3))
        tgt[:, -1] = tips_tgt[:, i]
        Lam = np.zeros((T, n, 3, 3))
        Lam[:, -1] = np.eye(3) * 100.0 / ar.M ** 2
        prior = estilo[:, i]
        zmin = np.full(n, ar.zmin_otros)
        zmin[-1] = ar.h0[i] - 0.0005
        th, _, _ = ik.resolver(Mroot, tgt, Lam, np.stack([prior, np.zeros_like(prior)]), prior=prior, wp=50.0, wr=0.5,
                               zmin=zmin, wg=400.0, iters=40, sigmas=(1e6,))
        eul[:, bones] = th.reshape(T, n, 3)
        pts = ik.fk(Mroot, th)
        err.append(np.linalg.norm(pts[:, -1] - tips_tgt[:, i], axis=-1))
    return eul, np.array(err).T


def estilo_patas(ar, T, rng, amp_deg=8.0):
    """Postura preferida por pata y cuadro (T, K, 9): ondas lentas al azar en la flexión (X) de fémur y tibia."""
    t = np.arange(T) / FPS
    out = np.zeros((T, len(ar.legs), 9))
    for i in range(len(ar.legs)):
        for k in (0, 3, 6):  # X de cada hueso
            f, ph = rng.uniform(0.15, 0.5), rng.uniform(0, 2 * math.pi)
            out[:, i, k] = math.radians(amp_deg) * np.sin(2 * math.pi * f * t + ph) * (0.5 if k == 6 else 1.0)
    return out


def curvas_otras(ar, T, fang_x=None, abd_x=None):
    """Ángulos de colmillos y abdomen (T, …) en grados → eul parcial."""
    eul = {}
    for bones, joints in ar.otras:
        name = ar.rig.names[bones[0]]
        e = np.zeros((T, len(bones), 3))
        if name.startswith("fang") and fang_x is not None:
            e[:, 0, 0] = np.radians(fang_x)
        if name.startswith("abdomen") and abd_x is not None:
            e[:, 0, 0] = np.radians(abd_x)
        lo, hi = ar.rig.lo[bones], ar.rig.hi[bones]
        eul[tuple(bones)] = np.clip(e, lo, hi)
    return eul


# ------------------------------------------------------------------------------------------------ clips

def marcha(ar, T=120, rng=None, v=0.105, periodo=1.0, deber=0.65, alto=0.09, y0=0.29):
    """Marcha de 8 patas en trípodes alternos (tetrápodos: L1 R2 L3 R4 / R1 L2 R3 L4) con un pequeño desfase de onda
    de atrás a delante; el cuerpo avanza hacia −Y a v m/s con un leve vaivén y un giro suave."""
    t = np.arange(T) / FPS
    K = len(ar.legs)
    yaw = np.radians(8.0) * np.sin(2 * math.pi * t / (T / FPS))
    b = np.stack([0.015 * np.sin(2 * math.pi * t / periodo), y0 - v * t,
                  ar.b0[2] - 0.02 + 0.008 * np.sin(4 * math.pi * t / periodo)], 1)
    b[:, 0] += ar.b0[0]
    Rw = np.stack([rot_xyz(np.radians(1.5) * math.sin(4 * math.pi * x / periodo), np.radians(2.0) * math.sin(2 * math.pi * x / periodo), y)
                   for x, y in zip(t, yaw)])
    grupo = np.where(ar.left, ar.order % 2, (ar.order + 1) % 2)
    fase = 0.5 * grupo + 0.06 * (3 - ar.order)

    def pie(i, tm):
        """Dónde pisa la pata i si su apoyo está centrado en el instante tm: la neutra bajo el cuerpo en ese instante."""
        bx, by = np.interp(tm, t, b[:, 0]), np.interp(tm, t, b[:, 1])
        yw = np.interp(tm, t, yaw)
        c, s = math.cos(yw), math.sin(yw)
        nl = ar.n_local[i]
        return np.array([bx + c * nl[0] - s * nl[1], by + s * nl[0] + c * nl[1], ar.h0[i]])

    tips = np.zeros((T, K, 3))
    stance = np.zeros((T, K), bool)
    for i in range(K):
        for f in range(T):
            ph = (t[f] / periodo + fase[i]) % 1.0
            if ph < deber:
                ts = t[f] - ph * periodo
                tips[f, i] = pie(i, ts + deber * periodo / 2)
                stance[f, i] = True
            else:
                ts_prev = t[f] - ph * periodo
                ts_next = ts_prev + periodo
                a = pie(i, ts_prev + deber * periodo / 2)
                c = pie(i, ts_next + deber * periodo / 2)
                u = (ph - deber) / (1 - deber)
                p = a + (c - a) * smooth01(u)
                p[2] = ar.h0[i] + alto * math.sin(math.pi * u)
                tips[f, i] = p
    trans, root_R = raiz_desde(ar, b, Rw)
    estilo = estilo_patas(ar, T, rng)
    otras = curvas_otras(ar, T, fang_x=6 * np.sin(2 * math.pi * t * 0.8), abd_x=4 * np.sin(2 * math.pi * t / periodo))
    return trans, root_R, tips, stance, estilo, otras


def alzarse(ar, T=120, rng=None, sube=(0.8, 1.7), baja=(3.6, 4.5), chasquido=2.6, peso_atras=0.0, retraso=0.1):
    """Se alza amenazando: el cuerpo gira 32° hacia arriba sobre las patas de atrás, las dos de delante de cada lado se
    levantan y ondean, colmillos abiertos y un chasquido; luego baja y vuelve a apoyarlas donde estaban."""
    t = np.arange(T) / FPS
    K = len(ar.legs)
    up = ease(t, *sube) * (1 - ease(t, *baja))
    pitch = -np.radians(32) * up
    # pivote cerca de las caderas de atrás: la cabeza del cuerpo sube y retrocede un poco
    # peso_atras: antes de levantar las delanteras, el cuerpo lleva el peso hacia atrás (y no lo devuelve hasta que
    # vuelven a apoyar); sin él, el centro de masas queda por delante de las patas de atrás (F3, equilibrio)
    peso = peso_atras * ease(t, sube[0] - 0.1, sube[0] + 0.25) * (1 - ease(t, baja[1] - 0.15, baja[1] + 0.25))
    b = np.stack([np.full(T, ar.b0[0]), ar.b0[1] + 0.10 * up + peso, ar.b0[2] + 0.17 * up + 0.004 * np.sin(2 * math.pi * 0.6 * t)], 1)
    Rw = np.stack([rot_xyz(p, 0.0, np.radians(3) * math.sin(2 * math.pi * 0.3 * x)) for p, x in zip(pitch, t)])
    tips = np.zeros((T, K, 3))
    stance = np.ones((T, K), bool)
    rest_tips = ar.b0 + ar.n_local
    for i in range(K):
        if ar.order[i] <= 1:  # las dos de delante: en el aire mientras está alzada
            lift = ease(t, sube[0] + retraso, sube[1] - 0.1) * (1 - ease(t, baja[0] + 0.1, baja[1] - retraso))
            # en el marco del cuerpo: la punta sube y va hacia delante, y ondea
            wave = 0.06 * np.sin(2 * math.pi * 1.4 * t + (0 if ar.left[i] else 1.2) + ar.order[i])
            local = ar.n_local[i] + np.stack([np.zeros(T), -0.10 * lift, (0.30 + wave) * lift], 1)
            world_body = b + np.einsum("tij,tj->ti", Rw, local)
            tips[:, i] = rest_tips[i] * (1 - lift[:, None]) + world_body * lift[:, None]
            tips[:, i, 2] = np.maximum(tips[:, i, 2], ar.h0[i])
            stance[:, i] = lift < 1e-3
        else:
            tips[:, i] = rest_tips[i]
    trans, root_R = raiz_desde(ar, b, Rw)
    estilo = estilo_patas(ar, T, rng, 6.0)
    fang = 22 * ease(t, sube[0] + 0.4, sube[1] + 0.1) * (1 - ease(t, baja[0] - 0.2, baja[0] + 0.4)) \
        - 18 * ease(t, chasquido, chasquido + 0.15) * (1 - ease(t, chasquido + 0.3, chasquido + 0.5))
    otras = curvas_otras(ar, T, fang_x=fang, abd_x=-20 * up + 3 * np.sin(2 * math.pi * 0.7 * t))
    return trans, root_R, tips, stance, estilo, otras


def mordisco(ar, T=120, rng=None, golpes=(0.7, 3.0)):
    """Mordisco con las ocho patas clavadas: se carga hacia atrás con la cabeza alta y los colmillos abiertos, lanza la
    cabeza hacia delante y abajo y cierra de golpe (3 cuadros), sujeta, vuelve; un segundo mordisco más corto."""
    t = np.arange(T) / FPS
    K = len(ar.legs)

    def golpe(t0):
        carga = ease(t, t0, t0 + 0.55)
        lanza = ease(t, t0 + 0.55, t0 + 0.68)
        vuelve = ease(t, t0 + 1.1, t0 + 1.8)
        return carga, lanza, vuelve

    c1, l1, v1 = golpe(golpes[0])
    c2, l2, v2 = golpe(golpes[1]) if len(golpes) > 1 else (np.zeros(T),) * 3
    dy = (0.07 * c1 - 0.25 * l1) * (1 - v1) + (0.05 * c2 - 0.18 * l2) * (1 - v2)
    dz = (0.02 * c1 - 0.05 * l1) * (1 - v1) + (0.015 * c2 - 0.035 * l2) * (1 - v2)
    pit = (-np.radians(10) * c1 + np.radians(11) * l1) * (1 - v1) + (-np.radians(8) * c2 + np.radians(8) * l2) * (1 - v2)
    b = np.stack([np.full(T, ar.b0[0]), ar.b0[1] + dy, ar.b0[2] + dz], 1)
    Rw = np.stack([rot_xyz(p, 0.0, 0.0) for p in pit])
    tips = np.repeat((ar.b0 + ar.n_local)[None], T, 0)
    stance = np.ones((T, K), bool)
    trans, root_R = raiz_desde(ar, b, Rw)
    estilo = estilo_patas(ar, T, rng, 5.0)
    fang = (24 * c1 - 44 * l1) * (1 - v1) + (20 * c2 - 38 * l2) * (1 - v2)
    otras = curvas_otras(ar, T, fang_x=fang, abd_x=(10 * c1 - 16 * l1) * (1 - v1) + (8 * c2 - 12 * l2) * (1 - v2))
    return trans, root_R, tips, stance, estilo, otras


def abalanzarse(ar, T=120, rng=None, salto=0.44, y0=0.30, t_up=0.4, t_off=1.35, t_land=1.80):
    """Se agacha, salta hacia delante (vuelo balístico de ~0,4 s con las patas delanteras estiradas hacia la presa),
    aterriza con las de delante primero, amortigua y se recompone."""
    t = np.arange(T) / FPS
    K = len(ar.legs)
    agache = ease(t, t_up, t_off - 0.1) * (1 - ease(t, t_off - 0.1, t_off))
    vuelo = ((t >= t_off) & (t <= t_land)).astype(float)
    u = np.clip((t - t_off) / (t_land - t_off), 0, 1)
    avance = salto * smooth01((t - (t_off - 0.1)) / (t_land - t_off + 0.1))
    amort = ease(t, t_land, t_land + 0.15) * (1 - ease(t, t_land + 0.15, t_land + 0.9))
    z = ar.b0[2] - 0.12 * agache + 0.20 * np.sin(math.pi * u) * vuelo - 0.08 * amort
    b = np.stack([np.full(T, ar.b0[0]), y0 + ar.b0[1] - avance, z], 1)
    pit = -np.radians(6) * agache - np.radians(8) * np.sin(math.pi * u) * vuelo + np.radians(2) * amort
    Rw = np.stack([rot_xyz(p, 0.0, 0.0) for p in pit])
    start = np.array([ar.b0[0], y0 + ar.b0[1], 0.0])
    end = np.array([ar.b0[0], y0 + ar.b0[1] - salto, 0.0])
    tips = np.zeros((T, K, 3))
    stance = np.ones((T, K), bool)
    for i in range(K):
        a = start + ar.n_local[i] * [1, 1, 0]
        c = end + ar.n_local[i] * [1, 1, 0]
        a[2] = c[2] = ar.h0[i]
        # despegue: las de atrás despegan las últimas; aterrizaje: las de delante primero
        off = t_off + 0.03 * (ar.order[i] - 1.5)
        land = t_land + 0.04 * ar.order[i]
        uu = np.clip((t - off) / (land - off), 0, 1)
        air = (t > off) & (t < land)
        # en el aire: punta en el marco del cuerpo, delanteras estiradas hacia delante y arriba, traseras recogidas
        reach = np.array([0.0, -0.12, 0.10]) if ar.order[i] <= 1 else np.array([0.0, 0.06, 0.05])
        local = ar.n_local[i] + reach * np.sin(math.pi * uu)[:, None]
        body_pt = b + np.einsum("tij,tj->ti", Rw, local)
        ground = a + (c - a) * smooth01(uu)[:, None]
        mix = np.sin(math.pi * uu)[:, None] ** 0.5
        p = np.where(air[:, None], ground * (1 - mix) + body_pt * mix, np.where((t <= off)[:, None], a, c))
        p[:, 2] = np.maximum(p[:, 2], ar.h0[i])
        tips[:, i] = p
        stance[:, i] = ~air
    trans, root_R = raiz_desde(ar, b, Rw)
    estilo = estilo_patas(ar, T, rng, 6.0)
    otras = curvas_otras(ar, T, fang_x=20 * vuelo * np.sin(math.pi * u) + 8 * amort, abd_x=12 * agache - 10 * vuelo * np.sin(math.pi * u))
    return trans, root_R, tips, stance, estilo, otras



def _rz(yaw):
    c, s_ = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


def fases_por_topologia(ar, onda=0.06):
    """Fase de cada pata a partir de la topología: en cada lado, patas alternas en dos grupos y el lado contrario en
    contrafase (trípodes con 6 patas, tetrápodos con 8, trote con 4), más una onda pequeña de atrás hacia delante."""
    n_lado = np.array([np.sum(ar.left == ar.left[i]) for i in range(len(ar.legs))])
    grupo = np.where(ar.left, ar.order % 2, (ar.order + 1) % 2)
    return (0.5 * grupo + onda * (n_lado - 1 - ar.order)) % 1.0


def marcha_ciclo(ar, T=24, rng=None, v=0.30, giro_deg_s=0.0, deber=0.6, alto=0.08, periodo_c=None, estilo_deg=0.0):
    """Un ciclo de marcha N-patas que cierra en bucle: cuerpo a velocidad v (m/s) hacia delante (−Y) y giro constante;
    pies clavados en el suelo durante el apoyo, arco de vuelo entre el punto de despegue y el de aterrizaje. Todo es
    función analítica del tiempo (sin interpolar), así el cuadro T coincide con el 0 respecto al cuerpo. Exportado
    «en el sitio» (paquete/glb.py --en-sitio), el avance y el giro van al root motion."""
    periodo_c = periodo_c or T
    P = periodo_c / FPS
    t = np.arange(T) / FPS
    K = len(ar.legs)
    w = math.radians(giro_deg_s)
    fase = fases_por_topologia(ar)

    def yaw(x):
        return w * x

    def pos(x):  # avance integrado a lo largo del rumbo (−Y girado)
        if abs(w) < 1e-9:
            return np.array([0.0, -v * x, 0.0])
        return np.array([v / w * (1 - math.cos(w * x)), -v / w * math.sin(w * x), 0.0])

    def cuerpo(x):
        sway = 0.012 * math.sin(2 * math.pi * x / P)
        return ar.b0 + pos(x) + _rz(yaw(x)) @ np.array([sway, 0.0, -0.02 + 0.007 * math.cos(4 * math.pi * x / P)])

    def pie(i, tm):
        nl = _rz(yaw(tm)) @ ar.n_local[i]
        p = cuerpo(tm) + nl
        return np.array([p[0], p[1], ar.h0[i]])

    b = np.stack([cuerpo(x) for x in t])
    Rw = np.stack([_rz(yaw(x)) @ rot_xyz(math.radians(1.2) * math.sin(4 * math.pi * x / P), math.radians(1.5) * math.sin(2 * math.pi * x / P), 0.0)
                   for x in t])
    tips = np.zeros((T, K, 3))
    stance = np.zeros((T, K), bool)
    for i in range(K):
        for f in range(T):
            ph = (t[f] / P + fase[i]) % 1.0
            ts = t[f] - ph * P
            if ph < deber:
                tips[f, i] = pie(i, ts + deber * P / 2)
                stance[f, i] = True
            else:
                a, c = pie(i, ts + deber * P / 2), pie(i, ts + P + deber * P / 2)
                u = (ph - deber) / (1 - deber)
                p = a + (c - a) * smooth01(u)
                p[2] = ar.h0[i] + alto * math.sin(math.pi * u)
                tips[f, i] = p
    trans, root_R = raiz_desde(ar, b, Rw)
    estilo = np.zeros((T, K, 9))
    if estilo_deg:
        estilo[..., 0] = math.radians(estilo_deg) * np.sin(2 * math.pi * t / P)[:, None]
    otras = curvas_otras(ar, T, fang_x=5 * np.sin(2 * math.pi * t / P), abd_x=3 * np.sin(2 * math.pi * t / P + 1.0))
    return trans, root_R, tips, stance, estilo, otras


def retroceder_herida(ar, T=36, rng=None, empuje=0.14):
    """Recibe un golpe de frente y a su izquierda: el cuerpo sale despedido hacia atrás y se ladea, el abdomen se
    encoge, los colmillos se cierran; las dos patas de delante de cada lado dan un paso atrás para no caer y el cuerpo
    se recompone algo más atrás que donde estaba."""
    t = np.arange(T) / FPS
    K = len(ar.legs)
    golpe = ease(t, 0.0, 0.16)
    vuelve = ease(t, 0.35, 1.2)
    atras = empuje * golpe * (1 - 0.35 * vuelve)  # se queda 0,65 del empuje atrás
    b = np.stack([np.full(T, ar.b0[0]) - 0.03 * golpe * (1 - vuelve), ar.b0[1] + atras,
                  ar.b0[2] + 0.03 * golpe * (1 - vuelve) - 0.02 * vuelve * (1 - ease(t, 1.0, 1.4))], 1)
    Rw = np.stack([rot_xyz(-math.radians(14) * g * (1 - v_), math.radians(9) * g * (1 - v_), math.radians(6) * g * (1 - v_))
                   for g, v_ in zip(golpe, vuelve)])
    tips = np.repeat((ar.b0 + ar.n_local)[None], T, 0).copy()
    stance = np.ones((T, K), bool)
    fin = np.array([0.0, empuje * 0.65, 0.0])
    for i in range(K):
        if ar.order[i] <= 1:  # paso atrás de las delanteras (primero el par de delante)
            a0, a1 = 0.10 + 0.08 * ar.order[i], 0.42 + 0.08 * ar.order[i]
            u = np.clip((t - a0) / (a1 - a0), 0, 1)
            air = (t > a0) & (t < a1)
            start = ar.b0 + ar.n_local[i]
            end = start + fin
            tips[:, i] = start + (end - start) * smooth01(u)[:, None]
            tips[:, i, 2] = ar.h0[i] + 0.07 * np.sin(math.pi * u)
            stance[:, i] = ~air
        else:  # las de atrás aguantan (clavadas), luego un reajuste corto al final
            a0, a1 = 0.75 + 0.06 * (ar.order[i] - 2), 1.0 + 0.06 * (ar.order[i] - 2)
            u = np.clip((t - a0) / (a1 - a0), 0, 1)
            air = (t > a0) & (t < a1)
            start = ar.b0 + ar.n_local[i]
            end = start + fin
            tips[:, i] = start + (end - start) * smooth01(u)[:, None]
            tips[:, i, 2] = ar.h0[i] + 0.05 * np.sin(math.pi * u)
            stance[:, i] = ~air
    trans, root_R = raiz_desde(ar, b, Rw)
    estilo = estilo_patas(ar, T, rng, 4.0)
    otras = curvas_otras(ar, T, fang_x=-25 * golpe * (1 - vuelve) + 5 * vuelve * (1 - ease(t, 1.1, 1.5)),
                         abd_x=22 * golpe * (1 - vuelve))
    return trans, root_R, tips, stance, estilo, otras


# El juego: clips cortos, con sus eventos (segundos) para las anotaciones (paquete/anotaciones.py)
JUEGO = {
    "caminar": dict(f=marcha_ciclo, T=24, kw=dict(v=0.30), bucle=True, eventos={}),
    "trotar": dict(f=marcha_ciclo, T=16, kw=dict(v=0.65, deber=0.5, alto=0.10), bucle=True, eventos={}),
    "girar": dict(f=marcha_ciclo, T=24, kw=dict(v=0.0, giro_deg_s=20.0, deber=0.6, alto=0.07), bucle=True, eventos={}),
    "alzarse": dict(f=alzarse, T=72, kw=dict(sube=(0.15, 0.95), baja=(2.15, 2.9), chasquido=1.55, peso_atras=0.24, retraso=0.35),
                    bucle=False,
                    eventos={"anticipacion": (0.15, 0.95), "activa": (0.95, 2.15), "impacto": [("fangL_end", 1.70), ("fangR_end", 1.70)],
                             "recuperacion": (2.15, 2.9), "cancelable": (2.4, 3.0)}),
    "mordisco": dict(f=mordisco, T=40, kw=dict(golpes=(0.15,)), bucle=False,
                     eventos={"anticipacion": (0.15, 0.70), "activa": (0.70, 0.83), "impacto": [("fangL_end", 0.83), ("fangR_end", 0.83)],
                              "recuperacion": (0.83, 1.95), "cancelable": (1.25, 1.67)}),
    "abalanzarse": dict(f=abalanzarse, T=60, kw=dict(salto=0.44, y0=0.22, t_up=0.10, t_off=0.80, t_land=1.25), bucle=False,
                        eventos={"anticipacion": (0.10, 0.80), "activa": (0.80, 1.25), "impacto": [("leg0_3", 1.25), ("leg7_3", 1.25)],
                                 "recuperacion": (1.25, 2.20), "cancelable": (1.9, 2.5)}),
    "retroceder_herida": dict(f=retroceder_herida, T=36, kw={}, bucle=False,
                              eventos={"recibido": 0.04, "cancelable": (0.9, 1.5)}),
}

CLIPS = {"marcha": marcha, "alzarse": alzarse, "mordisco": mordisco, "abalanzarse": abalanzarse}


def generar(ar, nombre, T=120, seed=0, juego=False):
    rng = np.random.default_rng(seed)
    if juego:
        spec = JUEGO[nombre]
        trans, root_R, tips, stance, estilo, otras = spec["f"](ar, spec["T"], rng, **spec["kw"])
    else:
        trans, root_R, tips, stance, estilo, otras = CLIPS[nombre](ar, T, rng)
    eul, err = resolver_patas(ar, trans, root_R, tips, estilo)
    for bones, e in otras.items():
        eul[:, list(bones)] = e
    if juego:  # huesos que la piel del modelo no deja mover bien (creature.json "forja" → "huesos_quietos")
        for h in ar.cfg.get("forja", {}).get("huesos_quietos", []):
            eul[:, ar.rig.bone(h)] = 0.0
    J, _ = ar.rig.forward(trans, root_R, eul)
    return {"trans": trans, "root_R": root_R, "euler": eul, "joints": J, "stance": stance, "tips_tgt": tips, "ik_err": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("info"), ap.add_argument("cfg"), ap.add_argument("out")
    ap.add_argument("--clips", default=None)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--juego", action="store_true", help="los clips cortos del juego (JUEGO), con sus eventos")
    a = ap.parse_args()
    a.clips = a.clips or ",".join(JUEGO if a.juego else CLIPS)
    rig = Rig(a.info)
    cfg = json.load(open(a.cfg, encoding="utf-8"))
    ar = Arana(rig, cfg, cfg.get("layout", {}).get("M_PER_PX", 0.0035), cargar_garras(a.cfg, rig))
    print(f"garras: {len(ar.garras)} patas con su malla baja (z ≥ 0)" if ar.garras else "sin garras.json: suelo solo en las articulaciones")
    for k, nombre in enumerate(a.clips.split(",")):
        g = generar(ar, nombre, a.frames, seed=100 + k, juego=a.juego)
        d = pathlib.Path(a.out) / nombre
        d.mkdir(parents=True, exist_ok=True)
        json.dump({"fps": FPS, "bones": rig.names, "trans": np.round(g["trans"], 6).tolist(),
                   "root_quat": np.round(matrix_to_quat(g["root_R"]), 7).tolist(), "euler": np.round(g["euler"], 6).tolist()},
                  open(d / "truth.json", "w"))
        tips = [rig.kp_names[j[-1]] for _, j in ar.legs]
        json.dump({"names": rig.kp_names, "joints": np.round(g["joints"], 5).tolist(), "tips": tips,
                   "legs": [ar.nombre_pata(i) for i in range(len(ar.legs))], "stance": g["stance"].astype(int).tolist(),
                   "h0": ar.h0.round(5).tolist(), "ik_err_mm_max": round(float(g["ik_err"].max() * 1000), 3),
                   **({"bucle": JUEGO[nombre]["bucle"], "eventos": JUEGO[nombre]["eventos"]} if a.juego else {})},
                  open(d / "truth_forja.json", "w"))
        J = g["joints"]
        zt = J[:, [rig.kp_names.index(n) for n in tips], 2] - ar.h0
        print(f"{nombre}: {len(J)} cuadros; IK de las puntas, error máx {g['ik_err'].max() * 1000:.2f} mm; "
              f"puntas bajo su apoyo: mín {zt.min() * 100:.2f} cm; apoyos {g['stance'].mean():.0%}; "
              f"altura mínima del resto {np.delete(J[..., 2], [rig.kp_names.index(n) for n in tips], 1).min() * 100:.1f} cm", flush=True)


if __name__ == "__main__":
    main()
