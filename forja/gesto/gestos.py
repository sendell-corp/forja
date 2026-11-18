"""Del gesto a metas por cuadro, basadas en el resultado (no en claves de pose).

«recoger caja»: acercarse dos pasos, agacharse (cadera y rodillas, espalda razonable), las dos palmas a los lados de la
caja, agarrar, levantar y quedar de pie con la caja delante. «dejar caja»: de pie con la caja, bajarla hasta el suelo,
soltar y volver a quedar de pie.

Lo único que se fija a mano son los **resultados**: dónde está la caja, a qué altura se agarra, dónde quedan los pies y
cuándo pasa cada cosa. Las poses clave del cuerpo (altura y retroceso de la pelvis, inclinación) **se buscan**: una
rejilla de candidatos resuelta en lote con la IK, y gana el que alcanza las metas con la espalda más recta, las rodillas
menos dobladas y el centro de masas mejor centrado sobre los pies (con la caja, si ya la lleva).
"""
import numpy as np

from forja.gesto.cuerpo import LADOS, marco

FPS = 30
CAJA = {"ancho": 0.36, "fondo": 0.26, "alto": 0.26, "masa_kg": 6.0}
MASA_CUERPO_KG = 70.0


def minjerk(u):
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 * (10 - 15 * u + 6 * u * u)


def curva(t, claves):
    """claves [(tiempo, valor)] → valores en t (T, …) con transiciones de mínimo tirón entre claves seguidas."""
    ts = np.array([k[0] for k in claves])
    vs = np.array([np.asarray(k[1], np.float64) for k in claves])
    out = np.empty((len(t),) + vs.shape[1:])
    for i, x in enumerate(t):
        j = np.searchsorted(ts, x, side="right") - 1
        if j < 0:
            out[i] = vs[0]
        elif j >= len(ts) - 1:
            out[i] = vs[-1]
        else:
            s = minjerk((x - ts[j]) / (ts[j + 1] - ts[j]))
            out[i] = vs[j] + (vs[j + 1] - vs[j]) * s
    return out


class Plan:
    """Metas por cuadro de un gesto (todo en Forja: Z arriba, el personaje mira a −Y)."""

    def __init__(self, cu, gesto, separacion=0.08, carga=0.30):
        self.cu, self.gesto = cu, gesto
        r, ix = cu.rig, cu.ix
        self.f, self.l, self.u = cu.adelante, cu.izquierda, np.array([0, 0, 1.0])
        self.tob0 = {s: r.rest[r.kp_names.index(f"foot_{s}")].copy() for s in LADOS}
        self.punta0 = {s: r.rest[r.kp_names.index(f"foot_{s}_tail")].copy() for s in LADOS}
        self.pel0 = r.B[0][:3, 3].copy()
        self.separacion = separacion
        self.caja = dict(CAJA)
        hz = self.caja["alto"] / 2
        punta_y = max(self.punta0[s][1] for s in LADOS) if self.f[1] > 0 else min(self.punta0[s][1] for s in LADOS)
        # la caja en el suelo, delante de las puntas de los pies (después de acercarse)
        self.caja_suelo = np.array([0.0, punta_y, 0.0]) + self.f * (separacion + self.caja["fondo"] / 2) + self.u * hz
        # la caja en brazos: delante del vientre, a la altura de la cintura alta
        self.caja_carga = np.array([0.0, 0.0, 0.86]) + self.f * carga
        self.dentro = {"L": -self.l, "R": self.l}
        self.ensanche = 0.03  # al acercarse, los pies se abren 3 cm cada uno (base para agacharse)

    # ---------------------------------------------------------------- metas de la mano sobre la caja
    def contacto(self, s, centro, altura=0.55):
        """Punto de la cara lateral de la caja (centro (…, 3)) donde apoya la palma del lado s."""
        lado = self.l if s == "L" else -self.l
        return centro + lado * self.caja["ancho"] / 2 + self.u * (altura - 0.5) * self.caja["alto"]

    def dedos_caja(self, T):
        d = -self.u + 0.35 * self.f
        return np.broadcast_to(d / np.linalg.norm(d), (T, 3)).copy()

    # ---------------------------------------------------------------- búsqueda de una pose clave
    def buscar(self, tob, centro_caja, lleva_caja, zs, dys, leans, mirar=None):
        """Rejilla gruesa (un paso de cada dos) y después una fina de 3×3×3 alrededor del mejor."""
        zs, dys, leans = (np.asarray(x, np.float64) for x in (zs, dys, leans))
        paso = [(x[1] - x[0]) if len(x) > 1 else 0.0 for x in (zs, dys, leans)]
        g = self._evaluar(tob, centro_caja, lleva_caja, zs[::2], dys[::2], leans[::2], mirar)
        fina = [np.clip(c + np.array([-1, 0, 1]) * p, x.min(), x.max()) if p else x
                for c, p, x in zip((g["z"], g["retroceso"], g["lean_deg"]), paso, (zs, dys, leans))]
        f = self._evaluar(tob, centro_caja, lleva_caja, np.unique(fina[0]), np.unique(fina[1]), np.unique(fina[2]), mirar)
        f["candidatos"] += g["candidatos"]
        return f

    def _evaluar(self, tob, centro_caja, lleva_caja, zs, dys, leans, mirar=None):
        cu = self.cu
        Z, D, A = np.meshgrid(zs, dys, np.radians(leans), indexing="ij")
        Z, D, A = Z.ravel(), D.ravel(), A.ravel()
        T = len(Z)
        pel = np.array([self.pel0[0], tob["L"][1] * 0 + (tob["L"][1] + tob["R"][1]) / 2 - self.pel0[1] * 0, 0.0])[None] \
            + np.stack([np.zeros(T), -self.f[1] * D, Z], 1)
        pel[:, 1] += self.pel0[1] - self.tob0["L"][1]  # la pelvis en reposo está sobre los tobillos
        metas = self._metas_manos(np.broadcast_to(centro_caja, (T, 3)))
        tobs = {s: np.broadcast_to(tob[s], (T, 3)).copy() for s in LADOS}
        cab = self._mirada(pel, A, np.broadcast_to(centro_caja if mirar is None else mirar, (T, 3)))
        trans, root_R, eul, err = cu.resolver(pel, A, cab, tobs, metas["muneca"], metas["dedos"],
                                              self._rodillas(pel, tobs), self._codos(pel, A, metas["muneca"]))
        _, P = cu.rig.forward(trans, root_R, eul)
        extra = (self.caja["masa_kg"] / MASA_CUERPO_KG, np.broadcast_to(centro_caja, (T, 3))) if lleva_caja else None
        c = cu.cdm(P, extra)
        # apoyo: del talón (tobillo) a la punta, en la dirección de avance
        ys = np.array([tob[s] @ self.f for s in LADOS] + [(tob[s] + self.f * 0.17) @ self.f for s in LADOS])
        lo, hi = ys.min(), ys.max()
        cy = c @ self.f
        margen = np.minimum(cy - lo, hi - cy)                       # > 0: dentro del apoyo
        rodilla = np.abs(eul[:, [cu.ix["shin_L"], cu.ix["shin_R"]], 0]).mean(1)
        e = np.maximum(err["muneca_L"], err["muneca_R"]) + np.maximum(err["tobillo_L"], err["tobillo_R"])
        coste = 1e5 * e ** 2 + 1.0 * A ** 2 + 20 * np.maximum(0, A - np.radians(48)) ** 2 + 0.5 * rodilla ** 2 + 300 * np.maximum(0, 0.05 - margen) ** 2
        b = int(coste.argmin())
        return {"pel": pel[b], "lean": float(A[b]), "z": float(Z[b]), "retroceso": float(D[b]),
                "err_mm": float(e[b] * 1000), "margen_cm": float(margen[b] * 100), "lean_deg": float(np.degrees(A[b])),
                "rodilla_deg": float(np.degrees(rodilla[b])), "candidatos": T}

    def _metas_manos(self, centro, abiertas=None):
        T = len(centro)
        dedos = self.dedos_caja(T)
        mun, dd = {}, {}
        for s in LADOS:
            p = self.contacto(s, centro)
            if abiertas is not None:
                lado = self.l if s == "L" else -self.l
                p = p + abiertas[:, None] * (lado * 0.06 + self.u * 0.03)
            mun[s] = self.cu.muneca_para(s, p, dedos, np.broadcast_to(self.dentro[s], (T, 3)))
            dd[s] = dedos
        return {"muneca": mun, "dedos": dd}

    def _rodillas(self, pel, tob):
        return {s: (pel + tob[s]) / 2 + self.f * 0.45 + (self.l if s == "L" else -self.l) * 0.10 for s in LADOS}

    def _codos(self, pel, lean, mun):
        out = {}
        for s in LADOS:
            lado = self.l if s == "L" else -self.l
            hombro = pel + self.u * 0.44 + lado * 0.18 + self.f * 0.44 * np.sin(np.asarray(lean))[:, None]
            out[s] = (hombro + mun[s]) / 2 + lado * 0.30 - self.f * 0.20
        return out

    def _mirada(self, pel, lean, objetivo):
        lean = np.asarray(lean)
        ojos = pel + self.u * 0.62 * np.cos(lean)[:, None] + self.f * 0.62 * np.sin(lean)[:, None]
        v = objetivo - ojos
        horiz = np.maximum((v * self.f).sum(-1), 0.05)
        phi = np.arctan2(-v[:, 2], horiz)          # > 0: mirar hacia abajo
        return np.clip(0.6 * phi - 0.8 * lean, -0.3, 0.55)  # los ojos hacen el resto: la cabeza gira solo parte

    def trayecto(self, t, t0, t1, a, b):
        """La caja entre el suelo y los brazos: cerca del suelo se mueve en vertical (lejos de las rodillas) y se acerca
        al cuerpo arriba, con un leve arco hacia delante. Vale para subir (a abajo) y para bajar (b abajo)."""
        u = np.clip((t - t0) / (t1 - t0), 0, 1)
        sube = a[2] < b[2]
        uu = u if sube else 1 - u
        lo, hi = (a, b) if sube else (b, a)
        sv = minjerk(np.clip(uu / 0.8, 0, 1))
        sh = minjerk(np.clip((uu - 0.3) / 0.7, 0, 1))
        p = lo + (hi - lo) * np.stack([sh, sh, sv], 1) + self.f * (0.04 * np.sin(np.pi * uu))[:, None]
        return p

    # ---------------------------------------------------------------- línea de tiempo
    def construir(self):
        return self._recoger() if self.gesto == "recoger" else self._dejar()

    def _relajadas(self, pel, t, balanceo=None):
        """Manos sueltas a los lados, colgando (con un leve balanceo opuesto a la pierna al andar)."""
        T = len(pel)
        out_m, out_d = {}, {}
        for k, s in enumerate(LADOS):
            lado = self.l if s == "L" else -self.l
            d = np.broadcast_to(-self.u + 0.08 * self.f, (T, 3)).copy()
            d /= np.linalg.norm(d, axis=-1, keepdims=True)
            p = pel + lado * 0.215 - self.u * 0.035 + self.f * 0.01
            if balanceo is not None:
                p = p + self.f * (balanceo if s == "L" else -balanceo)[:, None]
            out_m[s] = p
            out_d[s] = d
        return out_m, out_d

    def _pasos(self, t, inicio, pasos):
        """Tobillos (T, 3) por lado: plantados salvo en su paso [(lado, t0, t1, destino)], con 6 cm de altura."""
        tob = {s: np.repeat(inicio[s][None], len(t), 0) for s in LADOS}
        vuelo = {s: np.zeros(len(t), bool) for s in LADOS}
        for s, t0, t1, dest in pasos:
            u = np.clip((t - t0) / (t1 - t0), 0, 1)
            a = tob[s][np.searchsorted(t, t0)].copy()
            m = (t >= t0)
            tob[s][m] = a + (dest - a) * minjerk(u[m])[:, None] + self.u * (0.06 * np.sin(np.pi * u[m]))[:, None]
            vuelo[s] |= (t > t0) & (t < t1)
        return tob, vuelo

    def _recoger(self):
        cu = self.cu
        dur = 7.0
        t = np.arange(int(round(dur * FPS))) / FPS
        T = len(t)
        atras = 0.30
        ini = {s: self.tob0[s] - self.f * atras for s in LADOS}
        fin = {s: self.tob0[s] + (self.l if s == "L" else -self.l) * self.ensanche for s in LADOS}
        tob, vuelo = self._pasos(t, ini, [("L", 0.40, 0.95, fin["L"]), ("R", 1.05, 1.60, fin["R"])])
        # poses clave buscadas
        g = self.buscar(fin, self.caja_suelo, False, np.arange(0.36, 0.81, 0.04), np.arange(0.0, 0.29, 0.04),
                        np.arange(10, 76, 5))
        c = self.buscar(fin, self.caja_carga, True, np.arange(0.84, 0.905, 0.01), np.arange(-0.02, 0.11, 0.02),
                        np.arange(0, 16, 3))
        de_pie = self.pel0 + self.u * (-0.01)
        pel = curva(t, [(0.0, de_pie - self.f * atras), (0.35, de_pie - self.f * atras), (1.65, de_pie),
                        (1.9, de_pie), (3.05, g["pel"]), (3.6, g["pel"] - self.u * 0.008), (5.25, c["pel"]), (dur, c["pel"])])
        # balanceo lateral del peso sobre el pie de apoyo
        lat = curva(t, [(0, 0.0), (0.3, 0.0), (0.55, -0.025), (0.9, -0.025), (1.15, 0.025), (1.5, 0.025), (1.75, 0.0)])
        pel = pel + lat[:, None] * self.l
        pel = pel + self.u * (0.004 * np.sin(2 * np.pi * (t - 5.4) / 2.2) * (t > 5.4))[:, None]  # respira con la caja
        lean = curva(t, [(0.0, 0.0), (1.9, 0.0), (3.05, g["lean"]), (3.6, g["lean"]), (5.25, c["lean"]), (dur, c["lean"])])
        # caja planeada: quieta hasta levantarla, luego a la carga
        caja = self.trayecto(t, 3.6, 5.25, self.caja_suelo, self.caja_carga)
        caja[:, 2] += np.where(t > 5.4, 0.004 * np.sin(2 * np.pi * (t - 5.4) / 2.2), 0.0)
        agarre = 3.35
        abiertas = curva(t, [(0.0, 1.0), (3.05, 1.0), (agarre, 0.0), (dur, 0.0)])
        manos = self._metas_manos(caja, abiertas)
        bal = 0.05 * np.sin(np.pi * np.clip((t - 0.4) / 1.2, 0, 1) * 2) * (t < 1.7)
        rel_m, rel_d = self._relajadas(pel, t, bal)
        w = curva(t, [(0.0, 0.0), (1.85, 0.0), (3.05, 1.0), (dur, 1.0)])  # de sueltas a la caja
        mun = {s: rel_m[s] + (manos["muneca"][s] - rel_m[s]) * w[:, None] for s in LADOS}
        ded = {s: rel_d[s] + (manos["dedos"][s] - rel_d[s]) * w[:, None] for s in LADOS}
        mirar = curva(t, [(0.0, self.caja_suelo), (5.25, self.caja_suelo + self.u * 0.3),
                          (6.2, self.caja_carga + self.f * 3.0 + self.u * 0.5), (dur, self.caja_carga + self.f * 3.0 + self.u * 0.5)])
        eventos = {"agarre": int(round(agarre * FPS)), "suelta": None, "pasos": [[0.40, 0.95], [1.05, 1.60]],
                   "levanta": [3.6, 5.25]}
        return self._empaquetar(t, pel, lean, tob, vuelo, mun, ded, caja, mirar, eventos, {"agarre": g, "carga": c},
                                abiertas)

    def _dejar(self):
        dur = 6.5
        t = np.arange(int(round(dur * FPS))) / FPS
        fin = {s: self.tob0[s] + (self.l if s == "L" else -self.l) * self.ensanche for s in LADOS}
        tob = {s: np.repeat(fin[s][None], len(t), 0) for s in LADOS}
        vuelo = {s: np.zeros(len(t), bool) for s in LADOS}
        g = self.buscar(fin, self.caja_suelo, True, np.arange(0.36, 0.81, 0.04), np.arange(0.0, 0.29, 0.04),
                        np.arange(10, 76, 5))
        c = self.buscar(fin, self.caja_carga, True, np.arange(0.84, 0.905, 0.01), np.arange(-0.02, 0.11, 0.02),
                        np.arange(0, 16, 3))
        de_pie = self.pel0 + self.u * (-0.01)
        abajo = self.caja_suelo + self.u * 0.004   # se suelta a 4 mm del suelo: la cae la física
        pel = curva(t, [(0.0, c["pel"]), (0.6, c["pel"]), (2.3, g["pel"]), (2.9, g["pel"]), (4.4, de_pie), (dur, de_pie)])
        lean = curva(t, [(0.0, c["lean"]), (0.6, c["lean"]), (2.3, g["lean"]), (2.9, g["lean"]), (4.4, 0.0), (dur, 0.0)])
        caja = self.trayecto(t, 0.6, 2.3, self.caja_carga, abajo)
        suelta = 2.4
        abiertas = curva(t, [(0.0, 0.0), (suelta, 0.0), (2.75, 1.0), (dur, 1.0)])
        manos = self._metas_manos(caja, abiertas)
        rel_m, rel_d = self._relajadas(pel, t)
        w = curva(t, [(0.0, 1.0), (2.9, 1.0), (4.2, 0.0), (dur, 0.0)])
        mun = {s: rel_m[s] + (manos["muneca"][s] - rel_m[s]) * w[:, None] for s in LADOS}
        ded = {s: rel_d[s] + (manos["dedos"][s] - rel_d[s]) * w[:, None] for s in LADOS}
        mirar = curva(t, [(0.0, self.caja_carga + self.f * 3.0 + self.u * 0.5), (0.6, self.caja_suelo),
                          (3.6, self.caja_suelo), (4.6, self.caja_suelo + self.f * 3.0 + self.u * 1.2), (dur, self.caja_suelo + self.f * 3.0 + self.u * 1.2)])
        eventos = {"agarre": 0, "suelta": int(round(suelta * FPS)), "pasos": [], "baja": [0.6, 2.3]}
        return self._empaquetar(t, pel, lean, tob, vuelo, mun, ded, caja, mirar, eventos, {"agarre": g, "carga": c},
                                abiertas)

    def _empaquetar(self, t, pel, lean, tob, vuelo, mun, ded, caja, mirar, eventos, claves, abiertas):
        cu = self.cu
        T = len(t)
        R_manos = {s: cu.mano_R(s, ded[s], np.broadcast_to(self.dentro[s], (T, 3))) for s in LADOS}
        R_pies = {s: np.broadcast_to(cu.R_pie[s], (T, 3, 3)).copy() for s in LADOS}
        return {"t": t, "fps": FPS, "pel": pel, "lean": lean, "cabeza": self._mirada(pel, lean, mirar), "tobillos": tob,
                "vuelo": vuelo, "munecas": mun, "dedos": ded, "R_manos": R_manos, "R_pies": R_pies,
                "rodillas": self._rodillas(pel, tob), "codos": self._codos(pel, lean, mun), "caja": caja,
                "caja_medidas": self.caja, "eventos": eventos, "claves": claves, "abiertas": abiertas,
                "en_brazos": caja[:, 2] > 0.6}
