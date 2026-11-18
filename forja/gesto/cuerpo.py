"""El cuerpo entero a partir de metas: pelvis e inclinación, pies plantados, muñecas y orientación de pies y manos.

- Pelvis: posición de mundo y una inclinación hacia delante `lean` (bisagra de cadera). La inclinación se reparte:
  55 % la pelvis, 25 % `spine` y 20 % `chest`, así la espalda se dobla poco y sobre todo bisagra la cadera.
- Cuello y cabeza: `cabeza` (rad) de flexión extra, 40 % cuello y 60 % cabeza, para mirar la caja.
- Piernas y brazos: la IK de Forja (`ajuste/ik.py`, CadenaIK con límites articulares) en cadenas de dos huesos
  (muslo-tibia, brazo-antebrazo): el tobillo y la muñeca son metas fuertes; la rodilla y el codo, metas débiles que
  solo eligen el plano (rodillas hacia delante, codos hacia fuera).
- Pie y mano: la rotación de mundo se impone de forma analítica (el pie plano y en su orientación de apoyo; la mano con
  los dedos en su dirección y la palma hacia dentro), recortada a sus límites.
Todo en lote: T puede ser una línea de tiempo o una rejilla de candidatos.
"""
import numpy as np

from forja.ajuste.ik import CadenaIK
from forja.rig import euler_from_matrix, euler_xyz

M_PER_PX = 0.002          # 1 «px» de la IK = 2 mm
REPARTO = {"pelvis": 0.55, "spine": 0.25, "chest": 0.20}
LADOS = ("L", "R")


def rot_x(a):
    a = np.asarray(a, np.float64)
    R = np.zeros(a.shape + (3, 3))
    c, s = np.cos(a), np.sin(a)
    R[..., 0, 0] = 1
    R[..., 1, 1], R[..., 1, 2], R[..., 2, 1], R[..., 2, 2] = c, -s, s, c
    return R


def marco(y, x_aprox):
    """Rotación (…, 3, 3) con columnas X, Y, Z: Y = y, X lo más parecido a x_aprox y ortogonal, Z = X × Y."""
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    x = x_aprox - (x_aprox * y).sum(-1, keepdims=True) * y
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    z = np.cross(x, y)
    return np.stack([x, y, z], -1)


class Cuerpo:
    def __init__(self, h):
        self.h = h
        r = self.rig = h.rig
        self.ix = {n: i for i, n in enumerate(r.names)}
        self.nb = len(r.names)
        self.adelante = h.adelante()
        self.adelante = np.array([0.0, np.sign(self.adelante[1]) or -1.0, 0.0]) if abs(self.adelante[1]) > abs(self.adelante[0]) \
            else np.array([np.sign(self.adelante[0]), 0.0, 0.0])
        if self.adelante[1] >= 0:
            raise SystemExit("por ahora el personaje tiene que mirar a +Z de glTF (−Y de Forja), como los del juego")
        self.izquierda = np.cross(np.array([0, 0, 1.0]), self.adelante)  # mirando a −Y, la izquierda es +X
        self.piernas = {s: CadenaIK(r, [self.ix[f"thigh_{s}"], self.ix[f"shin_{s}"]], M_PER_PX) for s in LADOS}
        self.brazos = {s: CadenaIK(r, [self.ix[f"upper_arm_{s}"], self.ix[f"forearm_{s}"]], M_PER_PX) for s in LADOS}
        self.B0 = r.B[0]
        # orientación de apoyo de cada pie (la del reposo, plano en el suelo)
        self.R_pie = {s: r.B[self.ix[f"foot_{s}"]][:3, :3].copy() for s in LADOS}
        # la palma: eje local ±X de la mano que mira hacia dentro en el reposo, y el punto de contacto de la mano
        self.palma = {}
        dom = h.VJ[np.arange(len(h.V)), h.VW.argmax(1)]
        for s in LADOS:
            k = self.ix[f"hand_{s}"]
            B = r.B[k]
            dentro = -self.izquierda if s == "L" else self.izquierda
            sx = 1.0 if B[:3, 0] @ dentro > 0 else -1.0
            sk = h.skin_joints.index(h.rig_nodes[k])
            loc = (h.V_forja[dom == sk] - B[:3, 3]) @ B[:3, :3]
            xm = (loc[:, 0] * sx).max()
            cerca = loc[(loc[:, 0] * sx) > xm - 0.012]
            self.palma[s] = {"signo_x": sx, "punto": np.array([xm * sx, cerca[:, 1].mean(), cerca[:, 2].mean()])}
        # masas de los segmentos (fracciones de Winter), centradas entre la cabeza y la cola de cada hueso
        frac = {"pelvis": 0.142, "spine": 0.139, "chest": 0.216, "neck": 0.02, "head": 0.061, "upper_arm": 0.028,
                "forearm": 0.016, "hand": 0.006, "thigh": 0.1, "shin": 0.0465, "foot": 0.0145}
        self.masa = np.array([frac.get(n.rsplit("_", 1)[0] if n.endswith(("_L", "_R")) else n, 0.0) for n in r.names])
        self.masa /= self.masa.sum()

    # ---------------------------------------------------------------- piezas
    def raiz(self, pel, lean):
        Rw = rot_x(np.asarray(lean) * REPARTO["pelvis"] * np.sign(-self.adelante[1]))
        B0R = self.B0[:3, :3]
        root_R = np.einsum("ji,tjk,kl->til", B0R, Rw, B0R)
        return pel - self.B0[:3, 3], root_R

    def mano_R(self, s, dedos, dentro):
        """Rotación de mundo de la mano: Y = dirección de los dedos, ±X (la palma) hacia `dentro`."""
        return marco(dedos, dentro * self.palma[s]["signo_x"])

    def muneca_para(self, s, contacto, dedos, dentro):
        """Dónde poner la muñeca para que el punto de contacto de la palma caiga en `contacto`."""
        R = self.mano_R(s, dedos, dentro)
        return contacto - np.einsum("tij,j->ti", R, self.palma[s]["punto"])

    def resolver(self, pel, lean, cabeza, tobillos, munecas, dedos, rodillas, codos, prior=None, wp=0.0, iters=25):
        """Todas (T, …). tobillos/munecas/dedos/rodillas/codos: {lado: (T, 3)}. Devuelve trans, root_R, eul (T, nb, 3),
        J, P y el error (m) de tobillos y muñecas por lado."""
        r, ix = self.rig, self.ix
        T = len(pel)
        trans, root_R = self.raiz(pel, lean)
        eul = np.zeros((T, self.nb, 3))
        lean = np.asarray(lean)
        eul[:, ix["spine"], 0] = lean * REPARTO["spine"]
        eul[:, ix["chest"], 0] = lean * REPARTO["chest"]
        eul[:, ix["neck"], 0] = np.asarray(cabeza) * 0.4
        eul[:, ix["head"], 0] = np.asarray(cabeza) * 0.6
        for n in ("spine", "chest", "neck", "head"):
            eul[:, ix[n]] = np.clip(eul[:, ix[n]], r.lo[ix[n]], r.hi[ix[n]])
        _, P = r.forward(trans, root_R, eul)
        err = {}
        fuerte = np.eye(3) / M_PER_PX ** 2   # 1 px de residuo = 2 mm
        debil = np.eye(3) / 0.15 ** 2        # 1 px = 15 cm: solo orienta el plano de la rodilla o del codo
        for cad, padre, meta, polo, nombre in (
                [(self.piernas[s], "pelvis", tobillos[s], rodillas[s], f"tobillo_{s}") for s in LADOS] +
                [(self.brazos[s], "chest", munecas[s], codos[s], f"muneca_{s}") for s in LADOS]):
            Mp = P[:, ix[padre]]
            meta0 = meta
            if padre == "chest":  # la muñeca, al alcance del hombro (el brazo se estira hacia una meta lejana)
                hombro = (Mp @ r.off[cad.bones[0]])[:, :3, 3]
                alcance = 0.985 * (r.L[cad.bones[0]] + r.L[cad.bones[1]])
                v = meta - hombro
                n = np.linalg.norm(v, axis=-1, keepdims=True)
                meta = hombro + v * np.minimum(1.0, alcance / np.maximum(n, 1e-9))
            tgt = np.stack([polo, meta], 1)
            Lam = np.zeros((T, 2, 3, 3))
            Lam[:, 0] = debil
            Lam[:, 1] = fuerte
            starts = [cad.inicio_geometrico(Mp, tgt)]
            pr = None
            if prior is not None:
                pr = prior[:, cad.bones].reshape(T, -1)
                starts.append(pr)
            th, _, _ = cad.resolver(Mp, tgt, Lam, np.stack(starts), prior=pr, wp=wp, wr=0.5, iters=iters,
                                    sigmas=(1e6,))
            eul[:, cad.bones] = th.reshape(T, 2, 3)
            pts = cad.fk(Mp, th)
            err[nombre] = np.linalg.norm(pts[:, -1] - meta0, axis=-1)  # contra la meta pedida
        return trans, root_R, eul, err

    def orientar(self, trans, root_R, eul, R_pies, R_manos):
        """Pies y manos con su rotación de mundo (T, 3, 3) por lado, recortada a los límites del hueso."""
        r, ix = self.rig, self.ix
        _, P = r.forward(trans, root_R, eul)
        for hueso, padre, Rdes in ([(f"foot_{s}", f"shin_{s}", R_pies[s]) for s in LADOS] +
                                   [(f"hand_{s}", f"forearm_{s}", R_manos[s]) for s in LADOS]):
            k = ix[hueso]
            A = (P[:, ix[padre]] @ r.off[k])[:, :3, :3]
            e = euler_from_matrix(np.einsum("tji,tjk->tik", A, Rdes))
            eul[:, k] = np.clip(e, r.lo[k], r.hi[k])
        return eul

    def cdm(self, P, extra=None):
        """Centro de masas (T, 3) del cuerpo (segmentos de Winter); extra = (masa relativa, (T, 3)) para la caja."""
        r = self.rig
        cab = P[:, :, :3, 3]
        cola = cab + P[:, :, :3, 1] * r.L[None, :, None]
        c = np.einsum("b,tbi->ti", self.masa, (cab + cola) / 2)
        if extra is not None:
            m, p = extra
            c = (c + m * p) / (1 + m)
        return c
