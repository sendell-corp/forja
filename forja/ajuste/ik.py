"""IK por cadena (una pata, un colmillo, el abdomen) con los límites articulares del rig.

Las incógnitas son los ángulos Euler XYZ de cada hueso de la cadena, acotados a sus límites: así las **longitudes de
hueso quedan fijas** por construcción (es la cinemática directa del rig) y la pose siempre es una pose válida del juego.

Se resuelve con Levenberg–Marquardt acotado (conjunto activo: un ángulo en su límite que quiere salirse queda fijo en
ese paso), en lote para todos los cuadros y para varios arranques a la vez, con jacobiano por diferencias centrales.
Los residuos, todos en píxeles equivalentes:
- **datos:** L·(p_rig − p_objetivo) por articulación, con LᵀL = Λ (la información de la triangulación: un eje que
  ninguna vista mide no cuenta) y un peso robusto de Cauchy que se reajusta en pasadas (σ de 30 a 6 px);
- **previo:** √wp·(θ − θ_previo) (el cuadro vecino o la curva suavizada) y √wr·θ (hacia el reposo, muy débil);
- **suelo:** √wg·max(0, z_min − z)/M_PER_PX para cada articulación de la cadena (una sola cara: solo empuja hacia arriba).
"""
import numpy as np

from forja.rig import euler_from_matrix, euler_xyz


def raiz_info(Lam):
    """Factor L (…, 3, 3) con LᵀL = Λ, válido aunque Λ sea singular (ejes sin medir)."""
    ev, U = np.linalg.eigh(Lam)
    return np.sqrt(np.maximum(ev, 0.0))[..., :, None] * np.swapaxes(U, -1, -2)


def rot_between(a, b):
    """Rotación mínima (…, 3, 3) que lleva los vectores unitarios a → b."""
    v = np.cross(a, b)
    c = (a * b).sum(-1)
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -v[..., 2], v[..., 1], v[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -v[..., 0], -v[..., 1], v[..., 0]
    R = np.eye(3) + K + K @ K / np.maximum(1 + c, 1e-6)[..., None, None]
    flip = c < -0.9999
    if flip.any():
        R[flip] = np.diag([1.0, -1.0, -1.0])
    return R


class CadenaIK:
    def __init__(self, rig, bones, M_PER_PX, malla=None):
        """malla: (vértices de reposo (V, 3), pesos (V, n) sobre los huesos de la cadena) de la parte baja de la pata
        (modelo/garras.py). Si se da, el término de suelo también les exige z ≥ 0 con la piel lineal exacta."""
        self.rig, self.bones, self.n = rig, list(bones), len(bones)
        self.P = 3 * self.n
        self.lo = rig.lo[self.bones].ravel()
        self.hi = rig.hi[self.bones].ravel()
        self.M = M_PER_PX
        self.malla = None
        if malla is not None:
            v, w = (np.asarray(x, np.float64) for x in malla)
            self.malla = (np.concatenate([v, np.ones((len(v), 1))], 1), w, np.linalg.inv(rig.B[self.bones]))

    def fk(self, Mp, th):
        """Mp (…, 4, 4), th (…, P) → puntos (…, n + 1, 3): la cabeza del primer hueso y las colas."""
        _, pts = self.rig.chain_forward(Mp, self.bones, th.reshape(th.shape[:-1] + (self.n, 3)))
        return pts

    def fk_malla(self, Mp, th):
        """Puntos (…, n + 1, 3) y la altura z de la parte baja de la malla deformada (…, V) (o None).
        Solo hace falta la fila z de cada matriz de piel: z_v = Σ_k w_vk · (M_k·B_k⁻¹)[2]·v."""
        Ms, pts = self.rig.chain_forward(Mp, self.bones, th.reshape(th.shape[:-1] + (self.n, 3)))
        if self.malla is None:
            return pts, None
        vh, w, invB = self.malla
        Zrow = (Ms @ invB)[..., 2, :]  # (…, n, 4)
        Z = Zrow @ vh.T                # (…, n, V)
        return pts, (Z * w.T).sum(-2)

    def inicio_geometrico(self, Mp, tgt):
        """Cada hueso apunta a su objetivo, sin giro sobre su eje y dentro de sus límites (como el ik_init de anymove)."""
        T = len(Mp)
        th = np.zeros((T, self.n, 3))
        M = Mp
        y = np.array([0.0, 1.0, 0.0])
        for k, b in enumerate(self.bones):
            base = M @ self.rig.off[b]
            d = tgt[:, k] - base[:, :3, 3]
            nd = np.linalg.norm(d, axis=-1, keepdims=True)
            loc = np.einsum("tji,tj->ti", base[:, :3, :3], d / np.maximum(nd, 1e-9))
            e = euler_from_matrix(rot_between(np.broadcast_to(y, loc.shape), loc))
            e = np.clip(e, self.rig.lo[b], self.rig.hi[b])
            e = np.where(nd > 1e-6, e, 0.0)
            th[:, k] = e
            basis = np.zeros((T, 4, 4))
            basis[:, :3, :3] = euler_xyz(e)
            basis[:, 3, 3] = 1
            M = base @ basis
        return th.reshape(T, self.P)

    def _res(self, Mp, th, Ao, yo, sw, prior, swp, swr, zmin, swg):
        pts, low = self.fk_malla(Mp, th) if zmin is not None else (self.fk(Mp, th), None)
        pred = np.einsum("...kgij,...kj->...kgi", Ao, pts[..., 1:, :])
        r = [((pred - yo) * sw[..., None]).reshape(th.shape[:-1] + (-1,))]
        if prior is not None:
            r.append(swp[..., None] * (th - prior))
        r.append(swr * th)
        if zmin is not None:
            r.append(swg * np.maximum(0.0, zmin - pts[..., 1:, 2]) / self.M)
            if low is not None:  # la malla baja de la pata: z ≥ 0, con el peso del suelo de la punta
                r.append(swg[..., -1:] * np.maximum(0.0, -low) / self.M)
        return np.concatenate(r, -1), pts

    def resolver(self, Mp, tgt, Lam, starts, **kw):
        """Objetivos 3D con su información Λ (T, n, 3, 3): un solo grupo de observación por articulación."""
        return self.resolver_obs(Mp, obs_puntos(tgt, Lam), starts, **kw)

    def resolver_obs(self, Mp, obs, starts, *, prior=None, wp=0.0, wr=2.0, zmin=None, wg=0.0, iters=30,
                     sigmas=(30.0, 12.0, 6.0), h=1e-4):
        """Mp (T, 4, 4) del padre; obs = (A (T, n, G, 3, 3), y (T, n, G, 3), w (T, n, G)): por articulación de la cadena,
        G grupos de observaciones lineales A·p ≈ y en px (una vista: dos filas; un punto 3D: tres) con peso w y un peso
        robusto de Cauchy propio, así una vista mala de una articulación no arrastra a las demás. starts (S, T, P).
        prior (T, P) o None; wp escalar o (T,); zmin (n,) o None; wg escalar o (T, n).
        Devuelve θ (T, P) del mejor arranque por cuadro, su coste robusto (T,) y el error por articulación (T, n) en px."""
        Ao, yo, wo = obs
        S, T = starts.shape[:2]
        th = np.clip(starts, self.lo, self.hi)
        Mp_ = np.broadcast_to(Mp, (S,) + Mp.shape)
        swp = np.broadcast_to(np.sqrt(np.asarray(wp, np.float64)) * np.ones(T), (S, T))
        swg = np.broadcast_to(np.sqrt(np.asarray(wg, np.float64)) * np.ones((T, self.n)), (S, T, self.n))
        swr = np.sqrt(wr)
        sw = np.broadcast_to(np.sqrt(wo), (S,) + wo.shape).copy()
        eye = np.eye(self.P)

        def args(extra=False):
            """Entradas alineadas con θ (S, T, P); con extra=True, con un eje más tras (S, T) para las perturbaciones."""
            if not extra:
                return (Mp_, Ao, yo, sw, prior, swp, swr, zmin, swg)
            return (Mp_[:, :, None], Ao[:, None], yo[:, None], sw[:, :, None], None if prior is None else prior[:, None],
                    swp[..., None], swr, zmin, swg[:, :, None])

        def errores(th_):
            pts = self.fk(Mp_, th_)
            return np.linalg.norm(np.einsum("...kgij,...kj->...kgi", Ao, pts[..., 1:, :]) - yo, axis=-1), pts

        for sigma in sigmas:
            mu = np.full((S, T), 1e-2)
            r, _ = self._res(args()[0], th, *args()[1:])
            c0 = (r ** 2).sum(-1)
            for _ in range(iters):
                # jacobiano por diferencias centrales, todos los parámetros en un solo lote
                pert = th[..., None, :] + np.concatenate([h * eye, -h * eye])[None, None]  # (S, T, 2P, P)
                a = args(True)
                rp, _ = self._res(a[0], pert, *a[1:])
                Jm = (rp[..., :self.P, :] - rp[..., self.P:, :]) / (2 * h)  # (S, T, P, m)
                g = np.einsum("stpm,stm->stp", Jm, r)
                H = np.einsum("stpm,stqm->stpq", Jm, Jm)
                fixed = ((th <= self.lo + 1e-9) & (g > 0)) | ((th >= self.hi - 1e-9) & (g < 0))
                g = np.where(fixed, 0.0, g)
                H = np.where(fixed[..., :, None] | fixed[..., None, :], 0.0, H)
                Hd = H + (mu[..., None, None] * (np.diagonal(H, axis1=-2, axis2=-1)[..., None] * eye + 1e-6 * eye)) \
                    + fixed[..., None] * eye
                step = -np.linalg.solve(Hd, g[..., None])[..., 0]
                th_new = np.clip(th + step, self.lo, self.hi)
                r_new, _ = self._res(args()[0], th_new, *args()[1:])
                c1 = (r_new ** 2).sum(-1)
                ok = c1 < c0
                th = np.where(ok[..., None], th_new, th)
                r = np.where(ok[..., None], r_new, r)
                c0 = np.where(ok, c1, c0)
                mu = np.where(ok, mu * 0.3, mu * 10.0).clip(1e-7, 1e7)
            # pesos robustos (Cauchy) por articulación y grupo (vista) según su error en px
            e, _ = errores(th)
            sw = np.sqrt(wo / (1.0 + (e / sigma) ** 2))
        # coste comparable entre arranques: Cauchy de los datos + previo + suelo
        e, pts = errores(th)
        sigma = sigmas[-1]
        cost = (wo * sigma ** 2 * np.log1p((e / sigma) ** 2)).sum((-1, -2))
        if prior is not None:
            cost += ((swp[..., None] * (th - prior)) ** 2).sum(-1)
        cost += ((swr * th) ** 2).sum(-1)
        if zmin is not None:
            cost += ((swg * np.maximum(0.0, zmin - pts[..., 1:, 2]) / self.M) ** 2).sum(-1)
            _, low = self.fk_malla(Mp_, th)
            if low is not None:
                cost += ((swg[..., -1:] * np.maximum(0.0, -low) / self.M) ** 2).sum(-1)
        best = cost.argmin(0)
        ti = np.arange(T)
        ej = np.sqrt((wo * e[best, ti] ** 2).sum(-1) / np.maximum(wo.sum(-1), 1e-12))  # error por articulación (px)
        return th[best, ti], cost[best, ti], ej


def obs_puntos(tgt, Lam):
    """Objetivos 3D (T, n, 3) con información Λ → un grupo de observación por articulación (G = 1)."""
    Lc = raiz_info(Lam)
    T, n = tgt.shape[:2]
    w = (np.abs(Lam).sum((-1, -2)) > 0).astype(np.float64)[..., None]  # sin información: no cuenta
    return Lc[:, :, None], np.einsum("tkij,tkj->tki", Lc, tgt)[:, :, None], w


def obs_vistas(uv, w, A, b, joints):
    """Lecturas por vista uv (V, T, J, 2) con peso w (V, T, J), cámaras afines A (V, 2, 3), b (V, 2) → un grupo por vista
    para las articulaciones `joints` (G = V; la tercera fila de cada grupo es cero)."""
    V, T = w.shape[:2]
    n = len(joints)
    Ao = np.zeros((T, n, V, 3, 3))
    yo = np.zeros((T, n, V, 3))
    Ao[..., :2, :] = A[None, None]
    yo[..., :2] = np.moveaxis(uv[:, :, joints] - b[:, None, None], 0, 2)
    return Ao, yo, np.moveaxis(w[:, :, joints], 0, 2).copy()


def obs_unir(*obs):
    """Junta grupos de observación (por el eje G)."""
    return tuple(np.concatenate([o[i] for o in obs], 2) for i in range(3))
