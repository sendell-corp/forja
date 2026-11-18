"""La raíz (cefalotórax de la araña): rotación y traslación por cuadro a partir de los puntos que viajan rígidos con ella.

1. Kabsch ponderado (isótropo, con el menor autovalor de Λ como peso: un punto con un eje sin medir no cuenta) y
   reponderado con Cauchy, para que una lectura mala de una cadera no tuerza el cuerpo.
2. Gauss–Newton con la información completa Λ de cada punto (sí usa los ejes medidos de un punto visto en una vista).
3. Suavizado sin aplanar golpes (suavizado.py) de la traslación y del vector de rotación.
Convierte al formato del rig de anymove: root_R = B0ᵀ·R_mundo·B0 y trans = t + R_mundo·b0 − b0 (b0, cabeza de la raíz).
"""
import numpy as np

from forja.ajuste.ik import raiz_info
from forja.ajuste.suavizado import suavizar
from forja.rig import axis_angle_matrix


def kabsch(q, p, w):
    """q (K, 3) reposo, p (T, K, 3) observados, w (T, K) → R (T, 3, 3), t (T, 3) con p ≈ R q + t."""
    ws = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
    cq = np.einsum("tk,kj->tj", ws, q)
    cp = np.einsum("tk,tkj->tj", ws, p)
    H = np.einsum("tk,tki,tkj->tij", ws, q[None] - cq[:, None], p - cp[:, None])
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.swapaxes(Vt, 1, 2) @ np.swapaxes(U, 1, 2)))
    D = np.zeros_like(H)
    D[:, 0, 0], D[:, 1, 1], D[:, 2, 2] = 1, 1, d
    R = np.swapaxes(Vt, 1, 2) @ D @ np.swapaxes(U, 1, 2)
    return R, cp - np.einsum("tij,tj->ti", R, cq)


def log_rot(R):
    """(T, 3, 3) → vectores eje·ángulo (T, 3)."""
    c = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    ang = np.arccos(c)
    ax = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    s = np.sin(ang)
    return np.where(s[:, None] > 1e-9, ax * (ang / (2 * np.maximum(s, 1e-12)))[:, None], ax / 2)


def ajustar_raiz(rig, P, Lam, M_PER_PX, *, vistas=None, sigma_px=6.0, gn_iters=8, suave=(2.0, 0.004), perdida="huber", log=None):
    """P (T, J, 3), Λ (T, J, 3, 3) → trans (T, 3), root_R (T, 3, 3) en el formato del rig, y el error por punto (T, K) px.
    vistas = (uv (V, T, J, 2), w (V, T, J), A (V, 2, 3), b (V, 2)): el refinado usa las lecturas de cada vista con un
    peso robusto por punto y vista (si no, los puntos triangulados con su Λ)."""
    idx = rig.rigid_on_root()
    q = rig.rest[idx]
    p = P[:, idx]
    L = Lam[:, idx]
    Lc = raiz_info(L)
    if vistas is not None:
        uvv, wv, Av, bv = vistas
        uvk = np.moveaxis(uvv[:, :, idx], 0, 2)  # (T, K, V, 2)
        wk = np.moveaxis(wv[:, :, idx], 0, 2)    # (T, K, V)
    wmin = np.linalg.eigvalsh(L)[..., 0] * M_PER_PX ** 2  # ~conf² del eje peor medido
    w = wmin.copy()
    for _ in range(4):
        R, t = kabsch(q, p, np.maximum(w, 1e-9))
        e = np.linalg.norm(np.einsum("tkij,tkj->tki", Lc, np.einsum("tij,kj->tki", R, q) + t[:, None] - p), axis=-1)
        w = wmin / (1 + (e / sigma_px) ** 2)
    # Gauss–Newton en (rotación, traslación) con pesos de Cauchy: por punto (Λ completa) o por punto y vista
    for _ in range(gn_iters):
        def res(dr, dt):
            Rn = axis_angle_matrix(dr) @ R
            x = np.einsum("tij,kj->tki", Rn, q) + (t + dt)[:, None]
            if vistas is not None:
                return np.einsum("vij,tkj->tkvi", Av, x) + bv - uvk, Rn  # (T, K, V, 2) px
            return np.einsum("tkij,tkj->tki", Lc, x - p), Rn
        r0, _ = res(np.zeros_like(t), np.zeros_like(t))
        e = np.linalg.norm(r0, axis=-1)
        sw = np.sqrt((wk if vistas is not None else 1.0) / (1 + (e / sigma_px) ** 2))[..., None]
        h = 1e-6
        cols = []
        for k in range(6):
            dv = np.zeros((len(t), 6))
            dv[:, k] = h
            rp, _ = res(dv[:, :3], dv[:, 3:])
            cols.append(((rp - r0) * sw).reshape(len(t), -1) / h)
        Jm = np.stack(cols, -1)  # (T, filas, 6)
        rr = (r0 * sw).reshape(len(t), -1)
        Hm = np.einsum("tmi,tmj->tij", Jm, Jm) + 1e-6 * np.eye(6)
        step = -np.linalg.solve(Hm, np.einsum("tmi,tm->ti", Jm, rr)[..., None])[..., 0]
        R = axis_angle_matrix(step[:, :3]) @ R
        t = t + step[:, 3:]
    e = np.linalg.norm(np.einsum("tkij,tkj->tki", Lc, np.einsum("tij,kj->tki", R, q) + t[:, None] - p), axis=-1)
    # cuadros sin información suficiente en algún eje (p. ej. todas las lecturas de una vista descartadas): se
    # rellenan interpolando a sus vecinos en lugar de fiarse de un ajuste que no ve ese eje
    info = np.linalg.eigvalsh((L * (1 / (1 + (e / sigma_px) ** 2))[..., None, None]).sum(1))[:, 0]
    ok = info > 0.05 * np.median(info)
    if not ok.all() and ok.sum() >= 2:
        from forja.ajuste.suavizado import rellenar
        R0 = R[np.argmax(ok)]
        v = rellenar(log_rot(np.einsum("ij,tjk->tik", R0.T, R)), ok)
        R = R0[None] @ axis_angle_matrix(v)
        t = rellenar(t, ok)
        if log:
            log(f"raíz: {int((~ok).sum())} cuadros sin información suficiente, interpolados")
    if suave:
        lam, delta = suave
        # rotación: vector relativo a la rotación media (ángulos pequeños, sin saltos de 2π)
        R0 = R[len(R) // 2]
        v = log_rot(np.einsum("ij,tjk->tik", R0.T, R))
        v = suavizar(v, None, lam, delta * 3, perdida=perdida)
        R = R0[None] @ axis_angle_matrix(v)
        t = suavizar(t, None, lam, delta, perdida=perdida)
    B0r, b0 = rig.B[0, :3, :3], rig.B[0, :3, 3]
    root_R = np.einsum("ji,tjk,kl->til", B0r, R, B0r)
    trans = t + np.einsum("tij,j->ti", R, b0) - b0
    if log:
        log(f"raíz: error de los {len(idx)} puntos rígidos {np.median(e):.2f} px de mediana, p95 {np.percentile(e, 95):.2f} px")
    return trans, root_R, e
