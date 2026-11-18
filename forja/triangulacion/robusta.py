"""Triangulación robusta de N vistas afines (uv = A·p + b).

1. **Mínimos cuadrados ponderados por confianza.** Para un punto y un conjunto de lecturas, p minimiza
   Σ_v w_v ‖A_v p + b_v − uv_v‖² con w_v = conf_v². La matriz de información Λ = Σ_v w_v A_vᵀA_v (px²/m²) dice qué
   ejes quedan medidos: con dos vistas ortográficas (frente y lado) la X la da solo el frente, la Y solo el lado y la Z
   las dos; con la vista de arriba, cada eje lo miden dos vistas y una lectura mala se delata sola.
2. **Candidatos por vista.** En cada vista, cada articulación puede usar su lectura, la de su **gemela** en esa vista
   (la pata que se le superpone: en el lado, la pata simétrica; en el frente, la de delante o detrás del mismo lado) o
   **ninguna** (lectura descartada). Las gemelas salen solas de la pose de reposo (`gemelos`).
3. **Selección temporal (Viterbi).** Cada combinación tiene un coste propio (desacuerdo entre vistas, descartes,
   cambios de etiqueta, y opcionalmente quedar bajo el suelo) y un coste de transición entre cuadros (cuánto salta
   la lectura elegida en cada imagen). El camino de coste mínimo resuelve qué lecturas son buenas y, en la marcha, qué
   patas están arriba: una pata no sube y baja de un cuadro al siguiente.
"""
import itertools

import numpy as np


def gemelos(rig, lay, umbral_px=45.0):
    """Por vista, para cada articulación, las articulaciones de otras cadenas que se le superponen en la pose de reposo
    (misma posición dentro de su cadena y a menos de `umbral_px` de media a lo largo de la cadena).
    Devuelve {vista: {j: [j2, …]}}."""
    chains = [js for _, js in rig.chains()]
    P = lay.project(rig.rest)  # (V, J, 2)
    out = {}
    for vi, v in enumerate(lay.views):
        g = {}
        for a, b in itertools.combinations(range(len(chains)), 2):
            ca, cb = chains[a], chains[b]
            if len(ca) != len(cb):
                continue
            d = np.linalg.norm(P[vi, ca[1:]] - P[vi, cb[1:]], axis=-1).mean()  # sin la cabeza (rígida con la raíz)
            if d < umbral_px:
                for ja, jb in zip(ca, cb):
                    g.setdefault(ja, []).append(jb)
                    g.setdefault(jb, []).append(ja)
        out[v.name] = g
    return out


def _wls(uv, w, A, b):
    """uv (…, V, 2), w (…, V) → p (…, 3), Λ (…, 3, 3), residuo ponderado Σ w‖r‖² (…) y residuos por vista (…, V)."""
    AtA = np.einsum("vij,vik->vjk", A, A)  # (V, 3, 3)
    Lam = np.einsum("...v,vjk->...jk", w, AtA)
    rhs = np.einsum("...v,vij,...vi->...j", w, A, uv - b)
    # pseudo-inversa: los ejes que ninguna vista elegida mide quedan en 0 (y con información 0)
    ev, U = np.linalg.eigh(Lam)
    inv = np.where(ev > 1e-6 * max(ev.max(), 1e-12), 1.0 / np.maximum(ev, 1e-30), 0.0)
    p = np.einsum("...ij,...j,...kj,...k->...i", U, inv, U, rhs)
    r = np.einsum("vij,...j->...vi", A, p) + b - uv
    rv = np.linalg.norm(r, axis=-1)
    return p, Lam, (w * rv ** 2).sum(-1), rv


def triangular(uv, conf, A, b, wmin=0.05):
    """Triangulación directa con todas las lecturas: uv (V, T, J, 2), conf (V, T, J) → p (T, J, 3), Λ, residuos (V, T, J)."""
    w = np.maximum(conf, wmin) ** 2
    p, Lam, _, rv = _wls(np.moveaxis(uv, 0, -2), np.moveaxis(w, 0, -1), A, b)
    return p, Lam, np.moveaxis(rv, -1, 0)


def seleccionar(uv, conf, A, b, twins, vistas, *, sigma_r=4.0, cap_r=25.0, pen_descarte=4.0, pen_gemelo=1.5,
                conf_ref=0.5, sigma_t=12.0, cap_t=9.0, tr_descarte=2.0, suelo=None, sigma_suelo=0.02, wmin=0.05, log=None):
    """Elige por Viterbi, para cada articulación y cuadro, qué lectura usar en cada vista (propia, de una gemela o
    ninguna).

    uv (V, T, J, 2), conf (V, T, J); twins = gemelos(); vistas: nombres en el orden de V.
    suelo: (J,) altura mínima de cada articulación (o None): un candidato por debajo paga ((h − z)/sigma_suelo)².
    Devuelve p (T, J, 3), Λ (T, J, 3, 3), y la elección (T, J, V) con −1 = descartada, j = la lectura de j.
    """
    V, T, J = conf.shape
    w_all = np.maximum(conf, wmin) ** 2
    P = np.zeros((T, J, 3))
    Lam = np.zeros((T, J, 3, 3))
    choice = np.zeros((T, J, V), int)
    stats = {"propia": 0, "gemela": 0, "descartada": 0}
    for j in range(J):
        opts = [[j] + twins.get(vistas[v], {}).get(j, []) + [-1] for v in range(V)]
        combos = [c for c in itertools.product(*opts) if any(x >= 0 for x in c)]
        K = len(combos)
        C = np.array(combos)  # (K, V)
        # lecturas y pesos de cada combinación: (T, K, V, 2) y (T, K, V)
        src = np.where(C >= 0, C, 0)
        uvk = np.stack([uv[v][:, src[:, v]] for v in range(V)], 2)  # (T, K, V, 2)
        wk = np.stack([w_all[v][:, src[:, v]] for v in range(V)], 2) * (C >= 0)[None]
        p, L, res, _ = _wls(uvk, wk, A, b)
        unary = np.minimum(res / sigma_r ** 2, cap_r)
        cfk = np.stack([conf[v][:, j] for v in range(V)], 1)  # confianza de la lectura propia (T, V)
        for v in range(V):
            drop = C[:, v] < 0
            unary[:, drop] += pen_descarte * np.minimum(1.0, cfk[:, v] / conf_ref)[:, None]
            unary[:, (C[:, v] >= 0) & (C[:, v] != j)] += pen_gemelo
        if suelo is not None:
            known_z = L[..., 2, 2] > 1e-6
            unary += np.where(known_z, (np.maximum(0.0, suelo[j] - p[..., 2]) / sigma_suelo) ** 2, 0.0)
        # transición: salto en píxeles de la lectura elegida en cada vista. Si la vista se descarta en uno de los dos
        # cuadros, paga lo que salta su lectura propia con un tope `tr_descarte`: así un pico suelto (la lectura va y
        # vuelve: 2·cap_t) se descarta, pero un movimiento rápido y limpio no sale más barato descartando (en un
        # lanzamiento, la vista lateral entera se tiraba solo por moverse mucho).
        cost = unary[0].copy()
        back = np.zeros((T, K), int)
        for t in range(1, T):
            tr = np.zeros((K, K))
            for v in range(V):
                used = C[:, v] >= 0
                d2 = ((uvk[t, :, v][:, None] - uvk[t - 1, :, v][None]) ** 2).sum(-1) / sigma_t ** 2  # (K_t, K_{t-1})
                own = min(float(((uv[v][t, j] - uv[v][t - 1, j]) ** 2).sum()) / sigma_t ** 2, tr_descarte)
                tr += np.where(used[:, None] & used[None], np.minimum(d2, cap_t), own)
            tot = tr + cost[None]
            back[t] = tot.argmin(1)
            cost = tot.min(1) + unary[t]
        k = int(cost.argmin())
        for t in range(T - 1, -1, -1):
            P[t, j], Lam[t, j], choice[t, j] = p[t, k], L[t, k], C[k]
            for v in range(V):
                stats["descartada" if C[k, v] < 0 else "propia" if C[k, v] == j else "gemela"] += 1
            k = back[t, k]
    if log:
        n = sum(stats.values())
        log(f"triangulación: lecturas propias {stats['propia'] / n:.1%}, de una gemela {stats['gemela'] / n:.1%}, "
            f"descartadas {stats['descartada'] / n:.1%}")
    return P, Lam, choice
