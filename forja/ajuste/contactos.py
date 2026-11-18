"""Qué patas están apoyadas, con coherencia temporal, y dónde quedan clavadas.

Por pata, un modelo oculto de dos estados (apoyo / vuelo) resuelto con Viterbi:
- coste de «apoyo» en cada cuadro: ((z − h0)₊/σz)² + (v/σv)², con z la altura de la punta, h0 su altura de reposo
  (la punta toca el suelo ahí: la garra de la malla baja por debajo) y v su velocidad **3D** de un lado, la menor
  entre la de llegada y la de salida: apoyo es «quieta y baja». Al aterrizar, la de salida es 0; al despegar, la punta
  ya sube 1–3 cm en el primer cuadro (con solo la velocidad horizontal, el primer y el último cuadro de cada vuelo se
  tomaban por apoyo; con la central, el apoyo perdía un cuadro por cada lado);
- coste de «vuelo»: una constante κ (umbral);
- cada cambio de estado paga γ, y luego se quitan apoyos y vuelos más cortos que `min_run` cuadros.
Una pata no se apoya y se levanta de un cuadro al siguiente, aunque una lectura suelta diga otra cosa.

`plantar` da, por cada apoyo, el punto donde la punta queda clavada: la mediana de sus posiciones en el apoyo, en
el suelo (z = h0). La IK final la sujeta ahí: deslizamiento 0 por construcción mientras la pata alcance.
"""
import numpy as np


def _viterbi2(cost_st, cost_sw, switch):
    T = len(cost_st)
    c = np.array([cost_sw[0], cost_st[0]])
    back = np.zeros((T, 2), int)
    for t in range(1, T):
        stay, move = c, c[::-1] + switch
        back[t] = np.where(stay <= move, [0, 1], [1, 0])
        c = np.minimum(stay, move) + np.array([cost_sw[t], cost_st[t]])
    s = np.zeros(T, int)
    s[-1] = int(c.argmin())
    for t in range(T - 1, 0, -1):
        s[t - 1] = back[t, s[t]]
    return s.astype(bool)


def runs(mask):
    out, t, T = [], 0, len(mask)
    while t < T:
        e = t
        while e < T and mask[e] == mask[t]:
            e += 1
        out.append((bool(mask[t]), t, e))
        t = e
    return out


def limpiar(mask, min_run=3):
    m = mask.copy()
    for _ in range(3):
        for val, a, b in runs(m):
            if b - a < min_run and 0 < a and b < len(m):  # un tramo corto en medio: se une a sus vecinos
                m[a:b] = not val
    return m


def apoyos(tips, h0, *, sz=0.015, sv=0.006, kappa=4.0, switch=6.0, min_run=3):
    """tips (T, K, 3) posiciones de las puntas; h0 (K,) → apoyo (T, K) bool."""
    T, K = tips.shape[:2]
    d = np.linalg.norm(np.diff(tips, axis=0), axis=-1)  # (T − 1, K)
    v = np.minimum(np.vstack([d[:1], d]), np.vstack([d, d[-1:]]))
    out = np.zeros((T, K), bool)
    for k in range(K):
        cst = (np.maximum(0.0, tips[:, k, 2] - h0[k]) / sz) ** 2 + (v[:, k] / sv) ** 2
        out[:, k] = limpiar(_viterbi2(cst, np.full(T, kappa), switch), min_run)
    return out


def plantar(tips, stance, h0):
    """Objetivo de cada punta en sus apoyos (T, K, 3) (NaN fuera de apoyo): la mediana de su xy en el apoyo, en z = h0."""
    T, K = stance.shape
    tgt = np.full((T, K, 3), np.nan)
    for k in range(K):
        for val, a, b in runs(stance[:, k]):
            if val:
                tgt[a:b, k, :2] = np.median(tips[a:b, k, :2], axis=0)
                tgt[a:b, k, 2] = h0[k]
    return tgt
