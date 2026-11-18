"""Suavizado temporal que no aplana los golpes.

Para cada señal y_t con pesos w_t se busca x que minimiza Σ_t w_t (x_t − y_t)² + λ Σ_t ρ_δ(x_{t+1} − 2x_t + x_{t−1}),
con ρ_δ la pérdida de Huber sobre la aceleración: cuadrática para el temblor pequeño (lo quita) y lineal para las
aceleraciones grandes (un golpe, un aterrizaje, el cierre de un mordisco), que así se pagan poco y no se redondean.
Se resuelve por mínimos cuadrados reponderados (IRLS): cada pasada es un sistema lineal banda de T×T.
Los cuadros con peso 0 (sin lectura) se rellenan con la curva más suave que pasa por sus vecinos.
"""
import numpy as np


def _d2(T):
    D = np.zeros((max(T - 2, 0), T))
    for t in range(T - 2):
        D[t, t:t + 3] = (1.0, -2.0, 1.0)
    return D


def suavizar(y, w=None, lam=4.0, delta=None, iters=6, perdida="huber"):
    """y (T, …) señales en columnas; w (T, …) pesos ≥ 0 (o None = 1). delta: escala de la pérdida robusta en unidades
    de y por cuadro² (None = cuadrática pura). perdida: "huber" (las aceleraciones grandes pagan lineal) o "cauchy"
    (redescendente: un golpe casi no paga y no se recorta). Devuelve x con la forma de y."""
    y = np.asarray(y, np.float64)
    T = y.shape[0]
    if T < 3:
        return y.copy()
    shp = y.shape
    Y = y.reshape(T, -1)
    Wt = np.ones_like(Y) if w is None else np.broadcast_to(np.asarray(w, np.float64).reshape(T, -1) if np.ndim(w) > 1 else
                                                         np.asarray(w, np.float64)[:, None], Y.shape).copy()
    Yz = np.where(Wt > 0, Y, 0.0)
    D = _d2(T)
    X = np.empty_like(Y)
    for c in range(Y.shape[1]):
        wd = np.ones(T - 2)
        x = Yz[:, c]
        for _ in range(iters if delta is not None else 1):
            M = np.diag(Wt[:, c]) + lam * D.T @ (wd[:, None] * D) + 1e-9 * np.eye(T)
            x = np.linalg.solve(M, Wt[:, c] * Yz[:, c])
            if delta is None:
                break
            a = np.abs(D @ x)
            if perdida == "cauchy":
                wd = 1.0 / (1.0 + (a / delta) ** 2)
            else:
                wd = np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-12))
        X[:, c] = x
    return X.reshape(shp)


def rellenar(y, ok):
    """Interpola en el tiempo los cuadros sin dato (ok False), columna a columna."""
    y = np.array(y, np.float64)
    T = y.shape[0]
    Y = y.reshape(T, -1)
    O = np.broadcast_to(ok.reshape(T, -1) if np.ndim(ok) > 1 else ok[:, None], Y.shape)
    t = np.arange(T)
    for c in range(Y.shape[1]):
        if O[:, c].any() and not O[:, c].all():
            Y[~O[:, c], c] = np.interp(t[~O[:, c]], t[O[:, c]], Y[O[:, c], c])
    return Y.reshape(y.shape)
