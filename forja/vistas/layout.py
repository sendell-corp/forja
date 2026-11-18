"""El cuadro de N vistas ortográficas: proyectar, componer y partir.

Generaliza `mocap/layout.py` de anymove (Apache-2.0, dos vistas) a cualquier lista de vistas puestas una al lado de la
otra, todas a la misma escala (M_PER_PX) y con el mismo alto H:

- **front** (cámara en −Y mirando a +Y, la cara de la criatura): u crece con +X; v crece con −Z.
- **side** (cámara en −X mirando a +X, su flanco derecho): u crece con −Y (la cabeza a la derecha); v crece con −Z.
- **top** (cámara en +Z mirando hacia abajo, arriba de la imagen = +Y): u crece con +X; v crece con −Y, así que la
  cabeza (−Y) queda abajo, hacia quien mira, como en la vista de frente. Las columnas de front y top son la misma X.

Cada vista es una proyección afín uv = A·p + b (A de 2x3), lo que deja la triangulación lineal y lista para cámaras
calibradas (teléfonos) en F4: basta otra A y b por cuadro.
"""
import json
import pathlib

import numpy as np

H = 720
BG, FLOOR, DIVIDER = (128, 130, 134), (152, 154, 158), (34, 34, 38)

# eje de mundo y signo de u y de v en cada vista
_AXES = {"front": ((0, +1), (2, -1)), "side": ((1, -1), (2, -1)), "top": ((0, +1), (1, -1))}


class Vista:
    def __init__(self, name, w, h, m_per_px, centers):
        (au, su), (av, sv) = _AXES[name]
        self.name, self.w, self.h = name, w, h
        self.A = np.zeros((2, 3))
        self.A[0, au] = su / m_per_px
        self.A[1, av] = sv / m_per_px
        self.b = np.array([w / 2 - su * centers[au] / m_per_px, h / 2 - sv * centers[av] / m_per_px])

    def project(self, p):
        return np.asarray(p, np.float64) @ self.A.T + self.b


class Layout:
    """Las vistas de una criatura. `cfg` es su creature.json (la clave "layout" de anymove más "forja")."""

    def __init__(self, cfg=None, vistas=("front", "side", "top")):
        if isinstance(cfg, (str, pathlib.Path)):
            cfg = json.load(open(cfg, encoding="utf-8"))
        cfg = cfg or {}
        lay = cfg.get("layout", {})
        fj = cfg.get("forja", {})
        self.M = lay.get("M_PER_PX", 1.9 / H)
        self.GROUND_V = lay.get("GROUND_V", 630)
        self.Z_CENTER = (self.GROUND_V - H / 2) * self.M
        self.TOP_Y_CENTER = fj.get("TOP_Y_CENTER", 0.0)
        widths = {"front": lay.get("W_FRONT", 420), "side": lay.get("W_SIDE", 860), "top": fj.get("W_TOP", 640)}
        self.names = list(vistas)
        self.views = [Vista(n, widths[n], H, self.M, (0.0, self.TOP_Y_CENTER if n == "top" else 0.0, self.Z_CENTER))
                      for n in self.names]
        self.x0 = np.cumsum([0] + [v.w for v in self.views])[:-1].tolist()
        self.W = int(sum(v.w for v in self.views))
        self.H = H

    def __getitem__(self, name):
        return self.views[self.names.index(name)]

    def project(self, p):
        """Puntos de mundo (…, 3) → (V, …, 2) en píxeles de cada vista."""
        return np.stack([v.project(p) for v in self.views])

    def A(self):
        return np.stack([v.A for v in self.views]), np.stack([v.b for v in self.views])

    def compose(self, renders, bg=BG, floor=FLOOR, divider=DIVIDER):
        """renders: {vista: RGBA (H, w, 4)} → el cuadro RGB (H, W, 3). Frente y lado con pared y suelo; arriba, todo suelo."""
        out = np.zeros((H, self.W, 3), np.float32)
        for v, x0 in zip(self.views, self.x0):
            pan = out[:, x0:x0 + v.w]
            pan[:] = floor if v.name == "top" else bg
            if v.name != "top":
                pan[self.GROUND_V:] = floor
            img = renders[v.name].astype(np.float32)
            a = img[..., 3:4] / 255.0
            h, w = img.shape[:2]
            pan[:h, :w] = img[..., :3] * a + pan[:h, :w] * (1 - a)
        for x0 in self.x0[1:]:
            out[:, x0 - 2:x0 + 2] = divider
        return out.clip(0, 255).astype(np.uint8)

    def split(self, frame):
        return {v.name: frame[:, x0:x0 + v.w] for v, x0 in zip(self.views, self.x0)}

    def sub(self, vistas):
        """El mismo layout con solo algunas vistas (p. ej. frente y lado para el lector de dos vistas)."""
        out = Layout.__new__(Layout)
        out.__dict__.update(self.__dict__)
        out.names = list(vistas)
        out.views = [self[n] for n in vistas]
        out.x0 = np.cumsum([0] + [v.w for v in out.views])[:-1].tolist()
        out.W = int(sum(v.w for v in out.views))
        return out

    def blender_cams(self):
        """Lo que necesita Blender para poner cada cámara (ver vistas/blender/con_vistas.py)."""
        return {"M_PER_PX": self.M, "H": H, "Z_CENTER": self.Z_CENTER, "TOP_Y_CENTER": self.TOP_Y_CENTER,
                "views": {v.name: v.w for v in self.views}}
