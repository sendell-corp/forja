"""El rig de una criatura en numpy: cinemática directa idéntica a la de Blender y la topología de sus extremidades.

Lee el `rig_info.json` que escribe `blender/build_rig.py` de anymove (Apache-2.0): por hueso, su matriz de reposo
`matrix_local` (4x4, espacio de la armadura), su padre, su longitud y sus límites en grados (Euler XYZ); por
articulación, el hueso y el extremo (head/tail) donde está.

Convenciones (las de Blender y anymove):
- pose de un hueso = pose del padre · desplazamiento de reposo · base(R), con R = Rz·Ry·Rx (Euler «XYZ» de Blender);
- la raíz (hueso 0) es libre: su rotación `root_R` (3x3, en el espacio del propio hueso) y una traslación de mundo
  `trans`, como `apply_motion.py` (pose.location = B^T·trans);
- el eje Y local de cada hueso va de la cabeza a la cola.
"""
import json
import pathlib

import numpy as np


def euler_xyz(a):
    """Euler XYZ de Blender (…, 3) a matrices (…, 3, 3): Rz @ Ry @ Rx."""
    a = np.asarray(a, np.float64)
    x, y, z = a[..., 0], a[..., 1], a[..., 2]
    cx, sx, cy, sy, cz, sz = np.cos(x), np.sin(x), np.cos(y), np.sin(y), np.cos(z), np.sin(z)
    R = np.empty(a.shape[:-1] + (3, 3))
    R[..., 0, 0] = cy * cz
    R[..., 0, 1] = sx * sy * cz - cx * sz
    R[..., 0, 2] = cx * sy * cz + sx * sz
    R[..., 1, 0] = cy * sz
    R[..., 1, 1] = sx * sy * sz + cx * cz
    R[..., 1, 2] = cx * sy * sz - sx * cz
    R[..., 2, 0] = -sy
    R[..., 2, 1] = sx * cy
    R[..., 2, 2] = cx * cy
    return R


def euler_from_matrix(R):
    """Inversa de euler_xyz (…, 3, 3) → (…, 3)."""
    R = np.asarray(R, np.float64)
    y = -np.arcsin(np.clip(R[..., 2, 0], -1, 1))
    x = np.arctan2(R[..., 2, 1], R[..., 2, 2])
    z = np.arctan2(R[..., 1, 0], R[..., 0, 0])
    return np.stack([x, y, z], -1)


def quat_to_matrix(q):
    """Cuaterniones (…, 4) w, x, y, z → (…, 3, 3)."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def matrix_to_quat(R):
    """(…, 3, 3) → cuaterniones (…, 4) w, x, y, z con w ≥ 0 (método de Shepperd, estable)."""
    R = np.asarray(R, np.float64)
    shp = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    q = np.empty((len(R), 4))
    for i, m in enumerate(R):
        t = np.trace(m)
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            q[i] = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
        if q[i, 0] < 0:
            q[i] = -q[i]
    return q.reshape(shp + (4,))


def axis_angle_matrix(v):
    """Vector eje·ángulo (…, 3) → (…, 3, 3) (Rodrigues)."""
    v = np.asarray(v, np.float64)
    th = np.linalg.norm(v, axis=-1)[..., None, None]
    k = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class Rig:
    """Cinemática directa del rig de anymove (la misma cuenta que mocap/solve.py Rig.forward y que Blender)."""

    def __init__(self, info):
        if isinstance(info, (str, pathlib.Path)):
            info = json.load(open(info, encoding="utf-8"))
        self.info = info
        bones = info["bones"]
        self.names = [b["name"] for b in bones]
        self.parent = [b["parent"] for b in bones]
        self.B = np.array([b["matrix_local"] for b in bones], np.float64)
        self.L = np.array([b["length"] for b in bones], np.float64)
        self.off = np.stack([self.B[i] if p < 0 else np.linalg.inv(self.B[p]) @ self.B[i] for i, p in enumerate(self.parent)])
        lim = np.radians(np.array([b["limits_deg"] for b in bones], np.float64))
        self.lo, self.hi = lim[..., 0], lim[..., 1]
        self.kp = [(k["bone"], k["end"]) for k in info["keypoints"]]
        self.kp_names = [k["name"] for k in info["keypoints"]]
        self.rest = np.array([k["rest"] for k in info["keypoints"]], np.float64)
        self.children = [[j for j, p in enumerate(self.parent) if p == i] for i in range(len(bones))]
        self.head_joint = [b["head_joint"] for b in bones]
        self.tail_joint = [b["tail_joint"] for b in bones]

    # ---------------------------------------------------------------- cinemática directa
    def root_matrix(self, trans, root_R):
        """Matriz de mundo (T, 4, 4) del hueso raíz."""
        T = len(trans)
        M = np.repeat(self.off[0][None], T, 0).copy()
        M[:, :3, :3] = self.off[0][:3, :3] @ root_R
        M[:, :3, 3] += trans
        return M

    def forward(self, trans, root_R, eul):
        """trans (T, 3), root_R (T, 3, 3), eul (T, nb, 3) → articulaciones (T, J, 3) y matrices de mundo (T, nb, 4, 4)."""
        trans, root_R, eul = (np.asarray(a, np.float64) for a in (trans, root_R, eul))
        T, nb = eul.shape[:2]
        R = euler_xyz(eul)
        P = np.empty((T, nb, 4, 4))
        P[:, 0] = self.root_matrix(trans, root_R)
        for i in range(1, nb):
            basis = np.zeros((T, 4, 4))
            basis[:, :3, :3] = R[:, i]
            basis[:, 3, 3] = 1
            P[:, i] = P[:, self.parent[i]] @ self.off[i] @ basis
        return self.joints_from(P), P

    def joints_from(self, P):
        out = []
        for b, end in self.kp:
            out.append(P[:, b, :3, 3] if end == "head" else P[:, b, :3, 1] * self.L[b] + P[:, b, :3, 3])
        return np.stack(out, 1)

    def chain_forward(self, Mparent, bones, eul):
        """FK de una cadena (huesos en orden padre→hijo, el primero colgado de Mparent (…, 4, 4)) con eul (…, n, 3).
        Devuelve las matrices de mundo (…, n, 4, 4) y las posiciones (…, n + 1, 3): cabezas y la cola del último."""
        R = euler_xyz(eul)
        Ms, pts = [], []
        M = Mparent
        for k, b in enumerate(bones):
            basis = np.zeros(R.shape[:-3] + (4, 4))
            basis[..., :3, :3] = R[..., k, :, :]
            basis[..., 3, 3] = 1
            M = M @ self.off[b] @ basis
            Ms.append(M)
            pts.append(M[..., :3, 3])
        pts.append(M[..., :3, 1] * self.L[bones[-1]] + M[..., :3, 3])
        return np.stack(Ms, -3), np.stack(pts, -2)

    # ---------------------------------------------------------------- topología
    def joint(self, name):
        return self.kp_names.index(name)

    def bone(self, name):
        return self.names.index(name)

    def rigid_on_root(self):
        """Articulaciones que viajan rígidas con la raíz: sus dos extremos y las cabezas de los huesos que cuelgan de ella."""
        ride = [self.head_joint[0], self.tail_joint[0]] + [self.head_joint[c] for c in self.children[0]]
        return [self.kp_names.index(n) for n in dict.fromkeys(ride) if n in self.kp_names]

    def chains(self):
        """Cadenas que cuelgan de la raíz, siguiendo hijos únicos hasta el final: [(huesos, articulaciones)], donde las
        articulaciones son la cabeza del primero, las colas de cada hueso (en orden). Patas, colmillos, abdomen…"""
        out = []
        for c in self.children[0]:
            bones = [c]
            while len(self.children[bones[-1]]) == 1:
                bones.append(self.children[bones[-1]][0])
            joints = [self.kp_names.index(self.head_joint[bones[0]])] + [self.kp_names.index(self.tail_joint[b]) for b in bones]
            out.append((bones, joints))
        return out


def cargar_garras(cfg_path, rig):
    """garras.json junto a creature.json (forja/modelo/garras.py) → {articulación de contacto: (vértices (V, 3), pesos
    (V, n))} con los pesos en el orden de los huesos de su cadena en el rig. {} si no existe."""
    f = pathlib.Path(cfg_path).parent / "garras.json"
    if not f.exists():
        return {}
    g = json.load(open(f, encoding="utf-8"))
    out = {}
    chains = {rig.kp_names[js[-1]]: [rig.names[b] for b in bones] for bones, js in rig.chains()}
    for c, d in g.items():
        if chains.get(c) == d["huesos"]:
            out[c] = (np.array(d["vertices"], np.float64), np.array(d["pesos"], np.float64))
    return out


def load_motion(path):
    m = json.load(open(path, encoding="utf-8"))
    eul = np.array(m["euler"], np.float64)
    return {"fps": m.get("fps", 24), "bones": m["bones"], "trans": np.array(m["trans"], np.float64),
            "root_R": quat_to_matrix(np.array(m["root_quat"], np.float64)), "euler": eul}


def save_motion(path, bones, trans, root_R, eul, fps=24, extra=None):
    """motion.json en el formato de anymove (apply_motion.py lo pone en el rig de Blender)."""
    out = {"fps": fps, "bones": list(bones), "trans": np.round(trans, 6).tolist(),
           "root_quat": np.round(matrix_to_quat(root_R), 7).tolist(), "euler": np.round(eul, 6).tolist()}
    if extra:
        out.update(extra)
    json.dump(out, open(path, "w", encoding="utf-8"))
