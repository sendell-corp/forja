"""Un humanoide del juego leído directamente de su GLB: el rig de Forja (forja/rig.py), su piel y sus partes.

No hace falta Blender: el esqueleto del GLB (exportado por Blender, con el eje Y local de cada hueso hacia su hijo) se
pasa al formato de `Rig` (matrix_local de reposo en un mundo Z arriba, padre, longitud, límites y articulaciones):
- mundo de Forja = Cᵀ·mundo glTF, con C: (x, y, z) → (x, z, −y), el mismo conversor que paquete/glb.py;
- la raíz libre del rig es `pelvis`; el nodo `root` del juego (y lo que cuelga por encima) queda quieto;
- como B_hijo = B_padre · local_reposo, la pose de un hueso es padre · local_reposo · R(θ): θ = 0 es la pose del GLB y
  la rotación local que se exporta es local_reposo · R(θ), sin conversor de reposo.

La piel (lineal, 4 influencias) sirve para medir la malla deformada: suelo, manos contra la caja y piezas para Jolt.
"""
import numpy as np

from forja.paquete.glb import leer
from forja.rig import Rig, quat_to_matrix

C = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])  # Forja (Z arriba) → glTF (Y arriba)
C4 = np.eye(4)
C4[:3, :3] = C

# Límites (grados, Euler XYZ de Blender sobre el reposo del GLB). Se ponen por papel; el signo de las bisagras
# (rodilla, codo) se deduce de la geometría en `orientar_bisagras` y aquí se escriben para el lado que dobla bien.
LIMITES = {
    "pelvis": [[-180, 180]] * 3,
    "spine": [[-15, 50], [-25, 25], [-20, 20]],
    "chest": [[-15, 40], [-25, 25], [-20, 20]],
    "neck": [[-30, 45], [-40, 40], [-20, 20]],
    "head": [[-35, 35], [-45, 45], [-20, 20]],
    "thigh": [[-30, 130], [-40, 40], [-45, 45]],
    "shin": None,       # bisagra: 0..150 en el sentido que dobla hacia atrás
    "foot": [[-45, 45], [-15, 15], [-20, 20]],
    "upper_arm": [[-150, 150], [-90, 90], [-150, 150]],
    "forearm": None,    # bisagra: 0..150
    "hand": [[-70, 70], [-90, 90], [-40, 40]],
}


def _trs(n):
    M = np.eye(4)
    if "matrix" in n:
        return np.array(n["matrix"], np.float64).reshape(4, 4).T
    q = n.get("rotation", [0, 0, 0, 1])
    M[:3, :3] = quat_to_matrix(np.array([q[3], q[0], q[1], q[2]]))
    M[:3, 3] = n.get("translation", [0, 0, 0])
    return M  # la escala del juego es 1 (±1e-6): se ignora


class Humanoide:
    def __init__(self, glb_path):
        self.js, self.bin = leer(glb_path)
        js = self.js
        nodes = js["nodes"]
        self.node_idx = {n.get("name"): i for i, n in enumerate(nodes)}
        self.par = {}
        for i, n in enumerate(nodes):
            for c in n.get("children", []):
                self.par[c] = i
        self.local = [_trs(n) for n in nodes]
        self.W = [None] * len(nodes)
        for i in range(len(nodes)):
            self._mundo(i)
        skin = js["skins"][0]
        self.skin_joints = list(skin["joints"])
        jn = [nodes[j]["name"] for j in self.skin_joints]
        if "pelvis" not in jn:
            raise SystemExit(f"el esqueleto no tiene «pelvis»: {jn}")
        # huesos del rig: pelvis y sus descendientes en la piel, padres antes que hijos
        sub = []
        pila = [self.node_idx["pelvis"]]
        while pila:
            i = pila.pop(0)
            if i in self.skin_joints:
                sub.append(i)
            pila.extend(c for c in nodes[i].get("children", []) if c in self.skin_joints)
        self.rig_nodes = sub
        self.names = [nodes[i]["name"] for i in sub]
        self._piel()
        self.rig = self._rig()

    def _mundo(self, i):
        if self.W[i] is None:
            p = self.par.get(i)
            self.W[i] = self.local[i] if p is None else self._mundo(p) @ self.local[i]
        return self.W[i]

    # ---------------------------------------------------------------- piel
    def _acc(self, k):
        a = self.js["accessors"][k]
        bv = self.js["bufferViews"][a["bufferView"]]
        dt = {5126: np.float32, 5123: np.uint16, 5121: np.uint8, 5125: np.uint32}[a["componentType"]]
        nc = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}[a["type"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride", 0)
        isz = np.dtype(dt).itemsize * nc
        if stride and stride != isz:
            raw = np.frombuffer(bytes(self.bin), np.uint8, a["count"] * stride, off).reshape(-1, stride)[:, :isz]
            arr = raw.copy().view(dt).reshape(a["count"], nc)
        else:
            arr = np.frombuffer(bytes(self.bin), dt, a["count"] * nc, off).reshape(a["count"], nc)
        if a.get("normalized"):
            arr = arr.astype(np.float64) / np.iinfo(dt).max
        return arr.astype(np.float64)

    def _piel(self):
        js = self.js
        mesh_node = next(i for i, n in enumerate(js["nodes"]) if "skin" in n and "mesh" in n)
        self.mesh_node = mesh_node
        V, Jt, Wt, F = [], [], [], []
        base = 0
        for p in js["meshes"][js["nodes"][mesh_node]["mesh"]]["primitives"]:
            v = self._acc(p["attributes"]["POSITION"])
            V.append(v)
            Jt.append(self._acc(p["attributes"]["JOINTS_0"]).astype(int))
            Wt.append(self._acc(p["attributes"]["WEIGHTS_0"]))
            if "indices" in p:
                F.append(self._acc(p["indices"]).reshape(-1, 3).astype(int) + base)
            base += len(v)
        self.V = np.concatenate(V)
        self.VJ = np.concatenate(Jt)
        self.VW = np.concatenate(Wt)
        self.VW /= np.maximum(self.VW.sum(1, keepdims=True), 1e-9)
        self.F = np.concatenate(F) if F else None
        ibm = self._acc(self.js["skins"][0]["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
        self.IBM = ibm
        # comprobación: en reposo, W·IBM = I
        err = max(np.abs(self.W[j] @ ibm[k] - np.eye(4)).max() for k, j in enumerate(self.skin_joints))
        self.err_bind = float(err)

    def piel(self, Wg):
        """Wg: {nodo: matriz de mundo glTF (T, 4, 4)} de los huesos de la piel → vértices deformados (T, V, 3) en glTF."""
        T = len(next(iter(Wg.values())))
        Mk = np.stack([Wg[j] @ self.IBM[k] for k, j in enumerate(self.skin_joints)], 1)  # (T, K, 4, 4)
        vh = np.concatenate([self.V, np.ones((len(self.V), 1))], 1)
        out = np.zeros((T, len(self.V), 3))
        for k in range(Mk.shape[1]):  # por hueso, solo sus vértices (sin tensores T×V×4×4)
            wk = (self.VW * (self.VJ == k)).sum(1)
            sel = np.nonzero(wk > 0)[0]
            if len(sel):
                out[:, sel] += wk[sel, None] * np.einsum("tij,vj->tvi", Mk[:, k, :3], vh[sel])
        return out

    def mundos_gltf(self, P):
        """Matrices del rig (T, nb, 4, 4) en Forja → {nodo: mundo glTF (T, 4, 4)} para todos los huesos de la piel."""
        T = len(P)
        out = {}
        for k, j in enumerate(self.skin_joints):
            if j in self.rig_nodes:
                out[j] = C4[None] @ P[:, self.rig_nodes.index(j)]
            else:
                out[j] = np.broadcast_to(self.W[j], (T, 4, 4))
        return out

    # ---------------------------------------------------------------- rig
    def _rig(self):
        nodes = self.js["nodes"]
        B = [C4.T @ self.W[i] for i in self.rig_nodes]
        dom = self.VJ[np.arange(len(self.V)), self.VW.argmax(1)]  # hueso dominante de cada vértice (índice de piel)
        vw = (C4.T @ np.concatenate([self.V, np.ones((len(self.V), 1))], 1).T).T[:, :3]  # vértices de reposo en Forja
        self.V_forja = vw
        bones = []
        for k, i in enumerate(self.rig_nodes):
            nm = nodes[i]["name"]
            hijos = [c for c in nodes[i].get("children", []) if c in self.rig_nodes]
            if hijos:
                L = float(np.linalg.norm(self.local[hijos[0]][:3, 3]))
            else:
                # hoja (mano, pie, cabeza): hasta donde llega su parte de la malla a lo largo de su eje Y
                sk = self.skin_joints.index(i)
                pts = vw[dom == sk]
                y = (pts - B[k][:3, 3]) @ B[k][:3, 1]
                L = float(np.percentile(y, 98)) if len(y) else 0.1
            papel = nm.rsplit("_", 1)[0] if nm.endswith(("_L", "_R")) else nm
            lim = LIMITES.get(papel) or [[-180, 180]] * 3
            p = self.par.get(i)
            bones.append({"name": nm, "parent": self.rig_nodes.index(p) if p in self.rig_nodes else -1,
                          "matrix_local": B[k].tolist(), "length": L, "limits_deg": lim,
                          "head_joint": nm, "tail_joint": nm + "_tail" if not hijos else nodes[hijos[0]]["name"]})
        kps = [{"name": b["name"], "bone": k, "end": "head"} for k, b in enumerate(bones)]
        kps += [{"name": b["tail_joint"], "bone": k, "end": "tail"} for k, b in enumerate(bones) if b["tail_joint"].endswith("_tail")]
        info = {"bones": bones, "keypoints": kps}
        rig = Rig(_con_reposo(info))
        self.info = info
        self._orientar_bisagras(rig)
        rig = Rig(_con_reposo(info))
        return rig

    def _orientar_bisagras(self, rig):
        """Rodilla y codo: el sentido de X en que doblan hacia atrás (rodilla) o hacia delante (codo)."""
        fwd = self.adelante()
        for b in self.info["bones"]:
            papel = b["name"].rsplit("_", 1)[0]
            if papel not in ("shin", "forearm"):
                continue
            k = rig.names.index(b["name"])
            eul = np.zeros((1, len(rig.names), 3))
            J0, _ = rig.forward(np.zeros((1, 3)), np.eye(3)[None], eul)
            eul[0, k, 0] = np.radians(30)
            J1, _ = rig.forward(np.zeros((1, 3)), np.eye(3)[None], eul)
            t = rig.kp_names.index(rig.info["bones"][k]["tail_joint"])
            d = float((J1[0, t] - J0[0, t]) @ fwd)
            quiere = -1 if papel == "shin" else 1  # el tobillo va hacia atrás; la mano, hacia delante
            sgn = 1 if d * quiere > 0 else -1
            lo, hi = (0, 150) if sgn > 0 else (-150, 0)
            b["limits_deg"] = [[lo, hi], [-5, 5], [-5, 5]] if papel == "shin" else [[lo, hi], [-60, 60], [-10, 10]]
            b["bisagra_signo"] = sgn

    def adelante(self):
        """Hacia dónde mira el personaje en Forja (unitario, horizontal): de la nuca a la boca o del talón a la punta."""
        V = self.V_forja
        y = V[:, 2]
        cab = V[y > y.max() - 0.25]
        c = cab.mean(0)
        # la cara sobresale hacia delante: el lado con más masa lejos del eje de la cabeza
        dx = cab[:, :2] - c[:2]
        sx = np.sign(np.mean(dx[:, 0] ** 3)) if abs(np.mean(dx[:, 0] ** 3)) > abs(np.mean(dx[:, 1] ** 3)) else 0
        sy = np.sign(np.mean(dx[:, 1] ** 3)) if sx == 0 else 0
        # mejor señal: los pies apuntan hacia delante
        fl = self.info["bones"] if hasattr(self, "info") else None
        if fl:
            d = sum(np.array(b["matrix_local"])[:3, 1] for b in fl if b["name"].startswith("foot"))
            if not np.isscalar(d):
                d[2] = 0
                return d / np.linalg.norm(d)
        v = np.array([sx, sy, 0.0])
        return v / max(np.linalg.norm(v), 1e-9)


def _con_reposo(info):
    """Rig necesita la posición de reposo de cada articulación (`rest`): se calcula con su propia FK en θ = 0."""
    import copy
    info = copy.deepcopy(info)
    for k in info["keypoints"]:
        k["rest"] = [0, 0, 0]
    r = Rig(info)
    nb = len(r.names)
    J, _ = r.forward(np.zeros((1, 3)), np.eye(3)[None], np.zeros((1, nb, 3)))
    for k, p in zip(info["keypoints"], J[0]):
        k["rest"] = p.tolist()
    return info
