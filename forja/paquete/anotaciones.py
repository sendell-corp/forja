"""Anotaciones de cada clip para sendell-motion, en el espacio del juego (glTF: Y arriba, la criatura mira a +Z).

Formato (un JSON junto al GLB; tiempos en cuadros a `fps`, distancias en metros):
{
  "version": 1, "criatura": "spider", "modelo": "<glb>", "fps": 24,
  "clips": {
    "<animación del GLB>": {
      "cuadros": T, "duracion_s": T / fps, "bucle": bool,
      "root_motion": {"modo": "en_sitio", "x": [...], "z": [...], "yaw": [...],   // acumulado desde el cuadro 0
                      "avance_m": …, "giro_deg": …, "velocidad_m_s": …},
      "contactos": {"leg0_3": [[a, b], …], …},          // tramos de apoyo [a, b) de cada punta
      "impactos": [{"cuadro": f, "hueso": "fangL_end", "punto": [x, y, z], "velocidad_m_s": v}],   // punto: clip en el sitio
      "alcance": {"hueso": …, "m": …, "cuadro": f},     // lo más lejos que llega el efector, hacia delante, desde la
                                                        // posición inicial del cuerpo (con root motion), en la ventana activa
      "ventanas": {"anticipacion": [a, b], "activa": [a, b], "recuperacion": [a, b], "cancelable": [a, b]},
      "recibido": f                                     // golpe recibido (reacciones)
    }
  }
}
Las ventanas siguen el PLAN de sendell-motion: el warping de raíz y orientación actúa en la anticipación, la IK al blanco en
el cuadro de impacto, y la entrada a otro movimiento solo dentro de «cancelable».
"""
import numpy as np

from forja.paquete.glb import convertir, fk_juego


def _f(t, fps, T):
    return int(np.clip(round(t * fps), 0, T - 1))


def tramos(mask):
    out, t = [], 0
    while t < len(mask):
        if mask[t]:
            e = t
            while e < len(mask) and mask[e]:
                e += 1
            out.append([t, e])
            t = e
        else:
            t += 1
    return out


def anotar(rig, motion, meta, mapa, js, fps=24):
    T = len(motion["trans"])
    ev = meta.get("eventos", {})
    pistas, rm = convertir(rig, motion, mapa, js, en_sitio=True)
    pistas_mundo, _ = convertir(rig, motion, mapa, js, en_sitio=False)
    out = {"cuadros": T, "duracion_s": round(T / fps, 4), "bucle": bool(meta.get("bucle", False))}
    av = float(np.hypot(rm["x"][-1], rm["z"][-1]))
    out["root_motion"] = {"modo": "en_sitio", "x": np.round(rm["x"], 5).tolist(), "z": np.round(rm["z"], 5).tolist(),
                          "yaw": np.round(rm["yaw"], 5).tolist(), "avance_m": round(av, 4),
                          "giro_deg": round(float(np.degrees(rm["yaw"][-1])), 2),
                          "velocidad_m_s": round(av / max((T - 1) / fps, 1e-9), 4)}
    st = np.array(meta["stance"], bool)
    out["contactos"] = {n: tramos(st[:, k]) for k, n in enumerate(meta["tips"])}
    for w in ("anticipacion", "activa", "recuperacion", "cancelable"):
        if w in ev:
            out.setdefault("ventanas", {})[w] = [_f(ev[w][0], fps, T), _f(ev[w][1], fps, T)]
    if out["bucle"]:
        out["ventanas"] = {"cancelable": [0, T - 1]}
    if "recibido" in ev:
        out["recibido"] = _f(ev["recibido"], fps, T)
    if "impacto" in ev:
        out["impactos"] = []
        for hueso, t in ev["impacto"]:
            f = _f(t, fps, T)
            p = fk_juego(js, pistas, f)[hueso]
            p0 = fk_juego(js, pistas, max(f - 1, 0))[hueso]
            out["impactos"].append({"cuadro": f, "hueso": hueso, "punto": np.round(p, 4).tolist(),
                                    "velocidad_m_s": round(float(np.linalg.norm(p - p0) * fps), 3)})
        if "activa" in ev:
            a, b = _f(ev["activa"][0], fps, T), _f(ev["activa"][1], fps, T)
            body0 = fk_juego(js, pistas_mundo, 0)["body"]
            mejor = None
            for hueso in {h for h, _ in ev["impacto"]}:
                for f in range(a, b + 1):
                    d = float(fk_juego(js, pistas_mundo, f)[hueso][2] - body0[2])  # hacia delante = +Z
                    if mejor is None or d > mejor["m"]:
                        mejor = {"hueso": hueso, "m": round(d, 4), "cuadro": f}
            out["alcance"] = mejor
    return out, pistas
