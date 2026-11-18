"""La fuente procedural de punta a punta: genera los clips del juego, los limpia con Jolt, los mide, y saca el GLB con
los nombres de hueso del juego y las anotaciones para sendell-motion.

python -m forja.fuentes.fabricar rig_info.json creature.json MODELO.glb SALIDA [--clips caminar,mordisco,…]

SALIDA/
  forja_<criatura>.glb                 el modelo del juego con sus animaciones de siempre más «forja_<clip>» (en el sitio)
  forja_<criatura>.anotaciones.json    contactos, impactos, alcance, root motion y ventanas (paquete/anotaciones.py)
  informe.json                         medidas por clip, antes y después de la limpieza
  clips/<clip>/{bruto.json, limpio.json, meta.json}   motion.json del rig de Forja (para Blender y el examen)
"""
import argparse
import json
import pathlib

import numpy as np

from forja.fisica.limpieza import cargar_cascos, limpiar, pares_limpios
from forja.fuentes.procedural import FPS, JUEGO, Arana, generar
from forja.paquete import glb as G
from forja.paquete.anotaciones import anotar
from forja.rig import Rig, cargar_garras, load_motion, save_motion


def medir_clip(rig, m, meta):
    """Deslizamiento de las puntas en sus apoyos, puntas bajo su altura de apoyo y costura del bucle."""
    names = rig.kp_names
    tips = [names.index(n) for n in meta["tips"]]
    st = np.array(meta["stance"], bool)
    h0 = np.array(meta["h0"])
    J, P = rig.forward(m["trans"], m["root_R"], m["euler"])
    pf, ps = [], []
    for k, j in enumerate(tips):
        t = 0
        while t < len(J):
            if st[t, k]:
                e = t
                while e < len(J) and st[e, k]:
                    e += 1
                if e - t >= 2:
                    pf.append(np.linalg.norm(np.diff(J[t:e, j, :2], axis=0), axis=-1))
                    ps.append(float(np.linalg.norm(J[e - 1, j, :2] - J[t, j, :2])))
                t = e
            else:
                t += 1
    pf = np.concatenate(pf) * 100 if pf else np.zeros(1)
    out = {"deslizamiento_cm_cuadro_media": float(pf.mean()), "deslizamiento_cm_cuadro_max": float(pf.max()),
           "deslizamiento_cm_apoyo_media": float(np.mean(ps) * 100) if ps else 0.0,
           "puntas_bajo_apoyo_max_cm": float(max(0.0, (h0 - J[:, tips, 2]).max()) * 100)}
    if meta.get("bucle"):
        loc = lambda f: (J[f] - P[f, 0, :3, 3]) @ P[f, 0, :3, :3]
        pasos = np.array([np.abs(loc(f + 1) - loc(f)).max() for f in range(len(J) - 1)])
        salto = np.abs(loc(0) - loc(len(J) - 1)).max()
        out["bucle_salto_ultimo_a_primero_cm"] = float(salto * 100)  # la costura: del último cuadro al primero
        out["bucle_paso_p95_cm"] = float(np.percentile(pasos, 95) * 100)  # un paso normal entre cuadros, p95
        out["bucle_paso_max_cm"] = float(pasos.max() * 100)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("info"), ap.add_argument("cfg"), ap.add_argument("modelo"), ap.add_argument("out")
    ap.add_argument("--clips", default=",".join(JUEGO))
    ap.add_argument("--sin-limpieza", action="store_true")
    a = ap.parse_args()
    rig = Rig(a.info)
    cfg = json.load(open(a.cfg, encoding="utf-8"))
    mapa = json.load(open(pathlib.Path(a.cfg).parent / "forja_map.json", encoding="utf-8"))
    ar = Arana(rig, cfg, cfg.get("layout", {}).get("M_PER_PX", 0.0035), cargar_garras(a.cfg, rig))
    cascos = cargar_cascos(a.cfg)
    pares, _ = pares_limpios(rig, cascos)
    js, bin_ = G.leer(a.modelo)
    nombre = pathlib.Path(a.modelo).stem
    out = pathlib.Path(a.out)
    anot = {"version": 1, "criatura": cfg.get("name", "criatura"), "modelo": f"forja_{nombre}.glb", "fps": FPS, "clips": {}}
    informe = {}
    for k, c in enumerate(a.clips.split(",")):
        d = out / "clips" / c
        d.mkdir(parents=True, exist_ok=True)
        g = generar(ar, c, seed=200 + k, juego=True)
        tips = [rig.kp_names[j[-1]] for _, j in ar.legs]
        meta = {"tips": tips, "legs": [ar.nombre_pata(i) for i in range(len(ar.legs))], "stance": g["stance"].astype(int).tolist(),
                "h0": ar.h0.round(5).tolist(), "bucle": JUEGO[c]["bucle"], "eventos": JUEGO[c]["eventos"]}
        json.dump(meta, open(d / "meta.json", "w", encoding="utf-8"), ensure_ascii=False)
        save_motion(d / "bruto.json", rig.names, g["trans"], g["root_R"], g["euler"], FPS)
        m = load_motion(d / "bruto.json")
        print(f"== {c}: {len(m['trans'])} cuadros", flush=True)
        if a.sin_limpieza:
            m2, inf = m, {}
        else:
            m2, inf = limpiar(rig, ar, m, meta, cascos, pares, log=lambda s: print(s, flush=True))
        save_motion(d / "limpio.json", rig.names, m2["trans"], m2["root_R"], m2["euler"], FPS)
        inf["medidas_bruto"] = medir_clip(rig, m, meta)
        inf["medidas_limpio"] = medir_clip(rig, m2, meta)
        nombre_anim = f"forja_{c}"
        an, pistas = anotar(rig, m2, meta, mapa, js, FPS)
        inf["conversor_error_mm"] = G.comprobar(rig, m2, mapa, js, G.convertir(rig, m2, mapa, js)[0])
        G.anadir_animacion(js, bin_, nombre_anim, FPS, pistas)
        anot["clips"][nombre_anim] = an
        informe[c] = inf
        print("   ", {k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in inf["medidas_limpio"].items()}, flush=True)
    G.escribir(js, bin_, out / f"forja_{nombre}.glb")
    json.dump(anot, open(out / f"forja_{nombre}.anotaciones.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(informe, open(out / "informe.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    print("glb", out / f"forja_{nombre}.glb", [x.get("name") for x in js["animations"]])


if __name__ == "__main__":
    main()
