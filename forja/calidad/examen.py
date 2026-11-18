"""El examen de F2: un ajuste contra el movimiento verdadero de un clip sintético.

python -m forja.calidad.examen VERDAD/truth_forja.json AJUSTE.json [más ajustes…] [--nombres a,b,c] [--json SALIDA.json]

AJUSTE.json puede ser un motion.json de anymove (mocap/solve.py) o de Forja: se recalculan sus articulaciones con la
cinemática directa del rig (forja/rig.py, idéntica a Blender) si no las trae.

Métricas (cm salvo indicación):
- **error 3D** por articulación: media y p95, en total, por pata (cadera, rodilla, tobillo y punta) y en las puntas;
- **arriba/abajo** (la marcha): una pata está «arriba» si su punta está a más de 1 cm sobre su altura de apoyo. La
  verdad es la del generador (apoyo/vuelo); se descartan los cuadros de vuelo en que la punta verdadera aún no ha
  subido 1 cm (despegue y aterrizaje: nadie puede distinguirlos del apoyo). Acierto = cuadros iguales / cuadros;
- **penetración:** puntas por debajo de su altura de apoyo y demás articulaciones por debajo de z = 0; profundidad
  máxima y % de cuadros con más de 0,5 cm;
- **deslizamiento en apoyo:** en los apoyos verdaderos, cuánto se mueve en horizontal la punta ajustada por cuadro
  (media y máximo) y en todo el apoyo (media por apoyo);
- **golpes:** pico de velocidad de colmillos y cuerpo en el ajuste / en la verdad; **temblor:** RMS de la aceleración
  del ajuste que no está en la verdad (cm/cuadro²);
- **malla bajo el suelo** (si existe suelo_<ajuste>/suelo.json de blender_suelo.py): la z mínima de la malla deformada
  por cuadro; se da la profundidad máxima bajo z = 0 y cuánto más baja que la verdad en el mismo cuadro.
"""
import pathlib
import argparse
import json

import numpy as np

from forja.rig import Rig, quat_to_matrix

UMBRAL_ARRIBA = 0.01
TOL_PEN = 0.005


def articulaciones(path, rig):
    m = json.load(open(path, encoding="utf-8"))
    if "euler" in m and "trans" in m:
        J, _ = rig.forward(np.array(m["trans"]), quat_to_matrix(np.array(m["root_quat"])), np.array(m["euler"]))
        return J, m
    return np.array(m["joints"]), m


def _runs(mask):
    out, t = [], 0
    while t < len(mask):
        if mask[t]:
            e = t
            while e < len(mask) and mask[e]:
                e += 1
            out.append((t, e))
            t = e
        else:
            t += 1
    return out


def medir(truth, J, names, legs_chains, leg_names):
    """truth: dict de truth_forja.json; J (T, J, 3) del ajuste; legs_chains: [[índices de las 4 articulaciones]]."""
    A = np.array(truth["joints"])
    T = min(len(A), len(J))
    A, J = A[:T], J[:T]
    st = np.array(truth["stance"], bool)[:T]
    h0 = np.array(truth["h0"])
    tips = [names.index(n) for n in truth["tips"]]
    e = np.linalg.norm(A - J, axis=-1) * 100
    out = {"error_cm": {"media": float(e.mean()), "p95": float(np.percentile(e, 95)), "mediana": float(np.median(e)),
                        "puntas_media": float(e[:, tips].mean()), "puntas_p95": float(np.percentile(e[:, tips], 95))},
           "por_pata": {}}
    for nm, ch in zip(leg_names, legs_chains):
        out["por_pata"][nm] = {"media": float(e[:, ch].mean()), "p95": float(np.percentile(e[:, ch], 95)),
                               "punta": float(e[:, ch[-1]].mean())}
    resto = [j for j in range(A.shape[1]) if not any(j in c for c in legs_chains)]
    out["error_cm"]["cuerpo_colmillos_abdomen"] = float(e[:, resto].mean())
    # arriba / abajo
    zt = A[:, tips, 2] - h0
    zf = J[:, tips, 2] - h0
    up_t = ~st
    clear = st | (zt > UMBRAL_ARRIBA)
    up_f = zf > UMBRAL_ARRIBA
    ok = (up_f == up_t) & clear
    out["arriba_abajo"] = {"acierto": float(ok.sum() / clear.sum()), "cuadros": int(clear.sum()),
                           "descartados_transicion": int((~clear).sum()),
                           "por_pata": {nm: float(ok[:, k].sum() / max(clear[:, k].sum(), 1)) for k, nm in enumerate(leg_names)},
                           "falsos_arriba": int((up_f & ~up_t & clear).sum()), "falsos_abajo": int((~up_f & up_t & clear).sum())}
    # penetración
    pen_tip = np.maximum(0.0, h0 - J[:, tips, 2]) * 100
    others = [j for j in range(J.shape[1]) if j not in tips]
    pen_oth = np.maximum(0.0, -J[:, others, 2]) * 100
    out["penetracion"] = {"puntas_max_cm": float(pen_tip.max()), "puntas_pct_cuadros": float((pen_tip > TOL_PEN * 100).mean() * 100),
                          "resto_max_cm": float(pen_oth.max()), "resto_pct": float((pen_oth > TOL_PEN * 100).mean() * 100)}
    # deslizamiento en los apoyos verdaderos
    per_frame, per_stance = [], []
    for k, j in enumerate(tips):
        for a, b in _runs(st[:, k]):
            if b - a < 2:
                continue
            d = np.linalg.norm(np.diff(J[a:b, j, :2], axis=0), axis=-1) * 100
            per_frame.append(d)
            per_stance.append(float(np.linalg.norm(J[b - 1, j, :2] - J[a, j, :2]) * 100))
    pf = np.concatenate(per_frame) if per_frame else np.zeros(1)
    out["deslizamiento"] = {"cm_por_cuadro_media": float(pf.mean()), "cm_por_cuadro_max": float(pf.max()),
                            "cm_por_apoyo_media": float(np.mean(per_stance)) if per_stance else 0.0, "apoyos": len(per_stance)}
    # golpes: pico de velocidad (cm/cuadro) de colmillos y cuerpo, ajuste frente a verdad (1 = conservado)
    gol = {}
    for g, js in (("colmillos", [j for j, n in enumerate(names) if n.startswith("fang") and n.endswith("_end")]),
                  ("cuerpo", [names.index("body")] if "body" in names else [])):
        if js:
            vt = np.linalg.norm(np.diff(A[:, js], axis=0), axis=-1).max() * 100
            vf = np.linalg.norm(np.diff(J[:, js], axis=0), axis=-1).max() * 100
            gol[g] = {"pico_verdad_cm_cuadro": float(vt), "pico_ajuste_cm_cuadro": float(vf), "proporcion": float(vf / max(vt, 1e-9))}
    out["golpes"] = gol
    # temblor: aceleración del ajuste que no está en la verdad (RMS de la diferencia de segundas diferencias, cm/cuadro²)
    acc = lambda X: X[2:] - 2 * X[1:-1] + X[:-2]
    out["temblor_cm_cuadro2"] = float(np.sqrt((np.linalg.norm(acc(J) - acc(A), axis=-1) ** 2).mean()) * 100)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("truth"), ap.add_argument("info"), ap.add_argument("fits", nargs="+")
    ap.add_argument("--nombres")
    ap.add_argument("--json")
    a = ap.parse_args()
    rig = Rig(a.info)
    truth = json.load(open(a.truth, encoding="utf-8"))
    names = rig.kp_names
    chains = [js for _, js in rig.chains() if names[js[-1]] in truth["tips"]]
    legs = truth["legs"]
    res = {}
    labels = a.nombres.split(",") if a.nombres else a.fits
    sv = pathlib.Path(a.truth).parent / "suelo_truth" / "suelo.json"
    zt = np.array(json.load(open(sv))["min_z"]) if sv.exists() else None
    for lab, f in zip(labels, a.fits):
        J, _ = articulaciones(f, rig)
        res[lab] = medir(truth, J, names, chains, legs)
        r = res[lab]
        sf = pathlib.Path(f).parent / f"suelo_{pathlib.Path(f).stem}" / "suelo.json"
        if zt is not None and sf.exists():
            zf = np.array(json.load(open(sf))["min_z"])
            n = min(len(zt), len(zf))
            r["malla"] = {"bajo_suelo_max_cm": float(max(0.0, -zf.min()) * 100), "verdad_bajo_suelo_max_cm": float(max(0.0, -zt.min()) * 100),
                          "mas_baja_que_verdad_max_cm": float(max(0.0, (zt[:n] - zf[:n]).max()) * 100),
                          "cuadros_bajo_suelo_pct": float((zf < -0.005).mean() * 100)}
            print(f"   malla: {r['malla']['bajo_suelo_max_cm']:.2f} cm bajo el suelo como mucho (la verdad {r['malla']['verdad_bajo_suelo_max_cm']:.2f}); "
                  f"hasta {r['malla']['mas_baja_que_verdad_max_cm']:.2f} cm más baja que la verdad")
        print(f"{lab}: error {r['error_cm']['media']:.2f} cm (p95 {r['error_cm']['p95']:.2f}; puntas {r['error_cm']['puntas_media']:.2f}) | "
              f"arriba/abajo {r['arriba_abajo']['acierto']:.1%} | penetración puntas {r['penetracion']['puntas_max_cm']:.2f} cm "
              f"({r['penetracion']['puntas_pct_cuadros']:.1f} %), resto {r['penetracion']['resto_max_cm']:.2f} | "
              f"deslizamiento {r['deslizamiento']['cm_por_cuadro_media']:.3f} cm/cuadro, {r['deslizamiento']['cm_por_apoyo_media']:.2f} cm/apoyo")
        print("   por pata: " + " ".join(f"{k} {v['media']:.1f}/{v['p95']:.1f}" for k, v in r["por_pata"].items()))
    if a.json:
        json.dump(res, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
