"""Informe del examen de F2: la tabla a/b/c y las gráficas, a partir de los resultados de cada clip.

python -m forja.calidad.informe RESULTADOS rig_info.json SALIDA_IMG SALIDA_DATOS
RESULTADOS/<clip>/ con truth_forja.json, fit_a.json, fit_b.json, fit_c.json (y fit_b3.json), examen.json, lectura*.json.
Escribe SALIDA_DATOS/f2_tabla.md, f2_resumen.json y en SALIDA_IMG f2_marcha_alturas.png, f2_error_por_pata.png y
f2_palos.png. Colores: paleta categórica validada (slots 1–3) y la verdad en tinta.
"""
import json
import pathlib
import sys

import numpy as np

from forja.calidad.examen import articulaciones
from forja.rig import Rig

CLIPS = ["marcha", "alzarse", "mordisco", "abalanzarse"]
MET = [("a_anymove_2v", "a", "fit_a.json", "(a) anymove, 2 vistas"), ("b_forja_2v", "b", "fit_b.json", "(b) Forja, 2 vistas"),
       ("c_forja_3v", "c", "fit_c.json", "(c) Forja, 3 vistas")]
EXTRA = ("b3_forja_2v_lector3", "b3", "fit_b3.json", "(b3) Forja, 2 vistas del lector de 3")
COL = {"a": "#2a78d6", "b": "#eb6834", "c": "#1baf7a", "verdad": "#0b0b0b"}
SURF, INK2, GRID = "#fcfcfb", "#52514e", "#e4e3de"


def _estilo(plt):
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": "#0b0b0b", "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10, "legend.frameon": False})


def tabla(res, metodos):
    f = lambda x, d=2: f"{x:.{d}f}".replace(".", ",")
    rows = ["| Clip | Ajuste | Error 3D medio (cm) | p95 (cm) | Puntas (cm) | Arriba/abajo | Penetración de puntas (máx cm · % cuadros) | Malla bajo z = 0 (máx cm; verdad) | Deslizamiento en apoyo (cm/cuadro · cm/apoyo) |",
            "|---|---|---|---|---|---|---|---|---|"]
    for c in CLIPS:
        if c not in res:
            continue
        for key, short, _, label in metodos:
            r = res[c].get(key)
            if r is None:
                continue
            e, ud, pen, sl = r["error_cm"], r["arriba_abajo"], r["penetracion"], r["deslizamiento"]
            ma = r.get("malla")
            mtxt = f"{f(ma['bajo_suelo_max_cm'])} ({f(ma['verdad_bajo_suelo_max_cm'])})" if ma else "—"
            rows.append(f"| {c} | {label} | {f(e['media'])} | {f(e['p95'])} | {f(e['puntas_media'])} | "
                        f"{f(ud['acierto'] * 100, 1)} % | {f(pen['puntas_max_cm'])} · {f(pen['puntas_pct_cuadros'], 1)} % | {mtxt} | "
                        f"{f(sl['cm_por_cuadro_media'], 3)} · {f(sl['cm_por_apoyo_media'])} |")
    # total: media de los cuatro clips
    for key, short, _, label in metodos:
        rr = [res[c][key] for c in CLIPS if c in res and key in res[c]]
        if not rr:
            continue
        m = lambda g: float(np.mean([g(r) for r in rr]))
        rows.append(f"| **los 4 clips** | **{label}** | **{f(m(lambda r: r['error_cm']['media']))}** | **{f(m(lambda r: r['error_cm']['p95']))}** | "
                    f"{f(m(lambda r: r['error_cm']['puntas_media']))} | — | {f(max(r['penetracion']['puntas_max_cm'] for r in rr))} máx | — | "
                    f"{f(m(lambda r: r['deslizamiento']['cm_por_cuadro_media']), 3)} · {f(m(lambda r: r['deslizamiento']['cm_por_apoyo_media']))} |")
    return "\n".join(rows)


def tabla_golpes(res, metodos):
    f = lambda x, d=2: f"{x:.{d}f}".replace(".", ",")
    rows = ["| Clip | Ajuste | Pico de colmillos conservado | Pico del cuerpo conservado | Temblor (cm/cuadro²) | Malla: lo más que baja respecto a la verdad (cm) |",
            "|---|---|---|---|---|---|"]
    for c in CLIPS:
        for key, short, _, label in metodos:
            r = res.get(c, {}).get(key)
            if r is None:
                continue
            g = r.get("golpes", {})
            ma = r.get("malla")
            rows.append(f"| {c} | {label} | {f(g['colmillos']['proporcion'] * 100, 0)} % | {f(g['cuerpo']['proporcion'] * 100, 0)} % | "
                        f"{f(r.get('temblor_cm_cuadro2', float('nan')))} | {f(ma['mas_baja_que_verdad_max_cm']) if ma else '—'} |")
    return "\n".join(rows)


def tabla_patas(res, metodos):
    legs = None
    out = []
    for key, short, _, label in metodos:
        rr = [res[c][key] for c in CLIPS if c in res and key in res[c]]
        if not rr:
            continue
        legs = list(rr[0]["por_pata"].keys())
        vals = {L: (np.mean([r["por_pata"][L]["media"] for r in rr]), np.mean([r["por_pata"][L]["p95"] for r in rr])) for L in legs}
        out.append((label, vals))
    f = lambda x: f"{x:.1f}".replace(".", ",")
    rows = ["| Ajuste | " + " | ".join(legs) + " |", "|---|" + "---|" * len(legs)]
    for label, vals in out:
        rows.append(f"| {label} | " + " | ".join(f"{f(vals[L][0])} ({f(vals[L][1])})" for L in legs) + " |")
    return "\n".join(rows), out


def figuras(base, rig, img, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _estilo(plt)
    names = rig.kp_names
    # 1. alturas de las puntas en la marcha
    d = base / "marcha"
    tr = json.load(open(d / "truth_forja.json"))
    tips = [names.index(n) for n in tr["tips"]]
    h0 = np.array(tr["h0"])
    A = np.array(tr["joints"])
    st = np.array(tr["stance"], bool)
    fits = {s: articulaciones(d / fn, rig)[0] for _, s, fn, _ in MET if (d / fn).exists()}
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.4), sharex=True, sharey=True)
    order = sorted(range(len(tr["legs"])), key=lambda i: (0 if tr["legs"][i][0] == "L" else 1, int(tr["legs"][i][1])))
    for ax, k in zip(axs.ravel(), order):
        t = np.arange(len(A))
        ys = st[:, k]
        edges = np.flatnonzero(np.diff(np.r_[0, ys.astype(int), 0]))
        for s0, s1 in zip(edges[::2], edges[1::2]):
            ax.axvspan(s0 - 0.5, s1 - 0.5, color="#efeeea", lw=0)
        ax.axhline(1.0, color=INK2, lw=0.8, ls=":")
        ax.plot(t, (A[:, tips[k], 2] - h0[k]) * 100, color=COL["verdad"], lw=2.2, label="verdad")
        for s in ("a", "b", "c"):
            if s in fits:
                ax.plot(t, (fits[s][:, tips[k], 2] - h0[k]) * 100, color=COL[s], lw=1.6,
                        label=dict((x[1], x[3]) for x in MET)[s])
        r = {s: res["marcha"][key]["arriba_abajo"]["por_pata"][tr["legs"][k]] for key, s, _, _ in MET if key in res["marcha"]}
        ax.set_title(f"{tr['legs'][k]}  ·  arriba/abajo " + "  ".join(f"{s} {v:.0%}" for s, v in r.items()), fontsize=9, loc="left")
        ax.set_ylim(-4, 14)
    fig.supylabel("altura de la punta sobre su apoyo (cm)", color=INK2, fontsize=10)
    for ax in axs[1]:
        ax.set_xlabel("cuadro (24 fps)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Marcha de 8 patas: altura de cada punta (franjas grises: apoyo verdadero; punteado: umbral de 1 cm)",
                 y=0.94, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(img / "f2_marcha_alturas.png", dpi=110)
    plt.close(fig)
    # 2. error por pata (media de los 4 clips)
    _, rows = tabla_patas(res, MET)
    legs = list(rows[0][1].keys())
    lord = sorted(legs, key=lambda L: (0 if L[0] == "L" else 1, int(L[1])))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    w = 0.26
    x = np.arange(len(lord))
    for i, (label, vals) in enumerate(rows):
        s = MET[i][1]
        ax.bar(x + (i - 1) * w, [vals[L][0] for L in lord], w - 0.03, color=COL[s], label=label)
    ax.axhline(2.0, color=INK2, lw=1, ls="--")
    ax.text(len(lord) - 0.5, 2.05, "meta F2: 2 cm", ha="right", va="bottom", color=INK2, fontsize=9)
    ax.set_xticks(x, lord)
    ax.set_ylabel("error 3D medio por pata (cm)")
    ax.grid(axis="x", visible=False)
    h, l = ax.get_legend_handles_labels()
    fig.suptitle("Error medio de cada pata (cadera, rodilla, tobillo y punta), media de los 4 clips", x=0.01, y=0.985,
                 ha="left", fontsize=11)
    fig.legend(h, l, loc="upper left", ncol=3, bbox_to_anchor=(0.01, 0.92))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(img / "f2_error_por_pata.png", dpi=110)
    plt.close(fig)
    # 3. figuras de palos: un cuadro clave por clip, lado y arriba
    key_frames = {"marcha": 40, "alzarse": 60, "mordisco": 86, "abalanzarse": 38}
    fig, axs = plt.subplots(2, 4, figsize=(16, 8.4))
    for ci, c in enumerate(CLIPS):
        d = base / c
        if not (d / "truth_forja.json").exists():
            continue
        tr = json.load(open(d / "truth_forja.json"))
        A = np.array(tr["joints"])
        f = key_frames[c]
        sets = [("verdad", A)] + [(s, articulaciones(d / fn, rig)[0]) for _, s, fn, _ in MET if (d / fn).exists()]
        for row, (i0, i1, sx, vista) in enumerate([(1, 2, -1, "lado"), (0, 1, 1, "arriba")]):
            ax = axs[row, ci]
            for lab, Jx in sets:
                for b in rig.info["bones"]:
                    h, t_ = names.index(b["head_joint"]), names.index(b["tail_joint"])
                    ax.plot([sx * Jx[f, h, i0], sx * Jx[f, t_, i0]], [Jx[f, h, i1], Jx[f, t_, i1]], color=COL[lab],
                            lw=2.6 if lab == "verdad" else 1.5, alpha=1.0 if lab == "verdad" else 0.9, solid_capstyle="round")
            if row == 0:
                ax.axhline(0, color=INK2, lw=0.8)
                ax.set_ylim(-0.05, 1.3)
            ax.set_aspect("equal")
            ax.set_title(f"{c}, cuadro {f}, {vista}", loc="left", fontsize=10)
            ax.tick_params(labelsize=8)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COL["verdad"], lw=2.6, label="verdad")] + [Line2D([], [], color=COL[s], lw=1.5, label=l) for _, s, _, l in MET]
    fig.legend(handles=handles, loc="upper center", ncol=4)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(img / "f2_palos.png", dpi=100)
    plt.close(fig)


def main():
    base, info, img, datos = (pathlib.Path(x) for x in sys.argv[1:5])
    rig = Rig(info)
    res = {c: json.load(open(base / c / "examen.json")) for c in CLIPS if (base / c / "examen.json").exists()}
    lect = {c: {k: json.load(open(base / c / f"lectura{k}.json")) for k in ("2", "3") if (base / c / f"lectura{k}.json").exists()}
            for c in res}
    img.mkdir(parents=True, exist_ok=True)
    datos.mkdir(parents=True, exist_ok=True)
    met = MET + [EXTRA]
    tp, _ = tabla_patas(res, met)
    (datos / "f2_tabla.md").write_text(tabla(res, met) + "\n\nError medio por pata, media de los 4 clips (entre paréntesis, p95), cm:\n\n" + tp
                                       + "\n\nGolpes, temblor y malla:\n\n" + tabla_golpes(res, met) + "\n", encoding="utf-8")
    json.dump({"examen": res, "lectura": lect}, open(datos / "f2_resumen.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    figuras(base, rig, img, res)
    print((datos / "f2_tabla.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
