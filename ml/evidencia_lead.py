"""
Grafica de evidencia: datos temporales de varios servicios (3 por defecto) con la prediccion del modelo
final, el instante en que se enciende su alarma (instante de prediccion), el inicio real del cuello de
botella y el Prediction Lead Time de cada episodio, junto a la regla CPU > 80 % (baseline).

Usa las predicciones de la validacion cruzada (data/<resultados>/predicciones.csv): cada corrida la
predijo un modelo que no la vio al entrenar, igual que la demo en modo repeticion. No necesita Docker.

Uso (desde la raiz del proyecto):
  python ml/evidencia_lead.py                                          # rnd_lenta_06
  python ml/evidencia_lead.py --run var_inj_leve_notificaciones_r1
  python ml/evidencia_lead.py --run rnd_progresivo_08 --servicios reservaciones,pago,eventos
Genera data/<resultados>/evidencia_<run>.png y .csv (un episodio por fila) y muestra la tabla.

Lead time honesto, el mismo de la demo: solo cuenta una alarma que se ENCIENDE en los 30 s previos al
inicio del episodio tras >= 5 s apagada. Una alarma que ya estaba encendida no suma anticipacion, y a la
regla de CPU se le aplica lo mismo: la franja naranja muestra todos los segundos con CPU > 80 %.
El umbral dibujado es el del fold en que esa corrida fue de prueba (el que decidio su alarma).

Igual que la demo y ml/evidencia_vivo.py: la linea es la probabilidad calibrada (ml/calibrar.py) y se
aplica causa raiz (ml/causa_raiz.py): si el servicio estaba en alarma esperando a otro que tambien lo
estaba, la alarma pasa a la causa (franja morada "espera a <causa>") y no cuenta como aviso propio.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "ml"))
import calibrar as CAL  # noqa: E402
import causa_raiz as CR  # noqa: E402
import datos as D  # noqa: E402

SILENCIO = 5   # segundos apagada antes de un encendido para contar anticipacion (igual que demo.py)


def encendido(alarma, o):
    """Segundo en que se encendio la alarma que avisa el inicio `o` (lead honesto), o None."""
    for j in range(max(0, o - D.H), o):
        if alarma[j] and j - SILENCIO >= 0 and not alarma[j - SILENCIO:j].any():
            return j
    return None


def elegir_servicios(con_episodios, n):
    """Los servicios con episodios y, hasta completar n, los que llaman o son llamados por ellos
    (primero los mas cercanos en la cadena), y despues el resto."""
    elegidos = list(con_episodios)
    for s in con_episodios:
        llaman = sorted((c for c, deps in D.DEPENDE.items() if s in deps), key=lambda c: len(D.DEPENDE[c]))
        elegidos += [v for v in D.DEPENDE.get(s, []) + llaman if v not in elegidos]
    elegidos += [s for s in D.SERVICIOS if s not in elegidos]
    return elegidos[:max(n, len(con_episodios))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="rnd_lenta_06")
    ap.add_argument("--resultados", default="resultados_v12", help="carpeta de data/ con predicciones.csv")
    ap.add_argument("--servicios", default="", help="lista separada por comas (por defecto se eligen solos)")
    ap.add_argument("-n", type=int, default=3, help="cuantos servicios mostrar si no se da --servicios")
    a = ap.parse_args()

    res = os.path.join(HERE, "data", a.resultados)
    pred = pd.read_csv(os.path.join(res, "predicciones.csv"))
    pred = pred[pred["run"] == a.run]
    if pred.empty:
        raise SystemExit("La corrida %s no esta en %s/predicciones.csv" % (a.run, a.resultados))
    meta = json.load(open(os.path.join(HERE, "data", a.run, "meta.json")))
    m = pd.read_csv(os.path.join(HERE, "data", a.run, "metrics.csv"))
    m["sec"] = np.floor(m["timestamp"] - meta["t0_locust"]).astype(int)
    p95 = m.groupby(["servicio", "sec"])["lat_p95_ms"].mean()
    umbral_ms = {s: v["umbral_ms"] for s, v in
                 json.load(open(os.path.join(HERE, "data", "umbrales_propio.json")))["servicios"].items()}
    # episodios del etiquetado (ya fusionados): bn_ahora segundo a segundo tiene cortes dentro de un episodio
    eps = pd.read_csv(os.path.join(HERE, "data", "episodios_propio.csv"))
    eps = eps[eps["run_id"] == a.run]
    # la alarma guardada usa el umbral del fold de prueba de la corrida: se busca el que la reproduce
    cfg = json.load(open(os.path.join(res, "config.json")))
    candidatos = [u for u in cfg["umbrales_por_fold"] if ((pred["prob_lstm"] > u) == pred["alarma_lstm"]).all()]
    umbral = candidatos[0] if candidatos else None
    if len(candidatos) > 1:   # varios umbrales dan la misma alarma: hace falta saber el fold de la corrida
        from entrenar import asignar_folds
        p = cfg["parametros"]
        ds = D.cargar(os.path.join(HERE, "data"), p.get("sufijo", ""), p.get("objetivo", "y_30s"), cfg.get("excluidas", []))
        umbral = cfg["umbrales_por_fold"][asignar_folds(ds, p["folds"], p["seed_folds"])[a.run]]
    nombre = a.resultados.replace("resultados_", "")
    cal = CAL.cargar(res)
    umbral_cal = float(CAL.aplicar(cal, umbral)) if umbral is not None else None

    secs = np.arange(int(pred["t_rel"].min()), int(pred["t_rel"].max()) + 1)
    todas = pd.DataFrame({s: pred[pred["servicio"] == s].set_index("t_rel")["alarma_lstm"].reindex(secs)
                          .fillna(False).astype(bool) for s in D.SERVICIOS})
    propia, causa = CR.separar_tabla(todas)
    series, filas = {}, []
    for s in D.SERVICIOS:
        g = pred[pred["servicio"] == s].set_index("t_rel").reindex(secs)
        al = propia[s].values
        espera = causa[s].values
        al_cpu = np.nan_to_num(g["cpu"].values, nan=0.0) > 80.0
        series[s] = (CAL.aplicar(cal, g["prob_lstm"].values), al_cpu, espera)
        for e in eps[eps["servicio"] == s].itertuples():
            o = int(np.floor(e.t_inicio_rel)) - secs[0]
            if not 0 <= o < len(secs):
                continue
            j, jc = encendido(al, o), encendido(al_cpu, o)
            previas = [c for c in espera[max(0, o - D.H):o] if c]
            filas.append({"run": a.run, "servicio": s, "causa": e.causa, "inicio_real_s": int(secs[o]),
                          "fin_real_s": e.t_fin_rel,
                          "alarma_lstm_s": int(secs[j]) if j is not None else None,
                          "lead_lstm_s": int(o - j) if j is not None else None,
                          "lstm_ya_encendida": bool(j is None and o > 0 and al[o - 1]),
                          "aviso_en_causa": previas[-1] if (j is None and previas) else "",
                          "alarma_cpu_s": int(secs[jc]) if jc is not None else None,
                          "lead_cpu_s": int(o - jc) if jc is not None else None,
                          "cpu_ya_encendida": bool(jc is None and o > 0 and al_cpu[o - 1])})
    tabla = pd.DataFrame(filas, columns=["run", "servicio", "causa", "inicio_real_s", "fin_real_s", "alarma_lstm_s",
                                         "lead_lstm_s", "lstm_ya_encendida", "aviso_en_causa", "alarma_cpu_s",
                                         "lead_cpu_s", "cpu_ya_encendida"])
    for c in ("alarma_lstm_s", "lead_lstm_s", "alarma_cpu_s", "lead_cpu_s"):
        tabla[c] = tabla[c].astype("Int64")

    servicios = ([s.strip() for s in a.servicios.split(",") if s.strip()] if a.servicios else
                 elegir_servicios([s for s in D.SERVICIOS if s in set(tabla["servicio"])], a.n))
    fig, axes = plt.subplots(2 * len(servicios), 1, figsize=(12, 3.2 * len(servicios) + 1.2), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1] * len(servicios)})
    for k, s in enumerate(servicios):
        prob, al_cpu, espera = series[s]
        ep_s = tabla[tabla["servicio"] == s]
        ax_m, ax_p = axes[2 * k], axes[2 * k + 1]
        hay_espera = np.array([bool(c) for c in espera])
        ax_p.fill_between(secs, 0.95, 1.05, where=hay_espera, color="#8B6FC9", alpha=0.6, step="mid", lw=0)
        if hay_espera.any():
            i0 = int(np.argmax(hay_espera))
            ax_p.text(secs[i0], 0.92, "espera a %s" % espera[i0], va="top", fontsize=8.5, color="#8B6FC9",
                      fontweight="bold")
        lat = p95.loc[s].reindex(secs).clip(lower=1) if s in p95.index.get_level_values(0) else secs * np.nan
        ax_m.plot(secs, lat, color="tab:blue", lw=1.2)
        ax_m.axhline(umbral_ms[s], color="gray", ls="--", lw=1)
        ax_m.set_yscale("log")
        ax_m.set_ylabel("p95 (ms)")
        ax_m.set_title(s, loc="left", fontsize=11, fontweight="bold")
        ax_p.plot(secs, prob, color="tab:green", lw=1.4)
        ax_p.fill_between(secs, 0, 0.05, where=al_cpu, color="tab:orange", step="mid")
        if umbral is not None:
            ax_p.axhline(umbral_cal, color="gray", ls="--", lw=1)
        ax_p.set_ylim(0, 1.05)
        ax_p.set_ylabel("P(cuello\nen 30 s)" + ("\ncalibrada" if cal else ""))
        for ax in (ax_m, ax_p):
            ax.grid(alpha=0.3)
        for r in ep_s.itertuples():
            for ax in (ax_m, ax_p):
                ax.axvspan(r.inicio_real_s, r.fin_real_s, color="red", alpha=0.13)
                ax.axvline(r.inicio_real_s, color="red", lw=1.6)
                if not pd.isna(r.alarma_cpu_s):
                    ax.axvline(r.alarma_cpu_s, color="tab:orange", ls=":", lw=1.6)
                if not pd.isna(r.alarma_lstm_s):
                    ax.axvline(r.alarma_lstm_s, color="tab:green", ls="--", lw=1.6)
            caja = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
            cpu_txt = "CPU > 80 %%: %s" % ("%d s antes" % r.lead_cpu_s if not pd.isna(r.lead_cpu_s) else
                                           "ya encendida (no cuenta)" if r.cpu_ya_encendida else "sin aviso")
            if not pd.isna(r.alarma_lstm_s):
                medio = (r.alarma_lstm_s + r.inicio_real_s) / 2
                ax_p.annotate("", xy=(r.inicio_real_s, 0.5), xytext=(r.alarma_lstm_s, 0.5),
                              arrowprops=dict(arrowstyle="<->", color="black", lw=1.2))
                ax_p.text(medio, 0.56, "Lead Time = %d s" % r.lead_lstm_s, ha="center", va="bottom",
                          fontsize=9.5, fontweight="bold", bbox=caja)
                ax_p.text(medio, 0.42, cpu_txt, ha="center", va="top", fontsize=8.5, color="tab:orange", bbox=caja)
            else:
                estado = ("avisó la causa (%s)" % r.aviso_en_causa if r.aviso_en_causa else
                          "ya estaba encendida (no cuenta)" if r.lstm_ya_encendida else "sin aviso previo")
                ax_p.text(r.inicio_real_s + 2, 0.2, "%s: %s" % (nombre, estado), fontsize=8.5, color="dimgray",
                          va="bottom")
                ax_p.text(r.inicio_real_s + 2, 0.08, cpu_txt, fontsize=8.5, color="tab:orange", va="bottom")
    axes[-1].set_xlabel("segundos desde el inicio de la carga")
    leyenda = [Line2D([], [], color="tab:blue", label="latencia p95 (log)"),
               Line2D([], [], color="tab:green", label="probabilidad %s%s" % (nombre, " calibrada" if cal else "")),
               Line2D([], [], color="gray", ls="--", label="umbrales (p95 del servicio / alarma %s%s)"
                      % (nombre, " %.2f" % umbral_cal if umbral is not None else "")),
               Patch(color="#8B6FC9", alpha=0.6, label="alarma pasada a la causa"),
               Patch(color="red", alpha=0.13, label="cuello de botella real"),
               Line2D([], [], color="red", lw=1.6, label="inicio real"),
               Line2D([], [], color="tab:green", ls="--", lw=1.6, label="instante de predicción (%s)" % nombre),
               Line2D([], [], color="tab:orange", ls=":", lw=1.6, label="aviso de CPU > 80 % (lead honesto)"),
               Patch(color="tab:orange", label="segundos con CPU > 80 %")]
    fig.legend(handles=leyenda, loc="upper center", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1 - 0.35 / fig.get_figheight()))
    fig.suptitle("%s  (escenario %s)  ·  modelo %s: predicción vs. cuello de botella real"
                 % (a.run, meta.get("scenario", "?"), nombre), fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / fig.get_figheight()))

    out = os.path.join(res, "evidencia_%s" % a.run)
    fig.savefig(out + ".png", dpi=130)
    tabla.to_csv(out + ".csv", index=False)
    print(tabla.to_string(index=False))
    print("\nGrafica guardada en %s.png" % out)


if __name__ == "__main__":
    main()
