"""
Grafica de evidencia para una corrida NUEVA, de las que no estan en las predicciones guardadas de la
validacion cruzada (para esas, usar ml/evidencia_lead.py). La etiqueta con las mismas reglas del
entrenamiento, le aplica el modelo final y dibuja la latencia de todos los servicios y una fila por
servicio con su probabilidad, su cuello de botella real, su instante de prediccion y su Lead Time.

Uso (desde la raiz del proyecto, no necesita Docker):
  python ml/evidencia_vivo.py --run demo_progresivo_1
  python ml/evidencia_vivo.py --run demo_progresivo_1 --servicios notificaciones,ticket,pago
Genera data/<run>/evidencia_v12.png y .csv (un episodio por fila) y muestra la tabla. Ademas genera
data/<run>/cadena_v12.png: el trafico y, por servicio en el orden de la cadena de llamadas (reservaciones ->
pago -> ticket -> notificaciones; luego usuarios y eventos), el tiempo de cada peticion partido en trabajo propio
y espera al servicio que llama. Muestra como la lentitud de uno se transmite a los que lo llaman.

Las filas salen ordenadas por el momento en que cada servicio entro en problemas. En cada una, la zona
rosa es el cuello de botella PROPIO del servicio y la zona gris son los segundos en que estuvo lento
solo por esperar a otro servicio (la etiqueta "afectado" de etiquetar.py). El etiquetado se hace en una
carpeta temporal con las corridas normal_base, asi los umbrales son los del entrenamiento y no se toca el
dataset. Lead time honesto, el de datos.lead_times: solo cuenta una alarma que se ENCIENDE en los 30 s
previos al inicio tras >= 5 s apagada. Las predicciones se calculan despues de la corrida, sobre los
datos grabados, y empiezan a los 60 s de carga, como en la demo (antes esta "calentando").

Igual que la demo, aplica dos ajustes sin reentrenar (ver ml/causa_raiz.py y ml/calibrar.py):
  - causa raiz: si un servicio y alguno de los que llama estan en alarma, el aviso queda en el causante y
    el que llama muestra una franja morada "espera a <causa>" (no cuenta como aviso propio).
  - probabilidad calibrada: la linea de cada servicio es una probabilidad real (cuando marca 0,7 el
    problema llega ~70 % de las veces). La alarma sigue siendo la del modelo: puntaje > 0,90, que
    equivale a una probabilidad de ~0,51 (la linea punteada).
"""
import argparse
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(HERE, "ml"), os.path.join(HERE, "collector"), HERE]
import calibrar as CAL  # noqa: E402
import causa_raiz as CR  # noqa: E402
import datos as D  # noqa: E402
import demo  # noqa: E402
import run_campaign as RC  # noqa: E402
import verificar_vivo as V  # noqa: E402

BASES = ["normal_base_r1", "normal_base_r2", "normal_base_r3", "normal_base_r4"]
ARCHIVOS = ["metrics.csv", "edges.csv", "events.csv", "meta.json"]
COLOR = {"ticket": "#1F5FBF", "pago": "#008300", "reservaciones": "#6250D6",
         "notificaciones": "#1BAF7A", "usuarios": "#EDA100", "eventos": "#E87BA4"}
ROJO, NARANJA, FONDO, TINTA, TENUE, GUIA, GRIS, MORADO = ("#C0322B", "#C8590A", "#FBFAF6", "#14213D", "#4A5563",
                                                          "#6B7686", "#C9CED6", "#8B6FC9")


def etiquetar_aparte(run):
    """Consolida y etiqueta la corrida en una carpeta temporal, junto a las normal_base que fijan los umbrales."""
    tmp = tempfile.mkdtemp(prefix="evidencia_")
    for r in dict.fromkeys(BASES + [run]):
        os.makedirs(os.path.join(tmp, r))
        for f in ARCHIVOS:
            origen = os.path.join(HERE, "data", r, f)
            if os.path.exists(origen):
                shutil.copy(origen, os.path.join(tmp, r, f))
        open(os.path.join(tmp, r, "OK"), "w").write("ok")
    RC.DATA = tmp
    with contextlib.redirect_stdout(io.StringIO()):
        RC.consolidar()
    subprocess.run([sys.executable, os.path.join(HERE, "ml", "etiquetar.py"), "--data", tmp, "--solo-propio",
                    "--sufijo", "_propio"], cwd=HERE, check=True, stdout=subprocess.DEVNULL)
    ruta = os.path.join(tmp, "episodios_propio.csv")
    try:
        pd.read_csv(ruta)
    except pd.errors.EmptyDataError:   # sin ningun episodio etiquetar.py deja el archivo vacio y datos.cargar falla
        open(ruta, "w").write("run_id,servicio,t_inicio_rel,t_fin_rel,causa,previo_libre_s\n")
    return tmp


def clave(m):
    """Lo que define una corrida: dos corridas con la misma clave son el mismo escenario."""
    esc = m.get("scenario")
    c = (esc, m.get("duration"), tuple(m.get("inject") or []), m.get("base_users"))
    if esc != "normal":
        c += (m.get("peak_users"),)
    if esc == "burst":
        c += (m.get("burst_times") or "", m.get("burst_len") or 25, m.get("burst_ramp") or 0)
    return c


def aviso_entrenamiento(run, meta):
    """Avisa si el modelo final ya vio esta corrida o una con la misma configuracion."""
    res = os.path.dirname(V.MODELO)
    vistas = set(pd.read_csv(os.path.join(res, "predicciones.csv"), usecols=["run"])["run"])
    if run in vistas:
        return "esta corrida se uso para entrenar; para una medida honesta usa ml/evidencia_lead.py"
    for v in sorted(vistas):
        if clave(json.load(open(os.path.join(HERE, "data", v, "meta.json")))) == clave(meta):
            return "repite la configuracion de %s, que se uso para entrenar: datos nuevos, escenario conocido" % v
    return ""


def lead_por_episodio(ds, alarma):
    """Lead honesto de cada fila de ds.eps (datos.lead_times omite los episodios con poca ventana libre)."""
    eps, salida = ds.eps, {}
    for i in eps.index:
        ds.eps = eps.loc[[i]]
        t = D.lead_times(ds, alarma)
        salida[i] = float(t["lead_s"].iloc[0]) if len(t) else np.nan
    ds.eps = eps
    return salida


def aviso_en_causa(ds, run, srv, ini, alarma):
    """Si el servicio no aviso por si mismo: servicio causante (siguiendo la cadena de dependencias) que tuvo
    alarma en los 30 s previos al inicio, o "" si ninguno."""
    k = ds.run == run
    en = {s: bool(alarma[k & (ds.srv == s) & (ds.sec >= ini - D.H) & (ds.sec < ini)].any()) for s in D.SERVICIOS}
    return CR.raiz(srv, en) or ""


def armar_tabla(ds, run, episodios, alarma, alarma_todas, cpu, nombre):
    """`alarma` = aviso propio del servicio (despues de causa raiz); `alarma_todas` = todas las del modelo."""
    lv, lc = lead_por_episodio(ds, alarma), lead_por_episodio(ds, cpu)
    alarma_s, lead_s = "alarma_%s_s" % nombre, "lead_%s_s" % nombre
    columnas = ["run", "servicio", "causa", "inicio_real_s", "fin_real_s", alarma_s, lead_s, "aviso_en_causa",
                "lead_cpu_s"]
    filas = []
    for i, e in ds.eps.iterrows():
        ini = int(ds.sec[e["gi"]])
        x = episodios[(episodios["servicio"] == e["srv"]) & (np.floor(episodios["t_inicio_rel"]) == ini)].iloc[0]
        filas.append({"run": run, "servicio": e["srv"], "causa": e["causa"], "inicio_real_s": ini,
                      "fin_real_s": x["t_fin_rel"], alarma_s: ini - lv[i], lead_s: lv[i],
                      "aviso_en_causa": "" if lv[i] == lv[i] else aviso_en_causa(ds, run, e["srv"], ini, alarma_todas),
                      "lead_cpu_s": lc[i]})
    tabla = pd.DataFrame(filas, columns=columnas).sort_values("inicio_real_s").reset_index(drop=True)
    for c in (alarma_s, lead_s, "lead_cpu_s"):
        tabla[c] = pd.to_numeric(tabla[c]).astype("Int64")
    return tabla


def resumen_fila(eps_servicio, alarma, secs, nombre):
    """Texto a la derecha de la fila de un servicio: un renglon por cada cuello de botella."""
    if eps_servicio.empty:
        return "sin cuello" + chr(10) + "de botella"
    lineas = []
    for _, r in eps_servicio.iterrows():
        ini, lead, o = r["inicio_real_s"], r["lead_%s_s" % nombre], r["inicio_real_s"] - secs[0]
        if not pd.isna(lead):
            estado = "avisó %d s antes" % lead
        elif r.get("aviso_en_causa"):
            estado = "avisó la causa" + chr(10) + "  (%s)" % r["aviso_en_causa"]
        else:
            estado = "ya encendida" if 0 < o <= len(secs) and alarma[o - 1] else "sin aviso"
        lineas.append("%d s: %s" % (ini, estado))
    if len(lineas) > 4:
        lineas = lineas[:3] + ["y %d más" % (len(lineas) - 3)]
    return chr(10).join(lineas)


def franjas(mask):
    """Tramos [inicio, fin] (en posiciones) donde mask es True."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def dibujar(ds, run, meta, prob, umbral, propia, esperando, causa, afectado, servicios, tabla, nombre, salida):
    """prob = probabilidad que se muestra (calibrada) y umbral su corte; propia/esperando/causa de causa_raiz."""
    k = ds.run == run
    secs = np.arange(0, int(ds.sec[k & np.isfinite(prob)].max()) + 1)

    def serie(valores, s):
        m = k & (ds.srv == s)
        return pd.Series(valores[m], index=ds.sec[m]).reindex(secs)

    alarma_s = "alarma_%s_s" % nombre
    con_problema = list(dict.fromkeys(s for s in tabla["servicio"] if s in servicios))   # la tabla va por tiempo
    filas = con_problema + [s for s in servicios if s not in con_problema]
    n, alto_lat, alto_fila, sup, hueco, inf = len(filas), 2.3, 1.05, 0.6, 0.45, 1.7
    alto = sup + alto_lat + hueco + n * alto_fila + inf

    plt.rcParams.update({"font.size": 15, "font.family": "DejaVu Sans", "text.color": TINTA, "axes.labelcolor": TINTA,
                         "xtick.color": TINTA, "ytick.color": TINTA, "axes.edgecolor": "#B9BEC6"})
    fig = plt.figure(figsize=(10.4, alto), facecolor=FONDO)
    gs = fig.add_gridspec(2, 1, height_ratios=[alto_lat, n * alto_fila], left=0.2, right=0.795, top=1 - sup / alto,
                          bottom=inf / alto, hspace=hueco / ((alto_lat + n * alto_fila) / 2))
    sub = gs[1].subgridspec(n, 1, hspace=0.14)

    a_lat = fig.add_subplot(gs[0])
    a_lat.set_facecolor(FONDO)
    a_lat.spines[["top", "right"]].set_visible(False)
    for s in [x for x in servicios if x not in con_problema] + con_problema:
        # X ya trae la p95 con los huecos rellenados; 0 significa que no hubo peticiones en ese segundo
        lat = serie(np.expm1(ds.X[:, D.COLUMNAS.index("lat_p95_ms")]), s)
        a_lat.plot(secs, lat.where(lat > 0).ffill(limit=8), color=COLOR[s], lw=2.2 if s in con_problema else 1.2)
    a_lat.set_yscale("log")
    a_lat.set_yticks([10, 100, 1000])
    a_lat.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    a_lat.yaxis.set_minor_formatter(NullFormatter())
    a_lat.set_ylabel("latencia p95 (ms)", fontsize=14)
    a_lat.set_xlim(secs[0], secs[-1])
    a_lat.tick_params(labelbottom=False)
    a_lat.text(0, 1.06, "%s  ·  carga %s" % (run, meta.get("scenario", "?")), transform=a_lat.transAxes, va="bottom",
               fontweight="bold")
    a_lat.legend(handles=[Line2D([], [], color=COLOR[s], lw=2.6 if s in con_problema else 1.8, label=s) for s in servicios],
                 loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=12.5, handlelength=1.3)

    for i, s in enumerate(filas):
        ax = fig.add_subplot(sub[i], sharex=a_lat)
        ax.set_facecolor(FONDO)
        ax.spines[["top", "right"]].set_visible(False)
        p = serie(prob, s)
        eps_s = tabla[tabla["servicio"] == s]
        espera = afectado.loc[s].reindex(secs).fillna(0).astype(bool).values
        ax.fill_between(secs, 0, 1.05, where=espera, step="mid", color=GRIS, alpha=0.55, lw=0)
        for _, r in eps_s.iterrows():
            ax.axvspan(r["inicio_real_s"], r["fin_real_s"], color=ROJO, alpha=0.16, lw=0)
            ax.axvline(r["inicio_real_s"], color=ROJO, lw=2)
            if not pd.isna(r[alarma_s]):
                ax.axvline(r[alarma_s], color=TINTA, ls="--", lw=1.8)
        ax.axhline(umbral, color=GUIA, ls=":", lw=1.3)
        ax.fill_between(secs, 0, 0.1, where=np.nan_to_num(serie(ds.cpu, s).values) > 80, color=NARANJA, step="mid", lw=0)
        espera_s = serie(esperando.astype(float), s).fillna(0).values > 0
        causa_s = serie(causa, s).values
        ax.fill_between(secs, 0.93, 1.05, where=espera_s, step="mid", color=MORADO, alpha=0.6, lw=0)
        ultimo, ya = -1e9, set()
        for a0, a1 in franjas(espera_s):
            # una etiqueta por causa y fila, separadas lo que ocupa el texto (~30 % del eje)
            if a1 - a0 >= 2 and causa_s[a0] not in ya and secs[a0] - ultimo >= 0.3 * len(secs):
                ax.text(secs[a0], 0.9, "espera a %s" % causa_s[a0], va="top", fontsize=10, color=MORADO,
                        fontweight="bold")
                ultimo = secs[a0]
                ya.add(causa_s[a0])
        ax.plot(secs, p.ffill(limit=8), color=COLOR[s], lw=2)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 1])
        ax.tick_params(axis="y", labelsize=12)
        ax.tick_params(labelbottom=i == n - 1)
        ax.set_ylabel(s, rotation=0, ha="right", va="center", fontsize=13.5, labelpad=10,
                      fontweight="bold" if s in con_problema else "normal")
        propia_s = serie(propia.astype(float), s).fillna(0).values > 0
        ax.text(1.02, 0.5, resumen_fila(eps_s, propia_s, secs, nombre), transform=ax.transAxes, va="center",
                fontsize=12, clip_on=False)
        if i == 0:
            ax.text(demo.CALENTAR / 2, 0.25, "calentando", ha="center", va="center", fontsize=11, color=TENUE)
            ax.text(0, 1.1, "Probabilidad calibrada de cuello de botella en 30 s, por servicio", transform=ax.transAxes,
                    va="bottom", fontsize=13.5, color=TENUE)
            ax.text(1.02, 1.1, "inicio real: aviso", transform=ax.transAxes, va="bottom", fontsize=12,
                    fontweight="bold", clip_on=False)
        if i == n - 1:
            ax.set_xlabel("segundos desde el inicio de la carga")
    fig.legend(handles=[Patch(color=ROJO, alpha=0.16, label="cuello de botella propio"),
                        Patch(color=GRIS, alpha=0.55, label="lento, esperando a otro servicio"),
                        Line2D([], [], color=ROJO, lw=2, label="inicio real"),
                        Line2D([], [], color=TINTA, ls="--", lw=1.8, label="predicción: se enciende la alarma"),
                        Line2D([], [], color=GUIA, ls=":", lw=1.3, label="umbral de alarma (%s)" % ("%.2f" % umbral).replace(".", ",")),
                        Patch(color=NARANJA, label="CPU > 80 %"),
                        Patch(color=MORADO, alpha=0.6, label="alarma pasada a la causa")],
               loc="lower center", ncol=3, frameon=False, fontsize=12, handlelength=1.4, columnspacing=1.5)
    fig.savefig(salida, dpi=200, facecolor=FONDO)
    plt.close(fig)


CADENA = ["reservaciones", "pago", "ticket", "notificaciones"]   # cada uno llama al siguiente (y reservaciones,
OTROS = ["usuarios", "eventos"]                                   # ademas, a usuarios y eventos)


def tiempos_por_servicio(carpeta, t0):
    """Por servicio y segundo: tiempo medio por peticion, la parte que espera a los servicios que llama, la p95 y
    la CPU. La espera de cada llamada es lo que midio el que llama (edges.csv). Si el que llama esta mas ocupado que
    el llamado (mas CPU), la espera se limita a lo que tardo el llamado (su tiempo medio): sus llamadas parecen
    lentas aunque el otro responda rapido, y ese resto es trabajo propio. Si el llamado es el saturado, su cola
    (antes de que el mismo empiece a medir) sigue contando como espera. Suavizado con una media movil de 3 s."""
    m = pd.read_csv(os.path.join(carpeta, "metrics.csv"))
    m["sec"] = np.floor(m["timestamp"] - t0).astype(int)
    e = pd.read_csv(os.path.join(carpeta, "edges.csv"))
    e["sec"] = np.floor(e["timestamp"] - t0).astype(int)
    tot = m.groupby(["servicio", "sec"])[["lat_mean_ms", "lat_p95_ms", "cpu"]].mean()
    arista = e.groupby(["origen", "destino", "sec"])["lat_mean_ms"].mean()
    partes = []
    servicios = set(tot.index.get_level_values(0))
    for (o, d), g in arista.groupby(level=["origen", "destino"]):
        g = g.droplevel([0, 1])
        if d not in servicios or o not in servicios:
            partes.append(pd.DataFrame({"origen": o, "sec": g.index, "ms": g.values}))
            continue
        llamado = tot.loc[d, "lat_mean_ms"].reindex(g.index).fillna(g)
        mas_ocupado = (tot.loc[o, "cpu"].reindex(g.index).fillna(0) > tot.loc[d, "cpu"].reindex(g.index).fillna(0)).values
        ms = np.where(mas_ocupado, np.minimum(g.values, llamado.values), g.values)
        partes.append(pd.DataFrame({"origen": o, "sec": g.index, "ms": ms}))
    esp = pd.concat(partes).groupby(["origen", "sec"])["ms"].sum() if partes else pd.Series(dtype=float)
    out = {}
    for s in CADENA + OTROS:
        if s not in tot.index.get_level_values(0):
            continue
        t = tot.loc[s]
        t = t.reindex(range(int(t.index.min()), int(t.index.max()) + 1))
        w = esp.loc[s].reindex(t.index) if s in esp.index.get_level_values(0) else pd.Series(0.0, index=t.index)
        total = t["lat_mean_ms"].interpolate(limit=5).fillna(0)
        espera = np.minimum(w.interpolate(limit=5).fillna(0), total)
        out[s] = pd.DataFrame({"total": total.rolling(3, center=True, min_periods=1).mean(),
                               "espera": espera.rolling(3, center=True, min_periods=1).mean(),
                               "p95": t["lat_p95_ms"], "cpu": t["cpu"]})
        out[s]["propio"] = (out[s]["total"] - out[s]["espera"]).clip(lower=0)
    return out


def dibujar_cadena(ds, run, meta, carpeta, propia, esperando, causa, tabla, nombre, salida):
    """Como se propaga el problema: trafico arriba y, por servicio en el orden de la cadena, el tiempo de cada
    peticion partido en trabajo propio y espera al servicio que llama; con el inicio real, el aviso del modelo y
    las alarmas pasadas a la causa."""
    t0 = meta["t0_locust"]
    tiempos = tiempos_por_servicio(carpeta, t0)
    umbral_ms = {s: v["umbral_ms"] for s, v in
                 json.load(open(os.path.join(HERE, "data", "umbrales_propio.json")))["servicios"].items()}
    filas = [s for s in CADENA + OTROS if s in tiempos]
    k = ds.run == run
    xmax = max(int(ds.sec[k].max()), max(int(t.index.max()) for t in tiempos.values()))
    secs = np.arange(0, xmax + 1)
    ymax = 1.1 * max(tiempos[s]["total"].max() for s in filas)
    alturas = [1.5] + [1.25 if s in CADENA else 0.75 for s in filas]
    sup, inf, hueco = 0.95, 1.75, 0.18
    alto = sup + inf + sum(alturas) + hueco * len(alturas)
    plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans", "text.color": TINTA, "axes.labelcolor": TINTA,
                         "xtick.color": TINTA, "ytick.color": TENUE, "axes.edgecolor": "#B9BEC6"})
    fig = plt.figure(figsize=(10.4, alto), facecolor=FONDO)
    gs = fig.add_gridspec(len(alturas), 1, height_ratios=alturas, left=0.2, right=0.795, top=1 - sup / alto,
                          bottom=inf / alto, hspace=hueco * len(alturas) / sum(alturas))

    # trafico
    a_tr = fig.add_subplot(gs[0])
    a_tr.set_facecolor(FONDO)
    a_tr.spines[["top", "right"]].set_visible(False)
    hist = os.path.join(carpeta, "locust_stats_history.csv")
    if os.path.exists(hist):
        h = pd.read_csv(hist)
        h = h[h["Name"] == "Aggregated"]
        a_tr.plot(h["Timestamp"] - t0, h["User Count"], color=TINTA, lw=2.2, drawstyle="steps-post")
    a_tr.set_ylim(bottom=0)
    a_tr.set_xlim(0, xmax)
    a_tr.tick_params(labelbottom=False, length=0)
    a_tr.set_ylabel("tráfico", rotation=0, ha="right", va="center", fontsize=13.5, labelpad=10, fontweight="bold")
    a_tr.text(1.02, 0.5, "usuarios" + chr(10) + "simultáneos", transform=a_tr.transAxes, va="center", fontsize=12)
    fig.text(0.03, 1 - 0.3 / alto, "%s  ·  carga %s: cómo se propaga el problema" % (run, meta.get("scenario", "?")),
             va="top", fontweight="bold", fontsize=16)

    fallas = {}
    for inj in meta.get("inject") or []:
        p = inj.split(":")
        fallas.setdefault(p[0], []).append((float(p[1]), float(p[1]) + float(p[2])))
    ejes = []
    for i, s in enumerate(filas):
        ax = fig.add_subplot(gs[i + 1], sharex=a_tr)
        ejes.append(ax)
        ax.set_facecolor(FONDO)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0)
        t = tiempos[s].reindex(secs)
        x = t.index.values
        ax.fill_between(x, 0, t["propio"].fillna(0), color=COLOR[s], alpha=0.95, lw=0, step="mid", zorder=2)
        ax.fill_between(x, t["propio"].fillna(0), t["total"].fillna(0), color=COLOR[s], alpha=0.28, lw=0, step="mid",
                        zorder=2)
        ax.set_ylim(0, ymax)
        paso = 1000 if ymax > 1500 else (250 if ymax > 600 else 200)
        ax.set_yticks(np.arange(0, ymax, paso))
        ax.set_yticklabels([("%g s" % (v / 1000)).replace(".", ",") if v else "0" for v in np.arange(0, ymax, paso)],
                           fontsize=11)
        ax.grid(axis="y", color="#E4E1D8", lw=1, zorder=0)
        ax.fill_between(x, 0, ymax * 0.06, where=np.nan_to_num(t["cpu"].values) > 80, color=NARANJA, step="mid",
                        lw=0, zorder=3)
        # cuello de botella propio, aviso del modelo y alarmas pasadas a la causa (de la misma tabla y reglas)
        for _, r in tabla[tabla["servicio"] == s].iterrows():
            ax.axvspan(r["inicio_real_s"], r["fin_real_s"], color=ROJO, alpha=0.12, lw=0, zorder=1)
            ax.axvline(r["inicio_real_s"], color=ROJO, lw=2, zorder=4)
            if not pd.isna(r["alarma_%s_s" % nombre]):
                ax.axvline(r["alarma_%s_s" % nombre], color=TINTA, ls="--", lw=1.8, zorder=4)
        m = k & (ds.srv == s)
        esp_s = pd.Series(esperando[m].astype(float), index=ds.sec[m]).reindex(secs).fillna(0).values > 0
        cau_s = pd.Series(causa[m], index=ds.sec[m]).reindex(secs).values
        ax.fill_between(secs, ymax * 0.92, ymax, where=esp_s, color=MORADO, alpha=0.6, step="mid", lw=0, zorder=3)
        if esp_s.any():
            j = int(np.argmax(esp_s))
            ax.text(secs[j], ymax * 0.88, "alarma → %s" % cau_s[j], va="top", fontsize=10, color=MORADO,
                    fontweight="bold", zorder=5)
        for a0, a1 in fallas.get(s, []):
            ax.plot([a0, a1], [ymax * 0.97] * 2, color=TINTA, lw=3, solid_capstyle="butt", zorder=5)
            ax.text(a1 + 3, ymax * 0.97, "falla inyectada", va="center", fontsize=10, color=TINTA, zorder=5)
        # resumen: de los segundos lentos (p95 sobre su umbral), cuanto fue espera
        lento = t["p95"] > umbral_ms[s]
        if lento.sum() >= 3:
            frac = t.loc[lento, "espera"].sum() / max(t.loc[lento, "total"].sum(), 1e-9)
            nota = ("lento %d s:" % lento.sum()) + chr(10) + ("%d %% esperando" % round(100 * frac) if s != "notificaciones"
                                                              and s in D.DEPENDE else "trabajo propio")
        else:
            nota = "nunca lento"
        ax.text(1.02, 0.5, nota, transform=ax.transAxes, va="center", fontsize=12)
        ax.set_ylabel(s, rotation=0, ha="right", va="center", fontsize=13.5, labelpad=10,
                      fontweight="bold" if s in set(tabla["servicio"]) else "normal")
        ax.tick_params(labelbottom=i == len(filas) - 1)
    ejes[-1].set_xlabel("segundos desde el inicio de la carga")
    fig.canvas.draw()
    for a, b in zip(ejes[:3], ejes[1:4]):   # flechas "llama a" entre las filas de la cadena
        ya, yb = a.get_position().y0, b.get_position().y1
        fig.text(0.03, (ya + yb) / 2, "↓ llama a", fontsize=11, color=TENUE, va="center")
    if len(ejes) > 4:
        fig.text(0.03, (ejes[3].get_position().y0 + ejes[4].get_position().y1) / 2,
                 "reservaciones también llama a:", fontsize=11, color=TENUE, va="center")
    fig.legend(handles=[Patch(color=GUIA, alpha=0.95, label="trabajo propio (color sólido)"),
                        Patch(color=GUIA, alpha=0.28, label="esperando al servicio que llama"),
                        Patch(color=ROJO, alpha=0.12, label="cuello de botella propio"),
                        Line2D([], [], color=ROJO, lw=2, label="inicio real"),
                        Line2D([], [], color=TINTA, ls="--", lw=1.8, label="aviso del modelo"),
                        Patch(color=MORADO, alpha=0.6, label="alarma pasada a la causa"),
                        Patch(color=NARANJA, label="CPU > 80 %")],
               loc="lower center", ncol=3, frameon=False, fontsize=12, handlelength=1.4, columnspacing=1.5)
    fig.text(0.2, 0.5 * inf / alto + 0.012, "Cada fila: tiempo medio por petición, en la misma escala para todos.",
             fontsize=11.5, color=TENUE, va="top")
    fig.savefig(salida, dpi=200, facecolor=FONDO)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="carpeta de data/ con la corrida")
    ap.add_argument("--servicios", default="", help="lista separada por comas (por defecto, los 6)")
    a = ap.parse_args()

    carpeta = os.path.join(HERE, "data", a.run)
    if not os.path.exists(os.path.join(carpeta, "metrics.csv")):
        raise SystemExit("No existe data/%s/metrics.csv" % a.run)
    pedidos = [s.strip() for s in a.servicios.split(",") if s.strip()]
    desconocidos = sorted(set(pedidos) - set(COLOR))
    if desconocidos:
        raise SystemExit("Servicios desconocidos: %s" % ", ".join(desconocidos))
    servicios = [s for s in COLOR if not pedidos or s in pedidos]

    meta = json.load(open(os.path.join(carpeta, "meta.json")))
    tmp = etiquetar_aparte(a.run)
    try:
        costo = pd.read_csv(os.path.join(tmp, "calidad_corridas.csv")).set_index("run_id").loc[a.run, "costo_vs_base"]
        episodios = pd.read_csv(os.path.join(tmp, "episodios_propio.csv"))
        crudo = pd.read_csv(os.path.join(tmp, "dataset_etiquetado_propio.csv"),
                            usecols=["run_id", "servicio", "t_rel", "afectado"])
        ds = D.cargar(tmp, "_propio", "y_30s")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    episodios = episodios[episodios["run_id"] == a.run]
    crudo = crudo[crudo["run_id"] == a.run]
    afectado = crudo.assign(sec=np.floor(crudo["t_rel"]).astype(int)).groupby(["servicio", "sec"])["afectado"].max()
    if len(ds.eps):
        ds.eps = ds.eps[ds.eps["run"] == a.run]

    prob, umbral = V.predecir_todo(ds)
    prob[(ds.run == a.run) & (ds.sec < demo.CALENTAR)] = np.nan   # la demo no predice mientras calienta
    cubierto = np.isfinite(prob)
    alarma = cubierto & (prob > umbral)
    propia, esperando, causa = CR.separar(ds, alarma)
    cpu = D.alarma_cpu(ds) & cubierto
    nombre = os.path.basename(os.path.dirname(V.MODELO)).replace("resultados_", "")
    tabla = armar_tabla(ds, a.run, episodios, propia, alarma, cpu, nombre)

    cal = CAL.cargar(V.MODELO)
    prob_cal = np.where(cubierto, CAL.aplicar(cal, np.nan_to_num(prob)), np.nan)
    umbral_cal = cal["umbral_calibrado"] if cal else umbral
    salida = os.path.join(carpeta, "evidencia_%s" % nombre)
    dibujar(ds, a.run, meta, prob_cal, umbral_cal, propia, esperando, causa, afectado, servicios, tabla, nombre,
            salida + ".png")
    tabla.to_csv(salida + ".csv", index=False)
    salida_cadena = os.path.join(carpeta, "cadena_%s.png" % nombre)
    dibujar_cadena(ds, a.run, meta, carpeta, propia, esperando, causa, tabla, nombre, salida_cadena)

    print("Corrida %s | costo %.2f x lo normal%s | modelo %s, umbral %.2f%s"
          % (a.run, costo, " (CONTAMINADA)" if costo > RC.FACTOR_CONTAMINADA else "", nombre, umbral,
             " (probabilidad calibrada %.2f)" % umbral_cal if cal else " (sin calibracion: ml/calibrar.py)"))
    k = ds.run == a.run
    n_prop = int((D.inicios_alarma(ds, propia) & k).sum())
    n_esp = int((D.inicios_alarma(ds, esperando) & k).sum())
    print("Alarmas: %d propias, %d pasadas a la causa (causa raiz)" % (n_prop, n_esp))
    aviso = aviso_entrenamiento(a.run, meta)
    if aviso:
        print("Aviso: " + aviso)
    print(tabla.drop(columns="run").to_string(index=False) if len(tabla) else "Sin cuellos de botella en esta corrida")
    print("\nGrafica guardada en %s.png" % salida)
    print("Propagacion por la cadena: %s" % salida_cadena)


if __name__ == "__main__":
    main()
