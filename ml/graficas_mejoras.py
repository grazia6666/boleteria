"""
Figuras de las dos mejoras de v12 sin reentrenar (validacion cruzada, 175 corridas; README 13.7):
  causa_raiz_v12.png  - CPU > 80 % / v12 / v12 + causa raiz en 4 medidas
  calibracion_v12.png - lo que marca la grafica contra lo que pasa (antes y despues de calibrar)

Uso (desde la raiz, despues de python ml/causa_raiz.py y python ml/calibrar.py):
  python ml/graficas_mejoras.py
Las guarda en data/resultados_v12/ y copia en docs/img/ (las que muestra el README).
"""
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import BoxStyle, FancyBboxPatch, Patch  # noqa: E402

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ + "/ml")
import calibrar as CAL  # noqa: E402

BASE = os.path.join(RAIZ, "data", "resultados_v12") + os.sep
NARANJA, AZUL, CIRUELA, GRIS = "#C8590A", "#1F5FBF", "#9C3D8F", "#8A93A3"
FONDO, TINTA, TENUE, GUIA = "#FBFAF6", "#14213D", "#4A5563", "#E4E1D8"
coma = lambda v, n=2: ("%.*f" % (n, v)).replace(".", ",")  # noqa: E731
plt.rcParams.update({"font.size": 14, "font.family": "DejaVu Sans", "text.color": TINTA, "axes.labelcolor": TINTA,
                     "xtick.color": TINTA, "ytick.color": TENUE, "axes.edgecolor": "#B9BEC6"})


def barra(fig, ax, x, ancho, alto, color):
    """Barra de punta redondeada apoyada en la base."""
    caja = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    px, py = (x1 - x0) / caja.width, (y1 - y0) / caja.height
    r = 0.05 * px
    ax.add_patch(FancyBboxPatch((x, -r * py / px), ancho, alto + r * py / px,
                                boxstyle=BoxStyle("Round", pad=0, rounding_size=r),
                                mutation_aspect=py / px, fc=color, lw=0, zorder=3))


# ------------------------------------------------------------------ 1) causa raiz
t = pd.read_csv(BASE + "causa_raiz.csv").set_index("variante")
filas = [("CPU > 80 %", t.loc["CPU > 80 %"], NARANJA), ("v12", t.loc["modelo"], AZUL),
         ("v12 + causa raíz", t.loc["modelo + causa raiz (5 s)"], CIRUELA)]
paneles = [("falsas_alarmas_h", "Falsas alarmas\npor hora y servicio", "más bajo es mejor", 4.0, lambda v: coma(v, 1)),
           ("precision", "Precisión", "más alto es mejor", 1.0, lambda v: coma(v)),
           ("det_inyeccion_gradual_%", "Fallas graduales\navisadas", "más alto es mejor", 100, lambda v: "%d %%" % round(v)),
           ("propagacion_servicio_o_dependencia_%", "Cadenas avisadas\n(servicio o su causa)",
            "más alto es mejor", 100, lambda v: "%d %%" % round(v))]
fig = plt.figure(figsize=(12.5, 6.2), facecolor=FONDO)
gs = fig.add_gridspec(1, 4, left=0.04, right=0.99, top=0.66, bottom=0.2, wspace=0.32)
for k, (col, titulo, guia, ymax, fmt) in enumerate(paneles):
    ax = fig.add_subplot(gs[k])
    ax.set_facecolor(FONDO)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, ymax * 1.12)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xticks([])
    for i, (nombre, f, color) in enumerate(filas):
        barra(fig, ax, i - 0.38, 0.76, f[col], color)
        ax.text(i, f[col] + ymax * 0.025, fmt(f[col]), ha="center", va="bottom", zorder=4,
                fontweight="bold" if i == 2 else "normal", fontsize=13.5)
    ax.set_title(titulo, loc="left", fontsize=13.5, fontweight="bold", pad=24)
    ax.text(0, 1.02, guia, transform=ax.transAxes, fontsize=11.5, color=TENUE, va="bottom")
fig.text(0.04, 0.92, "Causa raíz: 23 % menos falsas alarmas y los mismos avisos", fontsize=20, fontweight="bold",
         va="bottom")
fig.text(0.04, 0.865, "Si un servicio y otro al que llama están en alarma, el aviso queda solo en el causante. "
         "Validación cruzada de v12: 175 corridas.", fontsize=12.5, color=TENUE, va="bottom")
fig.legend(handles=[Patch(color=c, label=n) for n, _, c in filas], loc="lower center", bbox_to_anchor=(0.5, 0.075),
           ncol=3, frameon=False, fontsize=13.5, handlelength=1.2, columnspacing=2.2)
fig.text(0.04, 0.02, "El F1 por servicio baja de %s a %s: la alarma del servicio que solo espera ya no cuenta como "
         "acierto; el aviso de esa cadena lo da el causante." % (coma(filas[1][1]["f1"]), coma(filas[2][1]["f1"])),
         fontsize=11.5, color=TENUE, va="bottom")
fig.savefig(BASE + "causa_raiz_v12.png", dpi=200, facecolor=FONDO)
plt.close(fig)
print(BASE + "causa_raiz_v12.png")

# ------------------------------------------------------------------ 2) calibracion (medida honesta: 3/4 -> 1/4)
pr = pd.read_csv(BASE + "predicciones.csv", usecols=["run", "prob_lstm", "y_objetivo", "evaluable"])
pr = pr[pr["evaluable"] == 1]
x, y = pr["prob_lstm"].values.astype(float), pr["y_objetivo"].values.astype(float)
runs = np.array(sorted(set(pr["run"])))
fold = pr["run"].map(dict(zip(runs, np.random.RandomState(42).permutation(len(runs)) % 4))).values
pc = np.empty(len(x))
for f in range(4):
    xs, ys = CAL.ajustar(x[fold != f], y[fold != f])
    pc[fold == f] = np.interp(x[fold == f], xs, ys)
cortes = np.linspace(0, 1, 11)


def curva(p):
    b = np.clip(np.digitize(p, cortes) - 1, 0, 9)
    d = pd.DataFrame({"b": b, "p": p, "y": y}).groupby("b").agg(n=("y", "size"), p=("p", "mean"), y=("y", "mean"))
    return d[d.n >= 300]


antes, ahora = curva(x), curva(pc)
print(antes.round(3), ahora.round(3), sep="\n")
cal = CAL.cargar(BASE)
fig, ax = plt.subplots(figsize=(10.4, 6.6), facecolor=FONDO)
fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.13)
ax.set_facecolor(FONDO)
ax.plot([0, 1], [0, 1], color=TENUE, ls=(0, (4, 4)), lw=1.4, zorder=2)
ax.text(0.62, 0.66, "perfecta: marca 0,7 → pasa 70 %", rotation=0, fontsize=11.5, color=TENUE, ha="right")
for d, color, nombre in ((antes, GRIS, "puntaje del modelo (antes)"), (ahora, AZUL, "probabilidad calibrada (ahora)")):
    ax.plot(d.p, d.y, color=color, lw=2.4, zorder=3)
    ax.scatter(d.p, d.y, s=70, color=color, edgecolor=FONDO, linewidth=2, zorder=4)
ax.axvline(cal["umbral_calibrado"], color=AZUL, lw=1.2, alpha=0.5, zorder=1)
ax.text(cal["umbral_calibrado"] + 0.012, 0.03, "alarma: %s calibrado\n= puntaje %s" % (coma(cal["umbral_calibrado"]),
        coma(cal["umbral_puntaje"])), fontsize=11, color=AZUL, va="bottom")
r7 = antes[(antes.p > 0.7) & (antes.p < 0.8)].iloc[0]
ax.annotate("marca %s, pasa solo %d %%" % (coma(r7.p), round(100 * r7.y)), (r7.p, r7.y), xytext=(10, -26),
            textcoords="offset points", fontsize=11.5, color=TINTA)
uc = ahora.iloc[-1]
ax.annotate("marca %s, pasa %d %%" % (coma(uc.p), round(100 * uc.y)), (uc.p, uc.y), xytext=(-14, 10),
            textcoords="offset points", ha="right", fontsize=11.5, color=TINTA, fontweight="bold")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks(np.linspace(0, 1, 6))
ax.set_yticks(np.linspace(0, 1, 6))
ax.set_xticklabels([coma(v, 1) for v in np.linspace(0, 1, 6)])
ax.set_yticklabels(["%d %%" % round(100 * v) for v in np.linspace(0, 1, 6)])
ax.grid(color=GUIA, lw=1, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(length=0)
ax.set_xlabel("lo que marca la gráfica")
ax.set_ylabel("veces que llegó el cuello de botella")
fig.text(0.1, 0.92, "Calibrada, la línea de la gráfica es una probabilidad real", fontsize=19, fontweight="bold",
         va="bottom")
fig.text(0.1, 0.865, "175 corridas de la validación cruzada. Se ajusta en 3/4 de las corridas y se mide en el 1/4 "
         "restante.", fontsize=12, color=TENUE, va="bottom")
ax.legend(handles=[Line2D([], [], color=GRIS, lw=2.4, marker="o", ms=8, label="puntaje del modelo (antes)"),
                   Line2D([], [], color=AZUL, lw=2.4, marker="o", ms=8, label="probabilidad calibrada (ahora)")],
          loc="upper left", frameon=False, fontsize=12.5)
fig.savefig(BASE + "calibracion_v12.png", dpi=200, facecolor=FONDO)
print(BASE + "calibracion_v12.png")
os.makedirs(os.path.join(RAIZ, "docs", "img"), exist_ok=True)
for f in ("causa_raiz_v12.png", "calibracion_v12.png"):
    shutil.copy(BASE + f, os.path.join(RAIZ, "docs", "img", f))
print("copiadas en docs/img/")
