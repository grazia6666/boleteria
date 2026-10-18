"""
Grafica una corrida: latencia p95, cola, CPU y requests/s por servicio,
con las inyecciones de cuello de botella sombreadas.

Uso (desde la raiz del proyecto):
  python ml/graficar_corrida.py                      # ultima corrida en data/
  python ml/graficar_corrida.py data/<run_id>        # una corrida concreta
  python ml/graficar_corrida.py data/<run_id> --smooth 3 --log --show
Genera data/<run_id>/grafica.png
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDEN = ["usuarios", "eventos", "reservaciones", "pago", "ticket", "notificaciones"]
COLORES = {"usuarios": "#1f77b4", "eventos": "#17becf", "reservaciones": "#d62728",
           "pago": "#ff7f0e", "ticket": "#2ca02c", "notificaciones": "#9467bd"}
PANELES = [("lat_p95_ms", "Latencia p95 (ms)"), ("cola", "Cola (requests en curso)"),
           ("cpu", "CPU (% del limite)"), ("rps", "Requests/s")]


def ultima_corrida():
    carpetas = [d for d in glob.glob(os.path.join(HERE, "data", "*"))
                if os.path.exists(os.path.join(d, "metrics.csv"))]
    if not carpetas:
        sys.exit("No hay corridas con metrics.csv en data/")
    return max(carpetas, key=os.path.getmtime)


def resumen_inyecciones(df, ev, duracion):
    """Compara el servicio inyectado y reservaciones dentro vs fuera de la inyeccion."""
    cols = ["lat_p95_ms", "cola", "cpu", "rps"]
    carga = (df["t"] >= 0) & (df["t"] <= duracion) if duracion else df["t"] >= 0
    en_alguna = pd.Series(False, index=df.index)
    for _, e in ev.iterrows():
        en_alguna |= (df["t"] >= e["ini"]) & (df["t"] <= e["fin"])
    print("\nResumen (promedios durante la carga, solo segundos con rps > 0)")
    for _, e in ev.iterrows():
        dentro = (df["t"] >= e["ini"]) & (df["t"] <= e["fin"])
        for srv in dict.fromkeys([e["servicio"], "reservaciones"]):
            s = df[(df["servicio"] == srv) & (df["rps"] > 0)]
            a = s[dentro.loc[s.index]][cols].mean()
            b = s[(carga & ~en_alguna).loc[s.index]][cols].mean()
            print(f"\n  {srv}  (inyeccion en {e['servicio']}, cpu_load={int(e['cpu_load'])}, "
                  f"{e['ini']:.0f}s-{e['fin']:.0f}s)")
            print(pd.DataFrame({"fuera": b, "dentro": a}).round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default="")
    ap.add_argument("--smooth", type=int, default=1, help="ventana de media movil en segundos")
    ap.add_argument("--log", action="store_true", help="escala log en latencia")
    ap.add_argument("--show", action="store_true", help="abrir la ventana ademas de guardar")
    args = ap.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = args.run_dir or ultima_corrida()
    run = run if os.path.isabs(run) else os.path.join(os.getcwd(), run)
    df = pd.read_csv(os.path.join(run, "metrics.csv"))
    meta = {}
    mp = os.path.join(run, "meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))

    t0 = meta.get("t0_locust", df["timestamp"].min())
    df["t"] = df["timestamp"] - t0
    duracion = meta.get("duration", 0)

    if args.smooth > 1:
        for c, _ in PANELES:
            df[c] = df.groupby("servicio")[c].transform(
                lambda s: s.rolling(args.smooth, min_periods=1).mean())

    ev = pd.DataFrame()
    ep = os.path.join(run, "events.csv")
    if os.path.exists(ep):
        ev = pd.read_csv(ep)
        ev["ini"] = ev["t_inicio"] - t0
        ev["fin"] = ev["t_fin"] - t0
    inyectados = set(ev["servicio"]) if len(ev) else set()

    fig, axes = plt.subplots(len(PANELES), 1, figsize=(13, 11), sharex=True)
    for ax, (col, titulo) in zip(axes, PANELES):
        for srv in ORDEN:
            d = df[df["servicio"] == srv]
            if d.empty:
                continue
            ax.plot(d["t"], d[col], label=srv, color=COLORES[srv],
                    linewidth=2.4 if srv in inyectados else 1.2,
                    alpha=1.0 if srv in inyectados else 0.75)
        for _, e in ev.iterrows():
            ax.axvspan(e["ini"], e["fin"], color="red", alpha=0.12)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        if duracion:
            ax.axvline(duracion, color="gray", linestyle="--", linewidth=1)
        ax.set_ylabel(titulo)
        ax.grid(alpha=0.3)
    if args.log:
        axes[0].set_yscale("log")

    for _, e in ev.iterrows():
        axes[0].text((e["ini"] + e["fin"]) / 2, 1.02, f"inyeccion: {e['servicio']} (cpu_load={int(e['cpu_load'])})",
                     transform=axes[0].get_xaxis_transform(), ha="center", fontsize=9, color="red")
    axes[-1].set_xlabel("segundos desde el inicio de la carga (lineas grises = inicio y fin de Locust)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.97))
    fig.suptitle(f"{meta.get('run_id', os.path.basename(run))}   |   escenario: {meta.get('scenario', '?')}",
                 y=0.995, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = os.path.join(run, "grafica.png")
    fig.savefig(out, dpi=130)
    print(f"Grafica guardada en {out}")
    fallos = int((df["scrape_ok"] == 0).sum())
    if fallos:
        print(f"Aviso: {fallos} filas con scrape_ok=0 (servicio sin responder a /metrics)")
    if len(ev):
        resumen_inyecciones(df, ev, duracion)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()