"""
Barrido de hiperparametros y de balanceo alrededor de la configuracion de v7.

Cada variante cambia UNA cosa respecto a v7 y se entrena con varias semillas, siempre con el mismo
reparto en folds (--seed-folds 42), para separar el efecto real del azar de la inicializacion.

Uso (desde la raiz, con la maquina libre: no correr durante una campana):
  python ml/barrido.py                      # todas las variantes, semillas 1 2 3
  python ml/barrido.py --solo base,parada_ap --semillas 1 2
  python ml/barrido.py --solo-resumen       # no entrena, solo resume lo que ya hay

Salida: data/barrido/<variante>_s<semilla>/ (resultados de entrenar.py) y data/barrido/resumen.csv
(media y desviacion por variante: F1, precision, recall, falsas alarmas/h y anticipacion).
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = ["--sufijo", "_propio", "--contexto", "--tendencias", "--seed-folds", "42", "--sin-final"]
VARIANTES = {
    "base": [],                                               # v7
    "sin_balanceo": ["--balanceo", "ninguno"],
    "submuestreo_1a1": ["--balanceo", "submuestreo", "--ratio", "1"],
    "submuestreo_1a3": ["--balanceo", "submuestreo", "--ratio", "3"],
    "parada_loss_pond": ["--parada", "loss_pond"],
    "parada_ap": ["--parada", "ap"],
    "hidden_32": ["--hidden", "32"],
    "hidden_128": ["--hidden", "128"],
    "2capas_dropout": ["--capas", "2", "--dropout", "0.2"],
    "lr_bajo": ["--lr", "3e-4", "--epocas", "30", "--paciencia", "5"],
    "weight_decay": ["--wd", "1e-4"],
    "ventana_60": ["--ventana", "60"],
    "batch_64": ["--batch", "64"],
}


def metricas(d):
    b = pd.read_csv(os.path.join(d, "metricas_B.csv"), index_col=0).loc["LSTM"]
    lt = pd.read_csv(os.path.join(d, "lead_time_LSTM.csv"))
    gr = lt[lt["causa"] == "inyeccion_gradual"]
    pf = pd.read_csv(os.path.join(d, "por_fold.csv"))
    return {"f1": b["f1"], "precision": b["precision"], "recall": b["recall"], "fbeta": b["fbeta"],
            "falsas_h": b["falsas_alarmas_h"], "f1_folds_std": pf[pf["metodo"] == "LSTM"]["f1"].std(),
            "graduales_det_%": 100 * gr["detectado"].mean(), "lead_gradual_s": gr["lead_s"].median(),
            ">=15s_%": 100 * (lt["lead_s"] >= 15).mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", default="", help="variantes separadas por coma")
    ap.add_argument("--semillas", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--solo-resumen", action="store_true")
    a = ap.parse_args()
    nombres = a.solo.split(",") if a.solo else list(VARIANTES)
    raiz = os.path.join(HERE, "data", "barrido")
    os.makedirs(raiz, exist_ok=True)

    if not a.solo_resumen:
        for n in nombres:
            for s in a.semillas:
                sal = os.path.join("barrido", "%s_s%d" % (n, s))
                if os.path.exists(os.path.join(HERE, "data", sal, "metricas_B.csv")):
                    print("ya existe %s, se salta" % sal)
                    continue
                print("== %s semilla %d ==" % (n, s), flush=True)
                cmd = [sys.executable, os.path.join(HERE, "ml", "entrenar.py")] + BASE + VARIANTES[n] + \
                    ["--seed", str(s), "--salida", sal]
                with open(os.path.join(raiz, "%s_s%d.log" % (n, s)), "w") as log:
                    rc = subprocess.run(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT).returncode
                if rc != 0:
                    print("!! fallo %s semilla %d (ver el .log)" % (n, s), flush=True)

    filas = []
    for n in VARIANTES:
        for s in a.semillas:
            d = os.path.join(raiz, "%s_s%d" % (n, s))
            if os.path.exists(os.path.join(d, "metricas_B.csv")):
                filas.append({"variante": n, "semilla": s, **metricas(d)})
    if not filas:
        return
    t = pd.DataFrame(filas)
    t.to_csv(os.path.join(raiz, "detalle.csv"), index=False)
    cols = ["f1", "precision", "recall", "falsas_h", "graduales_det_%", "lead_gradual_s", ">=15s_%"]
    r = t.groupby("variante")[cols].agg(["mean", "std"])
    r.columns = ["%s_%s" % c for c in r.columns]
    r["n"] = t.groupby("variante").size()
    base = r.loc["base", "f1_mean"] if "base" in r.index else np.nan
    r["delta_f1_vs_base"] = r["f1_mean"] - base
    r = r.sort_values("f1_mean", ascending=False).round(3)
    r.to_csv(os.path.join(raiz, "resumen.csv"))
    pd.set_option("display.width", 250)
    print(r[["n", "f1_mean", "f1_std", "delta_f1_vs_base", "precision_mean", "recall_mean", "falsas_h_mean",
             "graduales_det_%_mean", "lead_gradual_s_mean", ">=15s_%_mean"]].to_string())


if __name__ == "__main__":
    main()
