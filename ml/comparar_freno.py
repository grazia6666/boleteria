"""
Compara v15 (configuracion de v12 + freno de CPU) con la misma configuracion sin freno, entrenadas sobre las mismas
corridas de data_thr/ y con los mismos folds (bash ml/pipeline_thr.sh). Mide:
  1. validacion cruzada: precision, recall, F1, falsas alarmas/h, avisos y anticipacion por tipo de episodio
  2. a igual tasa de falsas alarmas (umbral unico movido hasta igualarlas)
  3. anticipacion real: avisos antes del primer pico de latencia (como ml/anticipacion_latencia.py)
  4. por escenario de carga
Uso (desde la raiz): python ml/comparar_freno.py      Guarda data_thr/comparacion_freno.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anticipacion_latencia as AL  # noqa: E402
import datos as D  # noqa: E402
import postproceso as P  # noqa: E402
from por_escenario import ESCENARIO  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data_thr")
VARIANTES = [("sin freno (config. v12)", "resultados_thr_base"), ("v15: con freno", "resultados_thr_freno")]


def prob_de(ds, ruta):
    p = pd.read_csv(ruta, usecols=["run", "servicio", "t_rel", "prob_lstm"])
    return p.set_index(["run", "servicio", "t_rel"])["prob_lstm"].reindex(
        pd.MultiIndex.from_arrays([ds.run, ds.srv, ds.sec])).values


def main():
    pd.set_option("display.width", 220)
    ds = D.cargar(DATA, "_propio", "y_30s")
    print("corridas: %d | episodios: %d" % (len(set(ds.run)), len(ds.eps)))
    um = {s: v["umbral_ms"] for s, v in json.load(open(os.path.join(DATA, "umbrales_propio.json")))["servicios"].items()}
    sobre = np.expm1(ds.X[:, D.COLUMNAS.index("lat_p95_ms")]) > np.array([um[s] for s in ds.srv])
    al, pr, cub = {}, {}, None
    for n, r in VARIANTES:
        a, c = P.cargar_alarma(ds, os.path.join(DATA, r, "predicciones.csv"))
        al[n] = a
        pr[n] = prob_de(ds, os.path.join(DATA, r, "predicciones.csv"))
        cub = c if cub is None else cub & c
    al = {n: a & cub for n, a in al.items()}
    al["CPU > 80 %"] = D.alarma_cpu(ds) & cub
    mB = ds.valid & (ds.evalu == 1) & cub

    filas = []
    for n, a in al.items():
        f = P.fila(ds, n, a, cub)
        det = AL.por_episodio(ds, a, sobre, n)
        gv = det[(det.causa == "inyeccion_gradual") & det.avisado]
        av = det[det.avisado]
        f["antes_del_pico_%"] = 100 * (~av.latencia_ya_cruzo).mean() if len(av) else np.nan
        f["grad_antes_del_pico_%"] = 100 * (~gv.latencia_ya_cruzo).mean() if len(gv) else np.nan
        f["grad_vs_pico_s"] = gv.lead_vs_latencia_s.median() if len(gv) else np.nan
        filas.append(f)
    t = pd.DataFrame(filas).set_index("variante")
    print("\n1) VALIDACION CRUZADA (vista B, mismas filas)")
    print(t[["precision", "recall", "f1", "falsas_alarmas_h", "det_inyeccion_gradual_%", "lead_inyeccion_gradual_s",
             "det_natural_%", "lead_natural_s", "det_propagacion_%", "det_inyeccion_abrupto_%", "antes_del_pico_%",
             "grad_antes_del_pico_%", "grad_vs_pico_s"]].round(3).to_string())

    print("\n2) A IGUAL TASA DE FALSAS ALARMAS (umbral unico)")
    for n, _ in VARIANTES:
        ts = np.arange(0.3, 0.9995, 0.0025)
        fas = np.array([D.falsas_alarmas_hora(ds, cub & (np.nan_to_num(pr[n]) > u), mB) for u in ts])
        for meta in (3.0, 2.0):
            u = ts[np.argmin(np.abs(fas - meta))]
            a = cub & (np.nan_to_num(pr[n]) > u)
            r = D.prf(ds.y, a, mB)
            lt = D.lead_times(ds, a)
            g = lt[lt.causa == "inyeccion_gradual"]
            print("  %-26s umbral %.4f falsas/h %.2f | P %.3f R %.3f F1 %.3f | graduales %.0f %% (mediana %s s)"
                  % (n, u, D.falsas_alarmas_hora(ds, a, mB), r["precision"], r["recall"], r["f1"],
                     100 * g.detectado.mean(), g.loc[g.detectado, "lead_s"].median()))

    print("\n3) POR ESCENARIO (F1 / falsas alarmas por hora)")
    esc = np.array([ESCENARIO.get(g, "otro") for g in ds.grupo])
    filas = []
    for e in ("normal", "progresivo"):
        m = mB & (esc == e)
        for n, a in al.items():
            r = D.prf(ds.y, a, m)
            filas.append({"escenario": e, "metodo": n, "F1": r["f1"], "falsas_h": D.falsas_alarmas_hora(ds, a, m)})
    print(pd.DataFrame(filas).pivot(index="metodo", columns="escenario", values=["F1", "falsas_h"]).round(3).to_string())
    t.to_csv(os.path.join(DATA, "comparacion_freno.csv"))
    print("\nGuardado: %s" % os.path.join(DATA, "comparacion_freno.csv"))


if __name__ == "__main__":
    main()
