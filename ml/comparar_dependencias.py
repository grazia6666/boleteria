"""
Ablacion pedida en la retroalimentacion: el MISMO LSTM con y sin las entradas de dependencias, y por separado el
efecto de la regla de causa raiz (que no es una entrada: se aplica despues, sin reentrenar).
  entradas de dependencias = cpu_dep, cola_dep, lat_p95_dep, d5_cpu_dep, d15_cpu_dep (--contexto en entrenar.py)
  causa raiz               = ml/causa_raiz.py, ventana 5 s
Entrenamiento: bash ml/pipeline_ablacion_dep.sh (mismas corridas, semillas 1-3 y folds; ensamble de 3).
Tablas:
  1. 2 x 2: (sin / con entradas de dependencias) x (sin / con causa raiz), umbral de cada ensamble, + CPU > 80 %
  2. solo las entradas, a igual tasa de falsas alarmas (3 y 2 por hora): media y desviacion de las 3 semillas
  3. por escenario de carga (normal / progresivo / burst)
  4. anticipacion real: avisos antes del primer pico de latencia (como ml/anticipacion_latencia.py)
Uso (desde la raiz):
  python ml/comparar_dependencias.py                  175 corridas de data/ (v12 contra v12_sindep)
  python ml/comparar_dependencias.py --data data_thr  replica en las 88 corridas de data_thr/
Guarda <data>/resultados_<sindep>/comparacion_dependencias*.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anticipacion_latencia as AL  # noqa: E402
import causa_raiz as CR  # noqa: E402
import datos as D  # noqa: E402
import postproceso as P  # noqa: E402
from por_escenario import ESCENARIO  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARES = {"data": ("resultados_v12_sindep", "resultados_v12"),
         "data_thr": ("resultados_thr_sindep", "resultados_thr_base")}


def prob_de(ds, ruta):
    p = pd.read_csv(ruta, usecols=["run", "servicio", "t_rel", "prob_lstm"])
    return p.set_index(["run", "servicio", "t_rel"])["prob_lstm"].reindex(
        pd.MultiIndex.from_arrays([ds.run, ds.srv, ds.sec])).values


def umbral_para(ds, pr, cub, mB, meta):
    """Umbral unico con `meta` falsas alarmas por hora (biseccion: las falsas alarmas bajan con el umbral)."""
    lo, hi = 0.0, 1.0
    for _ in range(20):
        u = (lo + hi) / 2
        if D.falsas_alarmas_hora(ds, cub & (pr > u), mB) > meta:
            lo = u
        else:
            hi = u
    return hi


def anticipacion(ds, al, sobre, nombre):
    det = AL.por_episodio(ds, al, sobre, nombre)
    av = det[det.avisado]
    g = det[(det.causa == "inyeccion_gradual") & det.avisado]
    return {"antes_del_pico_%": 100 * (~av.latencia_ya_cruzo).mean() if len(av) else np.nan,
            "grad_antes_del_pico_%": 100 * (~g.latencia_ya_cruzo).mean() if len(g) else np.nan,
            "grad_vs_pico_s": g.lead_vs_latencia_s.median() if len(g) else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", choices=sorted(PARES))
    a = ap.parse_args()
    data = os.path.join(HERE, a.data)
    sin, con = PARES[a.data]
    pd.set_option("display.width", 240)
    ds = D.cargar(data, "_propio", "y_30s")
    print("%s: %d corridas | %d episodios" % (a.data, len(set(ds.run)), len(ds.eps)))
    um = {s: v["umbral_ms"] for s, v in json.load(open(os.path.join(data, "umbrales_propio.json")))["servicios"].items()}
    sobre = np.expm1(ds.X[:, D.COLUMNAS.index("lat_p95_ms")]) > np.array([um[s] for s in ds.srv])

    al, cub = {}, None
    for n, r in (("sin dependencias", sin), ("con dependencias (v12)", con)):
        x, c = P.cargar_alarma(ds, os.path.join(data, r, "predicciones.csv"))
        al[n] = x
        cub = c if cub is None else cub & c
    mB = ds.valid & (ds.evalu == 1) & cub
    variantes = {}
    for n in ("sin dependencias", "con dependencias (v12)"):
        x = al[n] & cub
        variantes[n] = (x, x)
        variantes[n + " + causa raiz"] = (CR.separar(ds, x)[0], x)   # (alarma propia, todas las alarmas)
    cpu = D.alarma_cpu(ds) & cub
    variantes["CPU > 80 %"] = (cpu, cpu)

    # 1. tabla 2 x 2 (+ CPU)
    filas = []
    for n, (propia, todas) in variantes.items():
        f = P.fila(ds, n, propia, cub)
        f["cadena_servicio_o_dependencia_%"] = CR.evaluar(ds, n, propia, todas, mB)["propagacion_servicio_o_dependencia_%"]
        f.update(anticipacion(ds, propia, sobre, n))
        filas.append(f)
    t1 = pd.DataFrame(filas).set_index("variante")
    cols = ["precision", "recall", "f1", "falsas_alarmas_h", "det_inyeccion_gradual_%", "lead_inyeccion_gradual_s",
            "det_natural_%", "det_propagacion_%", "cadena_servicio_o_dependencia_%", "det_inyeccion_abrupto_%",
            "antes_del_pico_%", "grad_antes_del_pico_%", "grad_vs_pico_s"]
    print("\n1) 2 x 2: ENTRADAS DE DEPENDENCIAS x CAUSA RAIZ (umbral de cada ensamble, mismas filas)")
    print(t1[cols].round(3).to_string())

    # 2. solo las entradas, a igual tasa de falsas alarmas, por semilla
    filas = []
    for n, r in (("sin dependencias", sin), ("con dependencias (v12)", con)):
        for s in ("s1", "s2", "s3", "ensamble"):
            ruta = os.path.join(data, r, "predicciones.csv" if s == "ensamble" else os.path.join(s, "predicciones.csv"))
            pr = np.nan_to_num(prob_de(ds, ruta))
            for meta in (3.0, 2.0):
                u = umbral_para(ds, pr, cub, mB, meta)
                x = cub & (pr > u)
                m = D.prf(ds.y, x, mB)
                lt = D.lead_times(ds, x)
                g = lt[lt.causa == "inyeccion_gradual"]
                filas.append({"variante": n, "semilla": s, "meta_FA_h": meta, "umbral": u,
                              "FA_h": D.falsas_alarmas_hora(ds, x, mB), "precision": m["precision"],
                              "recall": m["recall"], "f1": m["f1"], "grad_avisadas_%": 100 * g.detectado.mean(),
                              "grad_lead_s": g.loc[g.detectado, "lead_s"].median()})
    t2 = pd.DataFrame(filas)
    sem = t2[t2.semilla != "ensamble"]
    resumen = sem.groupby(["meta_FA_h", "variante"])[["precision", "recall", "f1", "grad_avisadas_%", "grad_lead_s"]] \
        .agg(["mean", "std"]).round(3)
    print("\n2) SOLO LAS ENTRADAS, A IGUAL TASA DE FALSAS ALARMAS (3 semillas: media y desviacion)")
    print(resumen.to_string())
    print("\n   ensambles:")
    print(t2[t2.semilla == "ensamble"].drop(columns="semilla").round(3).to_string(index=False))

    # 3. por escenario
    esc = np.array([ESCENARIO.get(g, "otro") for g in ds.grupo])
    filas = []
    for e in ("normal", "progresivo", "burst"):
        m = mB & (esc == e)
        if not m.any():
            continue
        for n, (propia, _) in variantes.items():
            r = D.prf(ds.y, propia, m)
            filas.append({"escenario": e, "variante": n, "precision": r["precision"], "recall": r["recall"],
                          "f1": r["f1"], "falsas_h": D.falsas_alarmas_hora(ds, propia, m)})
    t3 = pd.DataFrame(filas)
    print("\n3) POR ESCENARIO DE CARGA")
    print(t3.pivot(index="variante", columns="escenario", values=["f1", "recall", "falsas_h"]).round(3).to_string())

    out = os.path.join(data, sin)
    t1.to_csv(os.path.join(out, "comparacion_dependencias.csv"))
    t2.to_csv(os.path.join(out, "comparacion_dependencias_igual_fa.csv"), index=False)
    t3.to_csv(os.path.join(out, "comparacion_dependencias_escenario.csv"), index=False)
    print("\nGuardado en %s (comparacion_dependencias*.csv)" % out)


if __name__ == "__main__":
    main()
