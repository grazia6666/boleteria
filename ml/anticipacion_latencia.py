"""
Anticipacion medida contra la LATENCIA, ademas de contra la etiqueta.

El Lead Time del proyecto mide el aviso contra el inicio del cuello de botella PROPIO (etiquetar.py). En los
episodios de propagacion y en los naturales, el servicio suele estar lento antes de ese inicio, esperando a otros,
asi que ese Lead Time puede contar como "anticipacion" un aviso que llego cuando la lentitud ya se veia. Este
script revisa cada aviso (Lead Time honesto, el mismo de datos.lead_times) y mira si la latencia p95 del servicio
ya habia pasado su umbral, aunque fuera un segundo, en la ventana de 30 s previa al inicio y antes del aviso.

Uso (desde la raiz, sobre las predicciones de la validacion cruzada):
  python ml/anticipacion_latencia.py
  python ml/anticipacion_latencia.py --resultados resultados_v11
Guarda data/<resultados>/anticipacion_latencia.csv (resumen) y anticipacion_latencia_episodios.csv (detalle).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causa_raiz as CR  # noqa: E402
import datos as D  # noqa: E402
import postproceso as P  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SILENCIO = 5


def por_episodio(ds, alarma, sobre, nombre):
    """Una fila por episodio con ventana suficiente: aviso honesto y primer cruce de la p95 antes del inicio."""
    filas = []
    for e in ds.eps.itertuples():
        o = e.gi
        s0 = ds.sstart[o] + D.W - 1
        lo = max(o - D.H, s0)
        js = np.arange(lo, o)
        js = js[(ds.bn[js] == 0) & (ds.util[js] == 1)]
        if len(js) < 3:
            continue
        hit = js[alarma[js]]
        limpio = [j for j in hit if j - SILENCIO >= s0 and not alarma[j - SILENCIO:j].any()]
        ventana = np.arange(lo, o)
        cruces = ventana[sobre[ventana]]
        primero = int(cruces[0]) if len(cruces) else None
        j = limpio[0] if limpio else None
        filas.append({"metodo": nombre, "run": e.run, "servicio": e.srv, "causa": e.causa,
                      "inicio_s": int(ds.sec[o]), "avisado": j is not None,
                      "aviso_s": int(ds.sec[j]) if j is not None else None,
                      "lead_etiqueta_s": o - j if j is not None else None,
                      "primer_cruce_s": int(ds.sec[primero]) if primero is not None else None,
                      "latencia_ya_cruzo": bool(j is not None and primero is not None and primero <= j),
                      "lead_vs_latencia_s": (primero - j) if (j is not None and primero is not None) else
                      (o - j if j is not None else None)})
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resultados", default="resultados_v12")
    a = ap.parse_args()
    out = os.path.join(HERE, "data", a.resultados)
    cfg = json.load(open(os.path.join(out, "config.json")))["parametros"]
    ds = D.cargar(os.path.join(HERE, "data"), cfg.get("sufijo", ""), cfg.get("objetivo", "y_30s"))
    lstm, cub = P.cargar_alarma(ds, os.path.join(out, "predicciones.csv"))
    lstm &= cub
    propia, _, _ = CR.separar(ds, lstm)
    um = {s: v["umbral_ms"] for s, v in json.load(open(
        os.path.join(HERE, "data", "umbrales%s.json" % cfg.get("sufijo", ""))))["servicios"].items()}
    p95 = np.expm1(ds.X[:, D.COLUMNAS.index("lat_p95_ms")])
    sobre = p95 > np.array([um[s] for s in ds.srv])

    det = pd.concat([por_episodio(ds, al, sobre, n) for n, al in
                     (("modelo", lstm), ("modelo + causa raiz", propia), ("CPU > 80 %", D.alarma_cpu(ds) & cub))])
    av = det[det["avisado"]]
    res = av.groupby(["metodo", "causa"]).agg(
        avisos=("run", "size"), lead_etiqueta_mediana_s=("lead_etiqueta_s", "median"),
        antes_del_primer_pico_pct=("latencia_ya_cruzo", lambda x: 100 * (~x).mean()),
        lead_vs_primer_pico_mediana_s=("lead_vs_latencia_s", "median"))
    tot = av.groupby("metodo").agg(avisos=("run", "size"),
                                   antes_del_primer_pico=("latencia_ya_cruzo", lambda x: int((~x).sum())))
    tot["antes_del_primer_pico_pct"] = 100 * tot["antes_del_primer_pico"] / tot["avisos"]
    pd.set_option("display.width", 200)
    print("Avisos (Lead Time honesto) y si llegaron antes del primer pico de latencia, por causa:")
    print(res.round(1).to_string())
    print("\nTotal:")
    print(tot.round(1).to_string())
    print("\n(lead_vs_primer_pico > 0: el aviso llego antes de que la p95 pasara su umbral; < 0: despues)")
    res.to_csv(os.path.join(out, "anticipacion_latencia.csv"))
    det.to_csv(os.path.join(out, "anticipacion_latencia_episodios.csv"), index=False)
    print("\nGuardado: %s y anticipacion_latencia_episodios.csv" % os.path.join(out, "anticipacion_latencia.csv"))


if __name__ == "__main__":
    main()
