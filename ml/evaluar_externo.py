"""
Evalua un modelo sobre un SUBCONJUNTO de corridas (por defecto las de calendario aleatorio, rnd_*), con
las mismas metricas que el resto del proyecto, para comparar modelos en las mismas corridas.

Dos fuentes de predicciones:
  --modelo RUTA        un modelo guardado (.pt) o un ensamble (.json): predice las corridas desde cero.
                       Valido como prueba honesta solo si el modelo NO se entreno con esas corridas
                       (p. ej. v8, entrenado antes de que existieran las corridas rnd_*).
  --resultados DIR     las predicciones de la validacion cruzada de un entrenamiento (predicciones.csv):
                       cada corrida la predijo un modelo que no la vio (p. ej. v9, que si incluye rnd_*).

Uso (desde la raiz, despues de etiquetar.py):
  python ml/evaluar_externo.py --modelo data/resultados_v8/modelo_ensamble.json --nombre v8
  python ml/evaluar_externo.py --resultados resultados_v9 --nombre v9

Imprime y guarda en data/externo_<nombre>_<prefijo>.csv:
  - vista B global: precision, recall, F1, falsas alarmas por hora (por servicio y del sistema) y precision
    por alarma (cada encendido), junto a la regla CPU > 80 %;
  - F1 y falsas alarmas por escenario de carga;
  - por causa del episodio: % avisado antes de empezar, anticipacion mediana y % con >= 15 s; y lo mismo
    contando el aviso sostenido (alarma encendida hasta 120 s antes y sin cortarse, *_sost). Con el
    calendario aleatorio, los fallos abruptos ya no se pueden "adivinar": su % avisado deberia quedar cerca
    del nivel del azar (falsas alarmas/h x 30 / 3600).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "ml"))
sys.path.insert(0, os.path.join(HERE, "collector"))
import datos as D  # noqa: E402
import postproceso as P  # noqa: E402
from por_escenario import ESCENARIO  # noqa: E402


def predecir_modelo(ds, ruta):
    import demo
    predecir, ck = demo.cargar_modelo(ruta)
    idx = np.where(ds.valid)[0]
    prob = np.full(ds.N, np.nan)
    for i in range(0, len(idx), 2048):
        b = idx[i:i + 2048]
        prob[b] = predecir(np.stack([ds.X[j - D.W + 1:j + 1] for j in b]))
    return np.isfinite(prob) & (prob > ck["umbral"]), np.isfinite(prob)


def precision_por_alarma(ds, al, m):
    prev = np.zeros(ds.N, dtype=bool)
    prev[1:] = al[:-1]
    prev[ds.pos == 0] = False
    enc = al & ~prev & m
    return (enc & (ds.y == 1)).sum() / enc.sum() if enc.sum() else float("nan")


def resumen(ds, nombre, al, cub, sel):
    m = ds.valid & (ds.evalu == 1) & cub & sel
    r = D.prf(ds.y, al, m)
    fa = D.falsas_alarmas_hora(ds, al, m)
    fila = {"metodo": nombre, "precision": r["precision"], "recall": r["recall"], "f1": r["f1"],
            "falsas_h_servicio": fa, "falsas_h_sistema": 6 * fa, "precision_por_alarma": precision_por_alarma(ds, al, m),
            "azar_%": 100 * fa * 30 / 3600}
    esc = np.array([ESCENARIO.get(g, "otro") for g in ds.grupo])
    for e in ("normal", "progresivo", "burst"):
        fila["f1_" + e] = D.prf(ds.y, al, m & (esc == e))["f1"]
        fila["falsas_h_" + e] = D.falsas_alarmas_hora(ds, al, m & (esc == e))
    lt = D.lead_times(ds, al)
    lt = lt[lt["run"].isin(set(ds.run[cub & sel]))]
    causas = []
    for c, g in lt.groupby("causa"):
        causas.append({"metodo": nombre, "causa": c, "episodios": len(g), "avisados_%": 100 * g["detectado"].mean(),
                       "anticipacion_s": g["lead_s"].median(), ">=15s_%": 100 * (g["lead_s"] >= 15).mean(),
                       "avisados_sost_%": 100 * g["detectado_ext"].mean(),
                       "anticip_sost_s": g["lead_ext_s"].median(), ">=30s_sost_%": 100 * (g["lead_ext_s"] >= 30).mean()})
    return fila, causas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--sufijo", default="_propio")
    ap.add_argument("--prefijo", default="rnd_", help="corridas a evaluar (prefijo del nombre)")
    ap.add_argument("--modelo", default="")
    ap.add_argument("--resultados", default="")
    ap.add_argument("--nombre", default="modelo")
    a = ap.parse_args()

    ds = D.cargar(a.data, a.sufijo, "y_30s")
    sel = np.array([r.startswith(a.prefijo) for r in ds.run])
    if a.modelo:
        al, cub = predecir_modelo(ds, a.modelo)
    else:
        al, cub = P.cargar_alarma(ds, os.path.join(a.data, a.resultados, "predicciones.csv"))
    al &= cub & sel
    cpu = D.alarma_cpu(ds) & cub & sel
    filas, causas = [], []
    for nombre, x in ((a.nombre, al), ("CPU > 80 %", cpu)):
        f, c = resumen(ds, nombre, x, cub, sel)
        filas.append(f)
        causas += c
    t, c = pd.DataFrame(filas), pd.DataFrame(causas)
    pd.set_option("display.width", 230)
    print("Corridas evaluadas: %d (prefijo '%s') | episodios: %d"
          % (len(set(ds.run[sel & cub])), a.prefijo, int(ds.eps["run"].str.startswith(a.prefijo).sum())))
    print("\nVISTA B GLOBAL:")
    print(t[["metodo", "precision", "recall", "f1", "falsas_h_servicio", "falsas_h_sistema",
             "precision_por_alarma", "azar_%"]].round(3).to_string(index=False))
    print("\nPOR ESCENARIO (F1 y falsas alarmas/h por servicio):")
    print(t[["metodo", "f1_normal", "falsas_h_normal", "f1_progresivo", "falsas_h_progresivo", "f1_burst",
             "falsas_h_burst"]].round(3).to_string(index=False))
    print("\nPOR CAUSA DEL EPISODIO:")
    print(c.sort_values(["causa", "metodo"]).round(1).to_string(index=False))
    out = os.path.join(a.data, "externo_%s_%s" % (a.nombre, a.prefijo.rstrip("_")))
    t.to_csv(out + ".csv", index=False)
    c.to_csv(out + "_causas.csv", index=False)
    print("\nGuardado: %s.csv y %s_causas.csv" % (out, out))


if __name__ == "__main__":
    main()
