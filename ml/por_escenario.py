"""
Metricas por ESCENARIO DE CARGA (normal / progresivo / burst), los tres del enunciado.

Usa las predicciones de la validacion cruzada que guarda entrenar.py (data/<resultados>/predicciones.csv):
cada corrida la predijo un modelo que no la vio al entrenar. Compara contra la regla CPU > 80 %.

Uso (desde la raiz, despues de entrenar.py):
  python ml/por_escenario.py                                   # modelo final v7
  python ml/por_escenario.py --resultados resultados_v7 --sufijo _propio

Que es cada columna (todas sobre las corridas de ese escenario):
  f1, precision, recall  por segundo, en la vista B (solo segundos sin cuello de botella en ese momento).
                         Alarma de v7 = probabilidad > umbral de su fold; alarma CPU = cpu > 80 %.
  falsas_h               veces que la alarma se ENCIENDE sin cuello de botella en los 30 s siguientes,
                         por hora de servicio observada.
  avisados_%             % de episodios con una alarma que se enciende en los 30 s previos al inicio,
                         tras >= 5 s apagada (lead time honesto, datos.lead_times).
  anticipacion_s         mediana de esos segundos de anticipacion, entre los episodios avisados.
  >=15s_%                % de episodios avisados con 15 s o mas de anticipacion.
Escenario de cada grupo: normal_base, normal_inj, var_inj, suave_inj, rnd_normal, rnd_lenta -> normal;
burst_inj, rnd_burst, rnd_rampa -> burst; progresivo_base, rnd_progresivo -> progresivo.
Salida: data/<resultados>/por_escenario_carga.csv
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datos as D  # noqa: E402
import postproceso as P  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ESCENARIO = {"normal_base": "normal", "normal_inj": "normal", "var_inj": "normal", "suave_inj": "normal",
             "burst_inj": "burst", "progresivo_base": "progresivo",
             "rnd_normal": "normal", "rnd_burst": "burst", "rnd_progresivo": "progresivo",
             "rnd_rampa": "burst", "rnd_lenta": "normal",
             "rnd_combo_prog": "progresivo", "rnd_combo_burst": "burst", "rnd_simultanea": "normal"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--resultados", default="resultados_v7")
    a = ap.parse_args()

    out = os.path.join(a.data, a.resultados)
    conf = json.load(open(os.path.join(out, "config.json")))
    par = conf["parametros"]
    D.W = par.get("ventana", 30)
    ds = D.cargar(a.data, par.get("sufijo", ""), par.get("objetivo", "y_30s"), conf.get("excluidas", []))
    lstm, cub = P.cargar_alarma(ds, os.path.join(out, "predicciones.csv"))
    lstm &= cub
    cpu = D.alarma_cpu(ds) & cub
    esc = np.array([ESCENARIO.get(g, "otro") for g in ds.grupo])
    mB = ds.valid & (ds.evalu == 1) & cub
    leads = {"LSTM": D.lead_times(ds, lstm), "CPU > 80 %": D.lead_times(ds, cpu)}

    filas = []
    for e in ("normal", "progresivo", "burst"):
        m = mB & (esc == e)
        runs = set(ds.run[(esc == e) & cub])
        for nombre, al in (("LSTM", lstm), ("CPU > 80 %", cpu)):
            r = D.prf(ds.y, al, m)
            lt = leads[nombre]
            lt = lt[lt["run"].isin(runs)]
            filas.append({"escenario": e, "metodo": nombre, "corridas": len(runs), "episodios": len(lt),
                          "f1": r["f1"], "precision": r["precision"], "recall": r["recall"],
                          "falsas_h": D.falsas_alarmas_hora(ds, al, m),
                          "avisados_%": 100 * lt["detectado"].mean() if len(lt) else np.nan,
                          "anticipacion_s": lt["lead_s"].median() if len(lt) else np.nan,
                          ">=15s_%": 100 * (lt["lead_s"] >= 15).mean() if len(lt) else np.nan})
    t = pd.DataFrame(filas).round(2)
    pd.set_option("display.width", 200)
    print(t.to_string(index=False))
    t.to_csv(os.path.join(out, "por_escenario_carga.csv"), index=False)
    print("\nGuardado: %s" % os.path.join(out, "por_escenario_carga.csv"))


if __name__ == "__main__":
    main()
