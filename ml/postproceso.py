"""
Post-proceso de las alarmas del LSTM, sin reentrenar: usa las predicciones de la validacion
cruzada que guarda entrenar.py (data/<--resultados>/predicciones.csv).

Uso (desde la raiz del proyecto, despues de entrenar.py):
  python ml/postproceso.py
  python ml/postproceso.py --k 3 --n 5 --resultados resultados_sin_lat

  suavizado k de n : la alarma suena en t solo si el LSTM (con el umbral de su fold) dijo "si"
                     en al menos k de los ultimos n segundos (t incluido). Es causal: solo mira
                     el pasado, asi que se puede usar en linea. Quita parpadeos sueltos a costa
                     de retrasar el encendido hasta k-1 s.
  hibrido          : alarma = LSTM suavizado  O  CPU > --cpu %. Recupera la saturacion de CPU
                     (progresivo_base), donde la regla simple le gana al LSTM.

k y n se fijan de antemano (3 de 5 por defecto). La rejilla que se imprime al final es un
analisis de SENSIBILIDAD sobre las mismas predicciones de test: no se usa para elegir k y n.

Salida: data/<--resultados>/postproceso.csv (una fila por variante, vista B + lead time).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datos as D  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAUSAS = ["inyeccion_gradual", "inyeccion_abrupto", "propagacion", "natural"]


def cargar_alarma(ds, ruta):
    """Alinea predicciones.csv con las filas de ds por (corrida, servicio, segundo)."""
    p = pd.read_csv(ruta, usecols=["run", "servicio", "t_rel", "alarma_lstm"])
    clave = pd.MultiIndex.from_arrays([ds.run, ds.srv, ds.sec])
    al = p.set_index(["run", "servicio", "t_rel"])["alarma_lstm"].reindex(clave)
    cubierto = al.notna().values
    return al.fillna(False).values.astype(bool), cubierto


def k_de_n(ds, alarma, k, n):
    """True en t si hay >= k alarmas en [t-n+1, t] dentro de la misma serie (corrida, servicio)."""
    if k <= 1 and n <= 1:
        return alarma.copy()
    c = np.concatenate([[0], np.cumsum(alarma.astype(int))])
    i = np.arange(ds.N)
    lo = np.maximum(i - n + 1, ds.sstart)
    return (c[i + 1] - c[lo]) >= k


def fila(ds, nombre, al, cubierto):
    mB = ds.valid & (ds.evalu == 1) & cubierto
    m = D.prf(ds.y, al, mB)
    r = {"variante": nombre, "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
         "fbeta": m["fbeta"], "falsas_alarmas_h": D.falsas_alarmas_hora(ds, al, mB)}
    lt = D.lead_times(ds, al)
    lt = lt[lt["run"].isin(set(ds.run[cubierto]))]
    for c in CAUSAS:
        s = lt[lt["causa"] == c]
        r["det_%s_%%" % c] = 100 * s["detectado"].mean() if len(s) else np.nan
        r["lead_%s_s" % c] = s["lead_s"].median() if len(s) else np.nan
    r["lead>=15s_%"] = 100 * (lt["lead_s"] >= 15).mean() if len(lt) else np.nan
    r["lead>=15s_gradual_%"] = 100 * (lt[lt["causa"] == "inyeccion_gradual"]["lead_s"] >= 15).mean()
    for g in ("progresivo_base", "suave_inj"):
        r["f1_%s" % g] = D.prf(ds.y, al, mB & (ds.grupo == g))["f1"]
    r["fp_normal_base"] = D.prf(ds.y, al, mB & (ds.grupo == "normal_base"))["fp"]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--sufijo", default="")
    ap.add_argument("--resultados", default="resultados")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--cpu", type=float, default=80.0)
    ap.add_argument("--excluir-contaminadas", action="store_true",
                    help="evaluar solo en las corridas que pasan el control de calidad")
    ap.add_argument("--salida-csv", default="postproceso.csv")
    a = ap.parse_args()

    out = os.path.join(a.data, a.resultados)
    conf = json.load(open(os.path.join(out, "config.json")))
    cfg = conf["parametros"]
    excluir = set(conf.get("excluidas", []))
    if a.excluir_contaminadas:   # evaluar solo en corridas limpias, aunque el modelo se entrenara con todas
        excluir |= set(D.corridas_contaminadas(a.data))
    ds = D.cargar(a.data, a.sufijo, cfg.get("objetivo", "y_30s"), sorted(excluir))
    print("Resultados: %s | objetivo: %s" % (a.resultados, ds.objetivo))
    lstm, cubierto = cargar_alarma(ds, os.path.join(out, "predicciones.csv"))
    lstm &= cubierto
    cpu = D.alarma_cpu(ds, a.cpu) & cubierto
    suave = k_de_n(ds, lstm, a.k, a.n) & cubierto
    kn = "%d de %d" % (a.k, a.n)

    filas = [fila(ds, "LSTM (original)", lstm, cubierto),
             fila(ds, "LSTM suavizado %s" % kn, suave, cubierto),
             fila(ds, "hibrido: LSTM | CPU>%g%%" % a.cpu, lstm | cpu, cubierto),
             fila(ds, "hibrido: LSTM %s | CPU>%g%%" % (kn, a.cpu), suave | cpu, cubierto),
             fila(ds, "threshold CPU>%g%%" % a.cpu, cpu, cubierto)]
    t = pd.DataFrame(filas).set_index("variante")

    pd.set_option("display.width", 200)
    print("\nVISTA B (solo segundos sin cuello de botella ahora):")
    print(t[["precision", "recall", "f1", "fbeta", "falsas_alarmas_h", "f1_progresivo_base",
             "f1_suave_inj", "fp_normal_base", "lead>=15s_%", "lead>=15s_gradual_%"]].round(3).to_string())
    print("\nLEAD TIME HONESTO: detectados (%) y mediana (s) por causa:")
    cols = [x for c in CAUSAS for x in ("det_%s_%%" % c, "lead_%s_s" % c)]
    print(t[cols].round(1).rename(columns=lambda s: s.replace("inyeccion_", "iny_")).to_string())

    print("\nSENSIBILIDAD al suavizado (no se usa para elegir k y n):")
    sens = []
    for k, n in ((1, 1), (2, 3), (3, 5), (4, 5), (5, 8), (8, 10)):
        al = k_de_n(ds, lstm, k, n) & cubierto
        r = fila(ds, "%d de %d" % (k, n), al, cubierto)
        sens.append(r)
    s = pd.DataFrame(sens).set_index("variante")
    print(s[["precision", "recall", "f1", "falsas_alarmas_h", "det_inyeccion_gradual_%",
             "lead_inyeccion_gradual_s"]].round(3).to_string())

    pd.concat([t, s.add_prefix("")], keys=["variantes", "sensibilidad"]).to_csv(
        os.path.join(out, a.salida_csv))
    print("\nGuardado: %s" % os.path.join(out, a.salida_csv))


if __name__ == "__main__":
    main()
