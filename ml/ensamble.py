"""
Ensamble de semillas: promedia las probabilidades de varios entrenamientos de la MISMA configuracion
(distinta semilla, mismos folds) y evalua el promedio sin reentrenar.

Uso (despues de ml/barrido.py):
  python ml/ensamble.py --dirs barrido/base_s1 barrido/base_s2 barrido/base_s3
  python ml/ensamble.py --dirs resultados_v8/s1 resultados_v8/s2 resultados_v8/s3 --salida resultados_v8

Sin mirar el test: en cada fold, el umbral del ensamble es la media de los umbrales que cada semilla
eligio en su validacion (config.json -> umbrales_por_fold). Todas las carpetas deben venir del mismo
reparto en folds (--seed-folds), que es lo que hace barrido.py.
Imprime las metricas de cada semilla por separado y las del ensamble (vista B y lead time honesto).
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
from entrenar import asignar_folds  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--dirs", nargs="+", required=True, help="carpetas de data/ con predicciones.csv")
    ap.add_argument("--salida-csv", default="")
    ap.add_argument("--salida", default="",
                    help="carpeta de data/ donde guardar predicciones.csv y config.json del ensamble, con el "
                         "mismo formato que entrenar.py (sirve para demo --replay, por_escenario y postproceso)")
    a = ap.parse_args()

    confs = [json.load(open(os.path.join(a.data, d, "config.json"))) for d in a.dirs]
    p0 = confs[0]["parametros"]
    seed_folds = p0["seed_folds"] if p0.get("seed_folds") is not None else p0["seed"]
    for c in confs[1:]:
        q = c["parametros"]
        if (q["seed_folds"] if q.get("seed_folds") is not None else q["seed"]) != seed_folds:
            raise SystemExit("Las carpetas no comparten el reparto en folds (--seed-folds): no se pueden promediar")
    D.W = p0.get("ventana", 30)
    ds = D.cargar(a.data, p0.get("sufijo", ""), p0.get("objetivo", "y_30s"), confs[0].get("excluidas", []))
    clave = pd.MultiIndex.from_arrays([ds.run, ds.srv, ds.sec])

    probs, filas = [], []
    for d in a.dirs:
        pr = pd.read_csv(os.path.join(a.data, d, "predicciones.csv"))
        probs.append(pr.set_index(["run", "servicio", "t_rel"])["prob_lstm"].reindex(clave).values)
        al, cub = P.cargar_alarma(ds, os.path.join(a.data, d, "predicciones.csv"))
        filas.append(P.fila(ds, d, al & cub, cub))
    probs = np.vstack(probs)
    cubierto = np.isfinite(probs).all(0)
    media = np.nanmean(probs, 0)

    fold_de = asignar_folds(ds, p0["folds"], seed_folds)
    fold = np.array([fold_de[r] for r in ds.run])
    umbral_fold = np.mean([c["umbrales_por_fold"] for c in confs], 0)
    alarma = cubierto & (media > umbral_fold[fold])
    filas.append(P.fila(ds, "ENSAMBLE (%d semillas)" % len(a.dirs), alarma, cubierto))
    cpu = D.alarma_cpu(ds) & cubierto
    filas.append(P.fila(ds, "CPU > 80 %", cpu, cubierto))

    t = pd.DataFrame(filas).set_index("variante")
    pd.set_option("display.width", 220)
    print("Umbral del ensamble por fold:", np.round(umbral_fold, 3))
    print(t[["precision", "recall", "f1", "fbeta", "falsas_alarmas_h", "det_inyeccion_gradual_%",
             "lead_inyeccion_gradual_s", "lead>=15s_%", "lead>=15s_gradual_%"]].round(3).to_string())
    if a.salida_csv:
        t.to_csv(os.path.join(a.data, a.salida_csv))
    if a.salida:
        out = os.path.join(a.data, a.salida)
        os.makedirs(out, exist_ok=True)
        pd.DataFrame({"run": ds.run, "servicio": ds.srv, "t_rel": ds.sec, "prob_lstm": media, "alarma_lstm": alarma,
                      "cpu": ds.cpu, "bn_ahora": ds.bn, "y_30s": ds.y_30s, "y_objetivo": ds.y,
                      "evaluable": ds.evalu})[cubierto].to_csv(os.path.join(out, "predicciones.csv"), index=False)
        json.dump({"umbrales_por_fold": [float(u) for u in umbral_fold], "parametros": p0,
                   "excluidas": confs[0].get("excluidas", []), "ensamble_de": a.dirs},
                  open(os.path.join(out, "config.json"), "w"), indent=2)
        print("Guardado: %s (predicciones.csv y config.json del ensamble)" % out)
        modelos = ["%s/models/lstm_final.pt" % d.replace("\\", "/").rstrip("/") for d in a.dirs]
        if all(os.path.exists(os.path.join(a.data, m)) for m in modelos):
            # modelo final del ensamble para la demo: promedio de los modelos finales de cada semilla, con la
            # mediana de los umbrales por fold del ensamble (lo mismo que hace entrenar.py con un modelo solo)
            json.dump({"modelos": modelos, "umbral": float(np.median(umbral_fold)),
                       "nota": "rutas relativas a data/; la demo promedia las probabilidades de los modelos"},
                      open(os.path.join(out, "modelo_ensamble.json"), "w"), indent=2)
            print("Guardado: %s (para ml/demo.py --modelo)" % os.path.join(out, "modelo_ensamble.json"))


if __name__ == "__main__":
    main()
