"""
Baseline por threshold (no se entrena) y referencia de persistencia.

Uso (desde la raiz del proyecto, despues de etiquetar.py):
  python ml/baseline.py
  python ml/baseline.py --cpu 80 --solo-test

  threshold   : alarma si la CPU del servicio > --cpu % (80 por defecto, lo que pide el PDF)
  persistencia: alarma si YA hay cuello de botella. Referencia: un modelo que no la supere
                no aporta anticipacion.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datos as D  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--sufijo", default="")
    ap.add_argument("--cpu", type=float, default=80.0)
    ap.add_argument("--solo-test", action="store_true", help="usar solo las corridas del split 'test'")
    a = ap.parse_args()

    ds = D.cargar(a.data, a.sufijo)
    mask = None
    if a.solo_test:
        import pandas as pd
        e = pd.read_csv(os.path.join(a.data, "dataset_etiquetado%s.csv" % a.sufijo), usecols=["run_id", "split"])
        test_runs = set(e[e["split"] == "test"]["run_id"])
        mask = np.array([r in test_runs for r in ds.run])
    alarmas = {"threshold CPU>%g%%" % a.cpu: D.alarma_cpu(ds, a.cpu),
               "persistencia": D.alarma_persistencia(ds)}
    rep = D.reporte(ds, alarmas, mask)
    D.imprimir_reporte(ds, rep, "BASELINE" + (" (solo split test)" if a.solo_test else " (todas las corridas)"))

    print("\nPor escenario (vista B, threshold CPU):")
    for g in sorted(set(ds.grupo)):
        m = D.prf(ds.y, alarmas["threshold CPU>%g%%" % a.cpu], ds.valid & (ds.evalu == 1) & (ds.grupo == g))
        print("  %-16s n=%6d  precision=%.3f  recall=%.3f  f1=%.3f" % (g, m["n"], m["precision"], m["recall"], m["f1"]))


if __name__ == "__main__":
    main()