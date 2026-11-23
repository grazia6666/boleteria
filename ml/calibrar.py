"""
Calibracion del puntaje del modelo: convierte el puntaje del ensamble (promedio de 3 LSTM, que no es una
probabilidad real: con 0,90 acierta ~72 %) en una probabilidad que si se cumple ("70 %" = acierta ~70 %).

Usa las predicciones fuera de fold de la validacion cruzada (data/<resultados>/predicciones.csv), solo los
segundos evaluables (sin cuello de botella en ese instante, la vista B de las metricas), y ajusta una
regresion isotonica: una curva creciente, asi que NO cambia que alarmas suenan, solo el numero que se
muestra. La alarma sigue siendo puntaje > umbral del modelo (0,90 en v12); su equivalente calibrado se
guarda como "umbral_calibrado".

Uso (desde la raiz):
  python ml/calibrar.py                         # v12 -> data/resultados_v12/calibracion.json
  python ml/calibrar.py --resultados resultados_v12_sindep
Antes de guardar mide la calibracion de forma honesta: ajusta en 3/4 de las corridas y la mide en el 1/4
restante (4 veces). La usan ml/demo.py y ml/evidencia_vivo.py si el archivo existe.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORTES = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001]


def ajustar(x, y, bloques=300):
    """Regresion isotonica (pool adjacent violators) sobre `bloques` grupos por cuantiles del puntaje.
    Devuelve los puntos (xs, ys) de la curva; se aplica con np.interp."""
    q = np.unique(np.quantile(x, np.linspace(0, 1, bloques + 1)))
    b = np.clip(np.searchsorted(q, x, side="right") - 1, 0, len(q) - 2)
    w = np.bincount(b, minlength=len(q) - 1).astype(float)
    s = np.bincount(b, weights=y, minlength=len(q) - 1)
    xm = np.bincount(b, weights=x, minlength=len(q) - 1)
    ok = w > 0
    w, s, xm = w[ok], s[ok], xm[ok] / w[ok]
    pila = []   # [suma de aciertos, filas, x inicial, x final]
    for wi, si, xi in zip(w, s, xm):
        pila.append([si, wi, xi, xi])
        while len(pila) > 1 and pila[-2][0] / pila[-2][1] > pila[-1][0] / pila[-1][1]:
            ultimo = pila.pop()
            pila[-1][0] += ultimo[0]
            pila[-1][1] += ultimo[1]
            pila[-1][3] = ultimo[3]
    xs = [v for p in pila for v in (p[2], p[3])]
    ys = [p[0] / p[1] for p in pila for _ in (0, 1)]
    return [float(v) for v in xs], [float(v) for v in ys]


def aplicar(cal, p):
    """Probabilidad calibrada de un puntaje (o array de puntajes). Sin calibracion devuelve el puntaje."""
    if not cal:
        return p
    return np.interp(p, cal["x"], cal["y"])


def cargar(modelo_o_resultados):
    """Lee calibracion.json junto al modelo (ruta a modelo_ensamble.json o a la carpeta de resultados).
    None si no existe."""
    d = modelo_o_resultados if os.path.isdir(modelo_o_resultados) else os.path.dirname(modelo_o_resultados)
    ruta = os.path.join(d, "calibracion.json")
    return json.load(open(ruta)) if os.path.exists(ruta) else None


def tabla(p, y, nombre):
    b = pd.cut(p, CORTES, right=False)
    t = pd.DataFrame({"segundos": pd.Series(y).groupby(b).size(),
                      "acierta_%": 100 * pd.Series(y).groupby(b).mean()})
    t.index = ["%.1f-%.1f" % (i.left, min(i.right, 1)) for i in t.index]
    t.index.name = nombre
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resultados", default="resultados_v12")
    ap.add_argument("--umbral", type=float, default=None, help="por defecto el del modelo_ensamble.json")
    a = ap.parse_args()
    out = os.path.join(HERE, "data", a.resultados)
    pr = pd.read_csv(os.path.join(out, "predicciones.csv"),
                     usecols=["run", "prob_lstm", "y_objetivo", "evaluable"])
    pr = pr[pr["evaluable"] == 1]
    x, y = pr["prob_lstm"].values.astype(float), pr["y_objetivo"].values.astype(float)
    umbral = a.umbral
    if umbral is None:
        umbral = json.load(open(os.path.join(out, "modelo_ensamble.json")))["umbral"]

    # medida honesta: ajustar en 3/4 de las corridas, medir en el 1/4 restante
    runs = np.array(sorted(set(pr["run"])))
    fold_de = dict(zip(runs, np.random.RandomState(42).permutation(len(runs)) % 4))
    fold = pr["run"].map(fold_de).values
    pc = np.empty(len(x))
    for f in range(4):
        xs, ys = ajustar(x[fold != f], y[fold != f])
        pc[fold == f] = np.interp(x[fold == f], xs, ys)
    pd.set_option("display.width", 160)
    print("%d segundos evaluables de %d corridas (predicciones fuera de fold de %s)" % (len(x), len(runs), a.resultados))
    print("\nCuanto acierta cada rango (ajuste en 3/4 de las corridas, medido en el 1/4 restante):")
    print(pd.concat([tabla(x, y, "rango"), tabla(pc, y, "rango")], axis=1, keys=["puntaje", "calibrado"])
          .round(1).to_string())

    xs, ys = ajustar(x, y)
    cal = {"x": xs, "y": ys, "umbral_puntaje": float(umbral),
           "umbral_calibrado": float(np.interp(umbral, xs, ys)),
           "origen": "%s/predicciones.csv, %d segundos evaluables de %d corridas" % (a.resultados, len(x), len(runs))}
    json.dump(cal, open(os.path.join(out, "calibracion.json"), "w"), indent=1)
    print("\nPuntaje %.2f (umbral de la alarma) = probabilidad calibrada %.2f"
          % (umbral, cal["umbral_calibrado"]))
    print("Guardado: %s" % os.path.join(out, "calibracion.json"))


if __name__ == "__main__":
    main()
