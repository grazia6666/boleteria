"""
Causa raiz: cuando un servicio y alguno de los servicios a los que llama estan en alarma a la vez, el
aviso se deja solo en el que causa el problema. La alarma del que llama (por ejemplo reservaciones,
que llama a pago) se marca "esperando a <causa>" y no cuenta como cuello de botella propio.

Regla (no se reentrena nada; se aplica sobre las alarmas del modelo, segundo a segundo y solo mirando
el pasado, asi que sirve en vivo):
  la alarma de s en t pasa a "esperando" si alguna dependencia de s (datos.DEPENDE) tuvo alarma en
  [t - VENTANA, t]. La causa es la dependencia en alarma; si esa a su vez espera a otra, se sigue la
  cadena hasta el final (reservaciones -> pago -> ticket: la causa es ticket).

Por que: la mayoria de las falsas alarmas de v12 eran servicios que llaman a otro que se estaba
cayendo. Con la regla bajan las falsas alarmas y no se pierde ningun aviso de cadena: el aviso pasa a
la dependencia. Se mide como "aviso en el servicio o en su dependencia" (30 s antes del inicio).
VENTANA = 5 s: con 0 s quita pocas falsas alarmas, con 10-20 s empieza a apagar avisos de episodios
naturales (carga alta en varios servicios a la vez).

Uso (desde la raiz; evalua sobre las predicciones de la validacion cruzada):
  python ml/causa_raiz.py
  python ml/causa_raiz.py --resultados resultados_v12_sindep --ventana 10
La usan ml/demo.py y ml/evidencia_vivo.py.
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
VENTANA = 5


def raiz(s, en_alarma, _camino=()):
    """Servicio causante para s, dado en_alarma = {servicio: alarma en los ultimos VENTANA s}.
    None si ninguna dependencia de s esta en alarma."""
    for d in D.DEPENDE.get(s, []):
        if en_alarma.get(d) and d not in _camino:
            return raiz(d, en_alarma, _camino + (s,)) or d
    return None


def _recientes(ds, alarma, w):
    """DataFrame (run, sec) x servicio: hubo alarma en [sec - w, sec]."""
    df = pd.DataFrame({"run": ds.run, "srv": ds.srv, "sec": ds.sec, "a": alarma.astype(np.int8)})
    pv = df.pivot_table(index=["run", "sec"], columns="srv", values="a", aggfunc="max")
    partes = []
    for r, g in pv.groupby(level=0):
        secs = g.index.get_level_values(1)
        g = g.droplevel(0).reindex(range(secs.min(), secs.max() + 1)).fillna(0)
        if w > 0:
            g = g.rolling(w + 1, min_periods=1).max()
        partes.append(g.assign(run=r).set_index("run", append=True).swaplevel())
    return pd.concat(partes) > 0


def separar(ds, alarma, w=VENTANA):
    """Divide la alarma del modelo en (propia, esperando, causa): `propia` es la que cuenta como aviso de
    cuello de botella del servicio; `esperando` la que se pasa a la dependencia; `causa` el nombre del
    servicio causante en las filas de `esperando` ("" en el resto)."""
    rec = _recientes(ds, alarma, w)
    esperando = np.zeros(ds.N, dtype=bool)
    causa = np.full(ds.N, "", dtype=object)
    for s, deps in D.DEPENDE.items():
        deps = [d for d in deps if d in rec.columns]
        idx = np.where((ds.srv == s) & alarma)[0]
        if not deps or not len(idx):
            continue
        R = rec.reindex(pd.MultiIndex.from_arrays([ds.run[idx], ds.sec[idx]])).fillna(False)
        en = R[deps].any(axis=1).values
        esperando[idx[en]] = True
        for i, fila in zip(idx[en], R[en].to_dict("records")):
            causa[i] = raiz(s, fila) or ""
    return alarma & ~esperando, esperando, causa


def separar_tabla(alarmas, w=VENTANA):
    """Lo mismo que separar() para UNA corrida dada como DataFrame (indice = segundo consecutivo, columnas =
    servicios, valores bool). Devuelve (propia, causa): DataFrames del mismo tamano; causa = "" sin espera."""
    rec = alarmas.astype(int).rolling(w + 1, min_periods=1).max() > 0
    propia = alarmas.astype(bool).copy()
    causa = pd.DataFrame("", index=alarmas.index, columns=alarmas.columns)
    for s in alarmas.columns:
        deps = [d for d in D.DEPENDE.get(s, []) if d in rec.columns]
        if not deps:
            continue
        en = alarmas[s].astype(bool) & rec[deps].any(axis=1)
        for t in alarmas.index[en.values]:
            causa.at[t, s] = raiz(s, rec.loc[t].to_dict()) or ""
        propia[s] = alarmas[s].astype(bool) & ~en
    return propia, causa


def aviso_en_dependencia(ds, alarma, eps=None, h=None):
    """Por episodio (filas de eps, por defecto ds.eps): alguna dependencia del servicio tuvo alarma en los
    h s previos al inicio (h = datos.H, 30 s)."""
    eps = ds.eps if eps is None else eps
    h = h or D.H
    clave = pd.Series(np.arange(ds.N), index=pd.MultiIndex.from_arrays([ds.run, ds.srv, ds.sec]))
    out = {}
    for i, e in eps.iterrows():
        o = int(ds.sec[e["gi"]])
        filas = []
        for d in D.DEPENDE.get(e["srv"], []):
            filas += list(clave.reindex(pd.MultiIndex.from_product([[e["run"]], [d], range(o - h, o)]))
                          .dropna().astype(int).values)
        out[i] = bool(len(filas) and alarma[filas].any())
    return pd.Series(out, dtype=bool)


def evaluar(ds, nombre, propia, todas, mB):
    """Metricas de vista B con la alarma `propia`; las de cadena cuentan tambien `todas` en la dependencia."""
    m = D.prf(ds.y, propia, mB)
    r = {"variante": nombre, "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
         "falsas_alarmas_h": D.falsas_alarmas_hora(ds, propia, mB)}
    lt = D.lead_times(ds, propia)
    lt = lt[lt["run"].isin(set(ds.run[mB]))]
    for c in ("inyeccion_gradual", "natural", "propagacion"):
        r["det_%s_%%" % c] = 100 * lt.loc[lt["causa"] == c, "detectado_ext"].mean()
    eps = ds.eps[(ds.eps["causa"] == "propagacion") & ds.eps["run"].isin(set(ds.run[mB]))]
    propio = {}
    for i in eps.index:   # aviso sostenido en el propio servicio, episodio por episodio
        sub, ds.eps = ds.eps, eps.loc[[i]]
        t = D.lead_times(ds, propia)
        ds.eps = sub
        propio[i] = bool(len(t) and t["detectado_ext"].iloc[0])
    dep = aviso_en_dependencia(ds, todas, eps)
    r["propagacion_servicio_o_dependencia_%"] = 100 * np.mean([propio[i] or dep[i] for i in eps.index])
    return r


def main():
    import postproceso as P
    ap = argparse.ArgumentParser()
    ap.add_argument("--resultados", default="resultados_v12")
    ap.add_argument("--ventana", type=int, default=VENTANA)
    ap.add_argument("--salida-csv", default="causa_raiz.csv")
    a = ap.parse_args()

    out = os.path.join(HERE, "data", a.resultados)
    cfg = json.load(open(os.path.join(out, "config.json")))["parametros"]
    ds = D.cargar(os.path.join(HERE, "data"), cfg.get("sufijo", ""), cfg.get("objetivo", "y_30s"))
    lstm, cub = P.cargar_alarma(ds, os.path.join(out, "predicciones.csv"))
    lstm &= cub
    mB = ds.valid & (ds.evalu == 1) & cub
    propia, esperando, causa = separar(ds, lstm, a.ventana)
    cpu = D.alarma_cpu(ds) & cub
    suave = P.k_de_n(ds, lstm, 3, 5) & cub
    filas = [evaluar(ds, "modelo", lstm, lstm, mB),
             evaluar(ds, "modelo + causa raiz (%d s)" % a.ventana, propia, lstm, mB),
             evaluar(ds, "modelo + suavizado 3 de 5", suave, suave, mB),
             evaluar(ds, "CPU > 80 %", cpu, cpu, mB)]
    t = pd.DataFrame(filas).set_index("variante")
    pd.set_option("display.width", 220)
    print("Resultados: %s | %d corridas | vista B, aviso sostenido en los episodios" % (a.resultados, len(set(ds.run))))
    print(t.round(3).to_string())

    ini = D.inicios_alarma(ds, lstm) & mB & (ds.y == 0)
    ini_e = D.inicios_alarma(ds, esperando) & mB & (ds.y == 0)
    print("\nFalsas alarmas por servicio (encendidos) y cuantas pasan a la causa:")
    print(pd.DataFrame({"modelo": pd.Series(ds.srv[ini]).value_counts(),
                        "pasan a la causa": pd.Series(ds.srv[ini_e]).value_counts()}).fillna(0).astype(int).to_string())
    print("\nCausas senaladas (segundos de alarma pasada):")
    print(pd.Series(causa[esperando]).value_counts().to_string())
    t.to_csv(os.path.join(out, a.salida_csv))
    print("\nGuardado: %s" % os.path.join(out, a.salida_csv))


if __name__ == "__main__":
    main()
