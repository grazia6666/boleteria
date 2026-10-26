"""
Etiquetado del dataset consolidado (lo genera run_campaign.py).

Uso (desde la raiz del proyecto):
  python ml/etiquetar.py
  python ml/etiquetar.py --piso-ms 50 --sufijo _piso50      # analisis de sensibilidad

Entrada : data/dataset_metrics.csv
Salidas : data/dataset_etiquetado{sufijo}.csv   una fila por servicio-muestra, con etiquetas y split
          data/episodios{sufijo}.csv            una fila por episodio de cuello de botella
          data/umbrales{sufijo}.json            umbral de latencia por servicio y parametros usados

DEFINICION (identica para todas las corridas y servicios)
  1. Umbral por servicio = max(umbral_mult x mediana de lat_p95 en corridas normal_base, piso_ms).
  2. Un servicio esta en "condicion de cuello de botella" en una muestra si
        lat_p95 > umbral   O   no respondio a /metrics (scrape_ok = 0)
     (solo dentro de la ventana de carga y pasados los primeros --ignorar-inicio segundos).
  3. Un EPISODIO exige que la condicion dure >= --sostenido SEGUNDOS seguidos (medidos con las
     marcas de tiempo, no contando filas: el muestreo no es uniforme cuando un servicio no
     responde). Dos episodios separados por menos de --fusion segundos se funden en uno.
  4. bn_ahora = 1 si la muestra cae dentro de un episodio.

ETIQUETAS DE PREDICCION (horizonte H = --horizonte segundos)
  y_30s          1 si hay cuello de botella en algun instante de (t, t+H]   (lo que pide el PDF)
  y_inicio_30s   1 si un episodio EMPIEZA en (t, t+H] y ahora NO hay cuello de botella
  evaluable      1 si la fila es utilizable y bn_ahora = 0 (la anticipacion se evalua SOLO aqui)
  utilizable     1 si esta en carga, tras el arranque y con horizonte completo dentro de la corrida
  dt             segundos que representa la fila (hasta la siguiente muestra)

Los porcentajes del resumen estan PONDERADOS POR TIEMPO (dt), no por numero de filas.
Split por CORRIDA (nunca por filas): estratificado por grupo, semilla fija.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRUPOS = ("normal_base", "normal_inj", "burst_inj", "progresivo_base", "var_inj", "suave_inj",
          "rnd_normal", "rnd_burst", "rnd_progresivo", "rnd_rampa", "rnd_lenta",
          "rnd_combo_prog", "rnd_combo_burst", "rnd_simultanea")


def grupo_de(run_id):
    for g in GRUPOS:
        if run_id.startswith(g):
            return g
    return "otro"


def segmentos(b):
    """Tramos consecutivos de True como (inicio, fin) inclusivos."""
    seg, i, n = [], 0, len(b)
    while i < n:
        if b[i]:
            j = i
            while j + 1 < n and b[j + 1]:
                j += 1
            seg.append((i, j))
            i = j + 1
        else:
            i += 1
    return seg


def asignar_splits(runs_por_grupo, seed):
    rng = np.random.RandomState(seed)
    split = {}
    for g, runs in runs_por_grupo.items():
        runs = sorted(runs)
        rng.shuffle(runs)
        n = len(runs)
        if n <= 2:
            n_test, n_val = (1 if n == 2 else 0), 0
        else:
            n_test, n_val = max(1, round(0.2 * n)), max(1, round(0.15 * n))
        for i, r in enumerate(runs):
            split[r] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return split


def pct(x, col):
    """Porcentaje del TIEMPO (ponderado por dt) en que col vale 1."""
    s = x["dt"].sum()
    return 100.0 * (x[col] * x["dt"]).sum() / s if s > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--umbral-mult", type=float, default=3.0)
    ap.add_argument("--piso-ms", type=float, default=150.0)
    ap.add_argument("--sostenido", type=float, default=3.0)
    ap.add_argument("--fusion", type=float, default=5.0)
    ap.add_argument("--horizonte", type=float, default=30.0)
    ap.add_argument("--ignorar-inicio", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sufijo", default="")
    ap.add_argument("--solo-propio", action="store_true",
                    help="en los servicios que llaman a otros, exigir tambien tiempo PROPIO alto (ver abajo)")
    ap.add_argument("--piso-propio-ms", type=float, default=50.0)
    a = ap.parse_args()
    H = a.horizonte

    d = pd.read_csv(os.path.join(a.data, "dataset_metrics.csv"))
    d = d.sort_values(["run_id", "servicio", "timestamp"]).reset_index(drop=True)
    d["grupo"] = d["run_id"].map(grupo_de)

    # ---------- tiempo propio = latencia media - suma de las llamadas a dependencias ----------
    # Las llamadas son secuenciales: si el servicio solo esta esperando a otro, su tiempo propio no
    # sube (~8 ms) aunque su latencia total si. Con --solo-propio esos tramos quedan como "afectado"
    # y no como cuello de botella del servicio: el cuello de botella es el de la dependencia.
    e = pd.read_csv(os.path.join(a.data, "dataset_edges.csv"), usecols=["run_id", "origen", "timestamp", "lat_mean_ms"])
    down = e.groupby(["run_id", "origen", "timestamp"])["lat_mean_ms"].sum().rename("lat_dep_ms").reset_index()
    d = d.merge(down.rename(columns={"origen": "servicio"}), on=["run_id", "servicio", "timestamp"], how="left")
    llama = d["servicio"].isin(set(e["origen"]))
    d.loc[~llama, "lat_dep_ms"] = 0.0
    d["lat_propia_ms"] = (d["lat_mean_ms"] - d["lat_dep_ms"]).clip(lower=0)

    # ---------- umbrales desde las corridas normales ----------
    b = d[d["run_id"].str.startswith("normal_base") & (d["en_carga"] == 1) & (d["t_rel"] >= a.ignorar_inicio)
          & (d["scrape_ok"] == 1) & (d["rps"] > 0)]
    if b.empty:
        raise SystemExit("No hay corridas normal_base para calcular la linea base.")
    mediana = b.groupby("servicio")["lat_p95_ms"].median()
    umbral = (mediana * a.umbral_mult).clip(lower=a.piso_ms)
    maximo_normal = b.groupby("servicio")["lat_p95_ms"].max()
    json.dump({"parametros": vars(a),
               "servicios": {s: {"mediana_base_ms": round(float(mediana[s]), 1),
                                 "max_normal_ms": round(float(maximo_normal[s]), 1),
                                 "umbral_ms": round(float(umbral[s]), 1)} for s in mediana.index}},
              open(os.path.join(a.data, f"umbrales{a.sufijo}.json"), "w"), indent=2)
    d["umbral_ms"] = d["servicio"].map(umbral)
    umbral_propio = (b.groupby("servicio")["lat_propia_ms"].median() * a.umbral_mult).clip(lower=a.piso_propio_ms)
    if a.solo_propio:
        print("\nUMBRAL DE TIEMPO PROPIO (ms, solo servicios que llaman a otros):")
        print(umbral_propio[sorted(set(e["origen"]))].round(1).to_string())

    dur = d[d["en_carga"] == 1].groupby("run_id")["t_rel"].max()

    for c, v in (("bn_ahora", 0), ("afectado", 0), ("ep_id", -1), ("y_30s", 0), ("y_inicio_30s", 0),
                 ("utilizable", 0), ("evaluable", 0)):
        d[c] = v
    d["ep_causa"] = ""
    d["t_hasta_inicio"] = np.nan
    d["dt"] = 1.0

    episodios = []
    for (run, srv), g in d.groupby(["run_id", "servicio"], sort=False):
        idx = g.index.values
        ts = g["timestamp"].values
        trel = g["t_rel"].values
        n = len(g)
        dt = np.clip(np.diff(ts, append=ts[-1] + 1.0), 0.0, 3.0)   # segundos que representa cada fila
        activa = (g["en_carga"].values == 1) & (trel >= a.ignorar_inicio)
        lat = g["lat_p95_ms"].values.astype(float)
        sin_resp = g["scrape_ok"].values == 0
        lento = ~np.isnan(lat) & (lat > umbral[srv])
        cond_total = activa & (lento | sin_resp)
        if a.solo_propio and srv in set(e["origen"]):
            prop = g["lat_propia_ms"].values.astype(float)
            heredado = ~np.isnan(prop) & (prop <= umbral_propio[srv])   # lento, pero esperando a otro
            cond = activa & ((lento & ~heredado) | sin_resp)
        else:
            cond = cond_total

        # sostenido y fusion medidos en SEGUNDOS
        segs = [s for s in segmentos(cond) if ts[s[1]] + dt[s[1]] - ts[s[0]] >= a.sostenido]
        merged = []
        for s in segs:
            if merged and ts[s[0]] - (ts[merged[-1][1]] + dt[merged[-1][1]]) < a.fusion:
                merged[-1] = (merged[-1][0], s[1])
            else:
                merged.append(s)

        bn = np.zeros(n, dtype=bool)
        ep_id = np.full(n, -1)
        ep_causa = np.array([""] * n, dtype=object)
        inj_s, inj_a = g["inj_servicio"].values, g["inj_alguna"].values
        inj_t = g["inj_tipo"].fillna("").values
        onsets = []
        for k, (i0, i1) in enumerate(merged, 1):
            bn[i0:i1 + 1] = True
            ep_id[i0:i1 + 1] = k
            if inj_s[i0:i1 + 1].any():
                tipos = pd.Series(inj_t[i0:i1 + 1][inj_s[i0:i1 + 1] == 1]).mode()
                causa = "inyeccion_" + (tipos.iloc[0] if len(tipos) else "desconocida")
            elif inj_a[i0:i1 + 1].any():
                causa = "propagacion"
            else:
                causa = "natural"
            ep_causa[i0:i1 + 1] = causa
            previo = (ts[i0] - (ts[merged[k - 2][1]] + dt[merged[k - 2][1]])) if k > 1 \
                else (trel[i0] - a.ignorar_inicio)
            onsets.append(ts[i0])
            episodios.append({"run_id": run, "servicio": srv, "ep_id": k, "t_inicio": ts[i0],
                              "t_inicio_rel": round(float(trel[i0]), 1),
                              "t_fin_rel": round(float(trel[i1] + dt[i1]), 1),
                              "duracion_s": round(float(ts[i1] + dt[i1] - ts[i0]), 1),
                              "causa": causa, "previo_libre_s": round(float(previo), 1),
                              "grupo": g["grupo"].iloc[0]})

        onsets = np.array(onsets)
        t_hasta = np.full(n, np.nan)
        if len(onsets):
            j = np.searchsorted(onsets, ts, side="right")
            ok = j < len(onsets)
            t_hasta[ok] = onsets[j[ok]] - ts[ok]
        cum = np.concatenate([[0], np.cumsum(bn)])
        idx_r = np.searchsorted(ts, ts + H, side="right")
        futuro = cum[idx_r] - cum[np.arange(n) + 1]
        utilizable = activa & (trel + H <= dur[run])

        d.loc[idx, "dt"] = dt
        d.loc[idx, "bn_ahora"] = bn.astype(int)
        d.loc[idx, "afectado"] = (cond_total & ~cond).astype(int)   # lento por esperar a una dependencia
        d.loc[idx, "ep_id"] = ep_id
        d.loc[idx, "ep_causa"] = ep_causa
        d.loc[idx, "t_hasta_inicio"] = t_hasta
        d.loc[idx, "y_30s"] = (futuro > 0).astype(int)
        d.loc[idx, "y_inicio_30s"] = ((t_hasta <= H) & ~bn).astype(int)
        d.loc[idx, "utilizable"] = utilizable.astype(int)
        d.loc[idx, "evaluable"] = (utilizable & ~bn).astype(int)

    # ---------- split por corrida ----------
    runs_por_grupo = d.groupby("grupo")["run_id"].unique().to_dict()
    split = asignar_splits({g: list(r) for g, r in runs_por_grupo.items()}, a.seed)
    d["split"] = d["run_id"].map(split)

    ep = pd.DataFrame(episodios)
    if len(ep):
        ep["split"] = ep["run_id"].map(split)
    d.to_csv(os.path.join(a.data, f"dataset_etiquetado{a.sufijo}.csv"), index=False)
    ep.to_csv(os.path.join(a.data, f"episodios{a.sufijo}.csv"), index=False)

    # ---------- resumen ----------
    print("\nUMBRALES (ms) = max(%.0f x mediana base, %.0f)" % (a.umbral_mult, a.piso_ms))
    t = pd.DataFrame({"mediana_base": mediana, "max_normal": maximo_normal, "umbral": umbral}).round(1)
    t["umbral > max_normal"] = np.where(t["umbral"] > t["max_normal"], "si", "NO")
    print(t.to_string())

    dtv = d[d["en_carga"] == 1]["dt"]
    print("\nCADENCIA DE MUESTREO (segundos entre filas, en carga): mediana %.2f | p1 %.2f | p99 %.2f"
          % (dtv.median(), dtv.quantile(0.01), dtv.quantile(0.99)))
    print("  filas con dt < 0.5 s: %.1f%%   filas con dt > 1.3 s: %.1f%%"
          % (100 * (dtv < 0.5).mean(), 100 * (dtv > 1.3).mean()))

    u = d[d["utilizable"] == 1]
    print("\nMUESTRAS UTILIZABLES: %d de %d (%.0f s de datos)" % (len(u), len(d), u["dt"].sum()))
    print("\nPor grupo (porcentaje del TIEMPO utilizable):")
    r = u.groupby("grupo").apply(lambda x: pd.Series({
        "segundos": x["dt"].sum(), "bn_ahora": pct(x, "bn_ahora"), "y_30s": pct(x, "y_30s")}))
    ev = u[u["evaluable"] == 1].groupby("grupo").apply(lambda x: pct(x, "y_inicio_30s")).rename("y_inicio_30s*")
    print(r.join(ev).round(1).to_string())
    print("  * sobre el tiempo evaluable (sin cuello de botella en ese instante)")

    inj = d[d["inj_servicio"] == 1].groupby("run_id")["servicio"].first()
    if len(inj):
        filas = []
        for run, srv in inj.items():
            x = u[u["run_id"] == run]
            filas.append({"run_id": run, "servicio_inyectado": srv,
                          "bn_inyectado_%": round(pct(x[x["servicio"] == srv], "bn_ahora"), 1),
                          "bn_reservaciones_%": round(pct(x[x["servicio"] == "reservaciones"], "bn_ahora"), 1)})
        print("\nPor corrida con inyeccion (porcentaje del TIEMPO en cuello de botella):")
        print(pd.DataFrame(filas).to_string(index=False))

    print("\nEPISODIOS: %d" % len(ep))
    if len(ep):
        print("\nPor causa:")
        print(ep["causa"].value_counts().to_string())
        print("\nPor servicio y causa:")
        print(pd.crosstab(ep["servicio"], ep["causa"]).to_string())
        print("\nPor split:")
        print(pd.crosstab(ep["split"], ep["causa"]).to_string())
        print("\nCorridas por split:")
        print(pd.Series(split).value_counts().to_string())
        sin = ep[ep["previo_libre_s"] < 15]
        print("\nEpisodios con menos de 15 s libres antes de empezar: %d de %d "
              "(su Lead Time esta acotado por ese margen)" % (len(sin), len(ep)))
        if (ep["split"] == "test").sum() < 15:
            print("AVISO: el split de prueba tiene pocos episodios (%d)." % (ep["split"] == "test").sum())
    print("\nArchivos en %s: dataset_etiquetado%s.csv, episodios%s.csv, umbrales%s.json"
          % (a.data, a.sufijo, a.sufijo, a.sufijo))


if __name__ == "__main__":
    main()