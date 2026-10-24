"""
Utilidades compartidas por baseline.py y entrenar.py: carga del dataset etiquetado, rejilla de 1 s,
ventanas y metricas (Precision, Recall, F1, Prediction Lead Time).  Compatible con Python 3.8.

Rejilla de 1 s: cada (corrida, servicio) se remuestrea a un valor por segundo (media de las filas de
ese segundo). Los segundos sin dato (servicio sin responder) arrastran el ultimo valor y quedan
marcados con la columna binaria `falta`.

Entradas del modelo en cada segundo (len(CONT) + 1 + 6 columnas; entrenar.py elige cuales usa):
  cpu/100, ram/256, log1p(rps), log1p(lat_media), log1p(lat_p95), log1p(cola), log1p(err_rps),
  contexto de dependencias (*_dep: maximo sobre los servicios a los que llama, directa o
  indirectamente; 0 en los servicios hoja), tendencias (d5_* / d15_*: cambio en 5 y 15 s),
  falta, one-hot del servicio y, al final, las tendencias largas (LARGO, entrenar.py --largo): cuanto
  crecieron el trafico y la CPU respecto a hace 60 y 120 s, y en cuantos segundos llegaria la CPU al
  100 % al ritmo del ultimo minuto. Van al final para no mover las columnas de los modelos v7-v9.
Una muestra = los ultimos W=30 segundos de UN servicio; etiqueta = `objetivo` en el ultimo segundo
(y_30s: hay cuello de botella en (t, t+H];  y_inicio_30s: EMPIEZA uno en (t, t+H] y ahora no hay).

Dos vistas de evaluacion:
  A (utilizable): todas las muestras con ventana y horizonte completos.
  B (evaluable) : solo las muestras SIN cuello de botella en ese instante. Es la vista honesta
                  de anticipacion; en A, "avisar" cuando el problema ya existe es trivial.
"""
import os

import numpy as np
import pandas as pd

W = 30   # segundos de historia que ve el modelo
H = 30   # horizonte de prediccion (debe coincidir con --horizonte de etiquetar.py)
SERVICIOS = ["usuarios", "eventos", "reservaciones", "pago", "ticket", "notificaciones"]
FEATS = ["cpu", "ram_mb", "rps", "lat_mean_ms", "lat_p95_ms", "cola", "err_rps"]
# servicios a los que llama cada uno (cadena sincrona: su latencia incluye la de estos)
DEPENDE = {"reservaciones": ["usuarios", "eventos", "pago", "ticket", "notificaciones"],
           "pago": ["ticket", "notificaciones"],
           "ticket": ["notificaciones"]}
DEP = ["cpu_dep", "cola_dep", "lat_p95_dep"]
TEND = ["cpu", "cola", "rps", "lat_p95_ms", "cpu_dep"]
CONT = FEATS + DEP + ["d%d_%s" % (k, f) for f in TEND for k in (5, 15)]
N_CONT = len(CONT)   # columnas continuas de los modelos v7-v9 (las que estandarizan sus checkpoints)
LARGO = ["d60_rps", "d120_rps", "d60_cpu", "d120_cpu", "t_saturar_cpu"]
ARRANQUE = 30        # s tras el inicio del trafico que las tendencias largas no usan como referencia
COLUMNAS = CONT + ["falta"] + ["srv_" + s for s in SERVICIOS] + LARGO
NF = len(COLUMNAS)
NORM = list(range(N_CONT)) + list(range(N_CONT + 1 + len(SERVICIOS), NF))   # columnas que se estandarizan


class Datos(object):
    pass


def _cambio(v, k):
    """v[t] - v[t-k] dentro de la serie (los primeros k segundos comparan con el primero)."""
    prev = np.empty_like(v)
    if len(v) > k:
        prev[k:] = v[:-k]
        prev[:k] = v[0]
    else:
        prev[:] = v[0]
    return v - prev


def _media(v, k=10):
    """Media movil causal de k s (suaviza el ruido segundo a segundo)."""
    c = np.concatenate([[0.0], np.cumsum(v)])
    i = np.arange(1, len(v) + 1)
    lo = np.maximum(i - k, 0)
    return (c[i] - c[lo]) / (i - lo)


def _cambio_largo(v, k, i0):
    """v[t] - v[ref] con ref = max(t - k, i0 + ARRANQUE): nunca compara con el arranque del trafico
    (Locust aun esta creando usuarios). Vale 0 antes de i0 + ARRANQUE."""
    t = np.arange(len(v))
    out = np.zeros(len(v))
    ok = t >= i0 + ARRANQUE
    ref = np.clip(np.maximum(t - k, i0 + ARRANQUE), 0, None)
    out[ok] = v[ok] - v[ref[ok]]
    return out


def largo(cpu, rps, i0):
    """Tendencias largas (columnas LARGO) de un servicio a partir de su CPU (%) y su rps por segundo;
    i0 = indice del primer segundo con trafico. Se usa igual al entrenar y en la demo en vivo."""
    c = _media(np.asarray(cpu, dtype=float))
    r = np.log1p(_media(np.clip(np.asarray(rps, dtype=float), 0, None)))
    pend = _cambio_largo(c, 60, i0) / 60.0          # puntos de CPU por segundo en el ultimo minuto
    seg = np.where(pend > 0.05, np.clip(100.0 - c, 0, None) / np.maximum(pend, 0.05), 600.0)
    return np.column_stack([_cambio_largo(r, 60, i0), _cambio_largo(r, 120, i0),
                            _cambio_largo(c, 60, i0) / 100.0, _cambio_largo(c, 120, i0) / 100.0,
                            np.log1p(np.clip(seg, 0, 600)) / np.log1p(600.0)])


def _transformar(f):
    return np.column_stack([
        f["cpu"].values / 100.0,
        f["ram_mb"].values / 256.0,
        np.log1p(f["rps"].clip(lower=0).values),
        np.log1p(f["lat_mean_ms"].clip(lower=0).values),
        np.log1p(f["lat_p95_ms"].clip(lower=0).values),
        np.log1p(f["cola"].clip(lower=0).values),
        np.log1p(f["err_rps"].clip(lower=0).values),
    ])


def corridas_contaminadas(data_dir):
    """Corridas que no pasan el control de calidad de run_campaign.py (data/calidad_corridas.csv)."""
    q = pd.read_csv(os.path.join(data_dir, "calidad_corridas.csv"))
    return sorted(q[q["calidad"] == "contaminada"]["run_id"])


def cargar(data_dir, sufijo="", objetivo="y_30s", excluir=(), excluir_grupos=()):
    d = pd.read_csv(os.path.join(data_dir, "dataset_etiquetado%s.csv" % sufijo))
    ep = pd.read_csv(os.path.join(data_dir, "episodios%s.csv" % sufijo))
    if len(excluir):
        d = d[~d["run_id"].isin(excluir)]
        ep = ep[~ep["run_id"].isin(excluir)]
    if len(excluir_grupos):
        d = d[~d["grupo"].isin(excluir_grupos)]
        ep = ep[~ep["grupo"].isin(excluir_grupos)]
    agg = {f: "mean" for f in FEATS}
    agg.update({"scrape_ok": "min", "bn_ahora": "max", "y_30s": "max", "y_inicio_30s": "max",
                "utilizable": "min", "evaluable": "min", "t_hasta_inicio": "min"})

    # primera pasada: rejilla de 1 s por (corrida, servicio)
    series = []
    for (run, srv), g in d.groupby(["run_id", "servicio"], sort=True):
        g = g.assign(sec=np.floor(g["t_rel"]).astype(int))
        a = g.groupby("sec").agg(agg)
        full = np.arange(a.index.min(), a.index.max() + 1)
        a = a.reindex(full)
        series.append((run, srv, g["grupo"].iloc[0], full, a, a[FEATS].ffill().fillna(0.0)))
    por_clave = {(run, srv): f.set_index(full) for run, srv, _, full, _, f in series}

    Xs, cpus, bns, ys, y30s, uts, evs, runs, srvs, grs, poss, sts, secs, eprows, ths = ([] for _ in range(15))
    start = 0
    for run, srv, grupo, full, a, f in series:
        T = len(a)
        falta = (a["scrape_ok"].isna() | (a["scrape_ok"] == 0)).values.astype(float)
        onehot = np.zeros((T, len(SERVICIOS)))
        onehot[:, SERVICIOS.index(srv)] = 1.0
        deps = [por_clave[(run, s)].reindex(full).ffill().fillna(0.0)
                for s in DEPENDE.get(srv, []) if (run, s) in por_clave]
        if deps:
            dep = np.column_stack([np.max([x["cpu"].values for x in deps], 0) / 100.0,
                                   np.log1p(np.max([x["cola"].clip(lower=0).values for x in deps], 0)),
                                   np.log1p(np.max([x["lat_p95_ms"].clip(lower=0).values for x in deps], 0))])
        else:
            dep = np.zeros((T, len(DEP)))
        base = np.column_stack([_transformar(f), dep])
        col = {n: base[:, i] for i, n in enumerate(FEATS + DEP)}
        tend = np.column_stack([_cambio(col[x], k) for x in TEND for k in (5, 15)])
        lg = largo(f["cpu"].values, f["rps"].values, int(np.searchsorted(full, 0)))   # full[i0] = s 0 de la carga
        Xs.append(np.column_stack([base, tend, falta, onehot, lg]))
        ths.append(a["t_hasta_inicio"].values)
        cpus.append(f["cpu"].values)
        bns.append(a["bn_ahora"].ffill().fillna(0).values.astype(int))
        ys.append(a[objetivo].ffill().fillna(0).values.astype(int))
        y30s.append(a["y_30s"].ffill().fillna(0).values.astype(int))
        uts.append(a["utilizable"].fillna(0).values.astype(int))
        evs.append(a["evaluable"].fillna(0).values.astype(int))
        runs.append(np.array([run] * T, dtype=object))
        srvs.append(np.array([srv] * T, dtype=object))
        grs.append(np.array([grupo] * T, dtype=object))
        poss.append(np.arange(T))
        sts.append(np.full(T, start))
        secs.append(full)
        e = ep[(ep["run_id"] == run) & (ep["servicio"] == srv)]
        for r in e.itertuples():
            gi = start + int(np.floor(r.t_inicio_rel)) - int(full[0])
            if start <= gi < start + T:
                eprows.append({"run": run, "srv": srv, "grupo": grupo, "causa": r.causa,
                               "previo_libre_s": r.previo_libre_s, "gi": gi})
        start += T

    ds = Datos()
    ds.X = np.vstack(Xs).astype(np.float32)
    ds.cpu = np.concatenate(cpus)
    ds.bn = np.concatenate(bns)
    ds.y = np.concatenate(ys)
    ds.y_30s = np.concatenate(y30s)
    ds.t_hasta = np.concatenate(ths)   # segundos hasta el inicio del proximo episodio (nan si no hay)
    ds.objetivo = objetivo
    ds.util = np.concatenate(uts)
    ds.evalu = np.concatenate(evs)
    ds.run = np.concatenate(runs)
    ds.srv = np.concatenate(srvs)
    ds.grupo = np.concatenate(grs)
    ds.pos = np.concatenate(poss)
    ds.sstart = np.concatenate(sts)
    ds.sec = np.concatenate(secs)
    ds.N = len(ds.y)
    ds.valid = (ds.util == 1) & (ds.pos >= W - 1)
    ds.eps = pd.DataFrame(eprows)
    return ds


# ------------------------------------------------------------------ metricas
def prf(y, alarma, mask, beta=0.5):
    yy = y[mask] == 1
    aa = alarma[mask]
    tp = int((yy & aa).sum())
    fp = int((~yy & aa).sum())
    fn = int((yy & ~aa).sum())
    tn = int((~yy & ~aa).sum())
    p = tp / float(tp + fp) if tp + fp else float("nan")
    r = tp / float(tp + fn) if tp + fn else float("nan")
    f1 = 2.0 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    b2 = beta * beta   # F-beta con beta < 1 pesa mas la precision que el recall
    den = (1 + b2) * tp + b2 * fn + fp
    fb = (1 + b2) * tp / float(den) if den else float("nan")
    fpr = fp / float(fp + tn) if fp + tn else float("nan")   # fraccion de segundos tranquilos con falsa alarma
    return {"precision": p, "recall": r, "f1": f1, "fbeta": fb, "fpr": fpr,
            "tp": tp, "fp": fp, "fn": fn, "n": int(mask.sum())}


def inicios_alarma(ds, alarma):
    """True en los segundos donde la alarma se ENCIENDE (apagada el segundo anterior de la misma serie)."""
    prev = np.zeros(ds.N, dtype=bool)
    prev[1:] = alarma[:-1]
    prev[ds.pos == 0] = False
    return alarma & ~prev


def falsas_alarmas_hora(ds, alarma, mask):
    """Encendidos de alarma sin cuello de botella en los proximos H s, por hora de servicio observada
    en `mask`. Cuenta encendidos, no segundos: una alarma larga cuenta una sola vez."""
    ini = inicios_alarma(ds, alarma) & mask & (ds.y == 0)
    horas = mask.sum() / 3600.0
    return ini.sum() / horas if horas else float("nan")


def lead_times(ds, alarma, min_ventana=3, silencio=5, sostenido=120):
    """Por episodio, lead time HONESTO: solo cuenta una alarma que se ENCIENDE dentro de los H s
    previos al inicio (en un segundo sin cuello de botella) tras >= `silencio` s apagada.
    lead = inicio - instante del encendido. Una alarma que ya estaba encendida al abrir la ventana
    no suma lead: es una alarma casi permanente, no anticipacion.
    Aviso SOSTENIDO (`detectado_ext` / `lead_ext_s`): ademas cuenta una alarma que se encendio hasta
    `sostenido` s antes (tras >= `silencio` s apagada, sin cuello de botella en medio) y siguio sonando
    hasta el inicio (cortes de menos de `silencio` s no la interrumpen); asi un aviso de 40 s no vale 0. Una alarma encendida desde antes de
    esos `sostenido` s, o desde el episodio anterior, sigue sin contar. lead_ext = el mayor de los dos.
    Columnas de control: `alarma_previa` (encendida al abrir la ventana) y `detectado_bruto` /
    `lead_bruto_s` (criterio anterior: cualquier segundo con alarma en la ventana).
    Episodios con menos de `min_ventana` segundos libres antes de empezar se excluyen."""
    filas = []
    for e in ds.eps.itertuples():
        o = e.gi
        s0 = ds.sstart[o] + W - 1   # primer segundo de la serie con prediccion
        lo = max(o - H, s0)
        js = np.arange(lo, o)
        js = js[(ds.bn[js] == 0) & (ds.util[js] == 1)]
        if len(js) < min_ventana:
            continue
        hit = js[alarma[js]]
        limpio = [j for j in hit if j - silencio >= s0 and not alarma[j - silencio:j].any()]
        lead = float(o - limpio[0]) if limpio else np.nan
        # aviso sostenido: se recorre hacia atras la racha de alarma que llega hasta el inicio. Un corte de
        # menos de `silencio` s no la interrumpe (tampoco contaria como encendido nuevo); la racha vale si
        # antes de ella hay >= `silencio` s apagada dentro de la ventana
        lo2 = max(o - sostenido, s0, o - int(np.floor(e.previo_libre_s)))
        k, ini, hueco = o - 1, None, 0
        while k >= lo2:
            if alarma[k]:
                ini, hueco = k, 0
            else:
                hueco += 1
                if hueco >= silencio:
                    break
            k -= 1
        sost = float(o - ini) if (ini is not None and hueco >= silencio) else np.nan
        lead_ext = np.nanmax([lead, sost]) if (lead == lead or sost == sost) else np.nan
        filas.append({"run": e.run, "srv": e.srv, "grupo": e.grupo, "causa": e.causa,
                      "ventana_s": len(js), "detectado": bool(limpio), "lead_s": lead,
                      "detectado_ext": bool(lead_ext == lead_ext), "lead_ext_s": lead_ext,
                      "alarma_previa": bool(alarma[lo]),
                      "detectado_bruto": bool(len(hit)),
                      "lead_bruto_s": float(o - hit[0]) if len(hit) else np.nan})
    return pd.DataFrame(filas)


def resumen_lead(df, por):
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(por)
    out = pd.DataFrame({
        "episodios": g.size(),
        "detectados_%": g["detectado"].mean() * 100,
        "lead_mediana_s": g["lead_s"].median(),
        "lead_media_s": g["lead_s"].mean(),
        "lead>=15s_%": g["lead_s"].apply(lambda s: (s >= 15).mean()) * 100,
        "detectados_ext_%": g["detectado_ext"].mean() * 100,
        "lead_ext_mediana_s": g["lead_ext_s"].median(),
        "lead_ext>=30s_%": g["lead_ext_s"].apply(lambda s: (s >= 30).mean()) * 100,
        "alarma_previa_%": g["alarma_previa"].mean() * 100,
        "detectados_bruto_%": g["detectado_bruto"].mean() * 100,
        "lead_bruto_mediana_s": g["lead_bruto_s"].median(),
    })
    return out.round(1)


def tabla_metricas(ds, alarmas, mask, y=None):
    y = ds.y if y is None else y
    filas = {}
    for nombre, al in alarmas.items():
        m = prf(y, al, mask)
        filas[nombre] = {k: m[k] for k in ("precision", "recall", "f1", "fbeta", "fpr", "tp", "fp", "fn")}
        filas[nombre]["falsas_alarmas_h"] = falsas_alarmas_hora(ds, al, mask)
    return pd.DataFrame(filas).T.round(3)


def alarma_cpu(ds, umbral=80.0):
    return (ds.cpu > umbral) & ds.valid


def alarma_persistencia(ds):
    return (ds.bn == 1) & ds.valid


def reporte(ds, alarmas, mask_extra=None, titulo=""):
    """Imprime y devuelve las tablas de comparacion para un conjunto de alarmas."""
    ex = np.ones(ds.N, dtype=bool) if mask_extra is None else mask_extra
    mA = ds.valid & ex
    mB = ds.valid & (ds.evalu == 1) & ex
    out = {}
    out["A"] = tabla_metricas(ds, alarmas, mA)
    out["B"] = tabla_metricas(ds, alarmas, mB)
    out["lead"] = {}
    for nombre, al in alarmas.items():
        if nombre.startswith("persistencia"):
            continue
        df = lead_times(ds, al)
        if mask_extra is not None and len(df):
            runs_ok = set(ds.run[ex])
            df = df[df["run"].isin(runs_ok)]
        out["lead"][nombre] = df
    return out


def imprimir_reporte(ds, rep, titulo=""):
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)
    print("\nVISTA A - todas las muestras utilizables (etiqueta %s):" % ds.objetivo)
    print(rep["A"].to_string())
    print("\nVISTA B - solo muestras SIN cuello de botella ahora (anticipacion pura):")
    print(rep["B"].to_string())
    for nombre, df in rep["lead"].items():
        print("\nPREDICTION LEAD TIME - %s (por causa del episodio):" % nombre)
        print(resumen_lead(df, "causa").to_string() if len(df) else "(sin episodios evaluables)")
    print("\n(La persistencia no se evalua en lead time: por construccion avisa al inicio, lead = 0.)")
    print("(Lead time honesto: solo cuenta alarmas que se ENCIENDEN en la ventana tras >=5 s apagadas.")
    print(" alarma_previa_% = episodios con la alarma ya encendida al abrir la ventana; *_bruto = criterio anterior.)")
    print("(fbeta = F0.5; falsas_alarmas_h = encendidos sin cuello de botella en 30 s, por hora de servicio.)")