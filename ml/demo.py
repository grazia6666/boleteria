"""
Demo en vivo del modelo final: mide los 6 servicios cada segundo, arma las mismas entradas que en el
entrenamiento y muestra la probabilidad de cuello de botella en los proximos 30 s, junto a la regla
CPU > 80 %. Cuando un servicio entra de verdad en cuello de botella, dice con cuantos segundos de
anticipacion aviso cada metodo.

Uso (desde la raiz del proyecto, con los servicios levantados: docker compose up -d):
  terminal 1:  python ml/demo.py
  terminal 2 (degradacion gradual y suave en pago: cpu_load sube de 0 a 40 en 150 s):
         python run_experiment.py --scenario normal --duration 280 --run-id demo_pago \
             --inject pago:70:170:40:150
  Es el caso que el modelo puede anticipar. Las rafagas (--scenario burst) y las inyecciones abruptas
  saturan de golpe, sin senal previa: sirven para mostrar que ningun metodo puede anticiparlas y que
  ambos las detectan en ~1 s.

La demo espera a que haya trafico y junta 60 s antes de predecir ("calentando"): el arranque de Locust
confunde al modelo. Si el trafico se corta, vuelve a esperar.

Modo repeticion (sin Docker, siempre igual; ideal para presentar):
  python ml/demo.py --replay suave_inj_40_ticket_r2 --velocidad 2
  Reproduce una corrida grabada con las predicciones de la VALIDACION CRUZADA (data/resultados_v7/
  predicciones.csv): cada corrida la predijo un modelo que no la vio al entrenar, asi que es honesto.
  En suave_inj_40_ticket_r2, v7 avisa 27 s antes del cuello de botella de ticket y la regla CPU > 80 %
  no lo detecta. El "estado real" es la etiqueta de cuello de botella propio de esa corrida.

Opciones:
  --replay RUN      corrida de data/ a reproducir (en vez de medir en vivo)
  --velocidad X     en repeticion, X segundos de la corrida por segundo real (por defecto 1)
  --resultados DIR  carpeta de data/ con predicciones.csv para --replay (por defecto resultados_v7)
  --modelo RUTA     modelo a usar (por defecto data/resultados_v7/models/lstm_final.pt)
  --umbral X        corte de la alarma (por defecto el guardado en el modelo, 0.83 en v7)
  --log RUTA.csv    guarda cada segundo: servicio, metricas, probabilidad y alarmas
  --sin-pantalla    una linea de texto por segundo en vez de la tabla (para grabar o depurar)
  --duracion S      terminar sola despues de S segundos (0 = hasta Ctrl+C)

La deteccion del cuello de botella real usa la misma regla que el etiquetado (etiqueta propia):
p95 > umbral del servicio y, en los que llaman a otros, tiempo propio > 50 ms, durante 3 s seguidos.
"""
import argparse
import collections
import csv
import json
import os
import sys
import threading
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "ml"))
sys.path.insert(0, os.path.join(HERE, "collector"))
import datos as D  # noqa: E402
import collector as C  # noqa: E402

PISO_PROPIO_MS = 50.0   # igual que etiquetar.py --piso-propio-ms
SOSTENIDO = 3           # segundos seguidos para declarar cuello de botella (etiquetar.py --sostenido)
SILENCIO = 5            # segundos apagada antes de un encendido para contar anticipacion (lead honesto)
HIST = 60               # segundos de historia guardados (ventana de 30 s + 15 s para las tendencias)
CALENTAR = 60           # segundos CON trafico antes de predecir: el arranque de Locust (usuarios sumandose)
                        # confunde al modelo en reservaciones; es el mismo efecto que deja ~30 s de falsa
                        # alarma en normal_base en la evaluacion

actual = {}             # servicio -> ultima medicion (la escriben los hilos de muestreo)
lock = threading.Lock()


# ------------------------------------------------------------------ medicion
def muestreador(srv, puerto, stop):
    """Hilo por servicio: igual que el collector, una lectura de /metrics por segundo."""
    prev = None
    while not stop.is_set():
        t = time.time()
        fila = {"rps": np.nan, "lat_mean_ms": np.nan, "lat_p95_ms": np.nan, "cola": np.nan,
                "err_rps": np.nan, "ok": 0, "lat_dep_ms": 0.0}
        try:
            cur = C.scrape(puerto)
            fila["cola"] = cur["inflight"]
            fila["ok"] = 1
            if prev is not None:
                dt = cur["t"] - prev["t"]
                d = C.delta_metrics(cur, prev, dt)
                if d:
                    fila.update(d)
                dep = 0.0
                for tgt, ecur in cur["edges"].items():
                    ed = C.delta_metrics(ecur, prev["edges"][tgt], dt) if tgt in prev["edges"] else None
                    if ed:
                        dep += ed["lat_mean_ms"]
                fila["lat_dep_ms"] = dep
            prev = cur
        except Exception:
            pass
        with lock:
            actual[srv] = fila
        stop.wait(max(0.0, 1.0 - (time.time() - t)))


def foto():
    """Una fila por servicio con todas las metricas del ultimo segundo."""
    out = {}
    with lock:
        for s in D.SERVICIOS:
            f = dict(actual.get(s, {"ok": 0}))
            st = C.latest.get(s, {})
            f["cpu"] = st.get("cpu", np.nan)
            f["ram_mb"] = st.get("ram_mb", np.nan)
            out[s] = f
    return out


# ------------------------------------------------------------------ entradas del modelo
def serie(snaps, srv):
    df = pd.DataFrame([s[srv] for s in snaps])
    for c in D.FEATS:
        if c not in df:
            df[c] = np.nan
    return df[D.FEATS].astype(float).ffill().fillna(0.0), df.get("ok", pd.Series([0] * len(df)))


def entradas(snaps, srv):
    """Mismas 27 columnas que D.cargar (ver datos.py), sobre la historia reciente; devuelve las ultimas W."""
    f, ok = serie(snaps, srv)
    T = len(f)
    falta = (ok.fillna(0).values == 0).astype(float)
    deps = [serie(snaps, d)[0] for d in D.DEPENDE.get(srv, [])]
    if deps:
        dep = np.column_stack([np.max([x["cpu"].values for x in deps], 0) / 100.0,
                               np.log1p(np.max([x["cola"].clip(lower=0).values for x in deps], 0)),
                               np.log1p(np.max([x["lat_p95_ms"].clip(lower=0).values for x in deps], 0))])
    else:
        dep = np.zeros((T, len(D.DEP)))
    base = np.column_stack([D._transformar(f), dep])
    col = {n: base[:, i] for i, n in enumerate(D.FEATS + D.DEP)}
    tend = np.column_stack([D._cambio(col[x], k) for x in D.TEND for k in (5, 15)])
    onehot = np.zeros((T, len(D.SERVICIOS)))
    onehot[:, D.SERVICIOS.index(srv)] = 1.0
    return np.column_stack([base, tend, falta, onehot])[-D.W:]


def cargar_modelo(ruta):
    import torch
    import torch.nn as nn
    ck = torch.load(ruta, weights_only=False)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(len(ck["cols"]), ck["hidden"], batch_first=True)
            self.fc = nn.Linear(ck["hidden"], 1)

        def forward(self, x):
            o, _ = self.lstm(x)
            return self.fc(o[:, -1]).squeeze(-1)

    net = Net()
    net.load_state_dict(ck["state"])
    net.eval()
    mu, sd = np.array(ck["mu"], dtype=np.float32), np.array(ck["sd"], dtype=np.float32)

    def predecir(X):   # X: (servicios, W, 27)
        X = X.astype(np.float32).copy()
        X[:, :, :D.N_CONT] = (X[:, :, :D.N_CONT] - mu) / sd
        with torch.no_grad():
            return torch.sigmoid(net(torch.from_numpy(X[:, :, ck["cols"]]))).numpy()
    return predecir, ck


# ------------------------------------------------------------------ eventos
class Estado(object):
    """Historia de un servicio: alarmas de ambos metodos y deteccion del cuello de botella real."""

    def __init__(self):
        self.al_v7 = collections.deque(maxlen=HIST)
        self.al_cpu = collections.deque(maxlen=HIST)
        self.racha = 0
        self.calma = 0
        self.en_bn = False


def lead(alarmas, o):
    """Lead honesto: encendido dentro de los 30 s previos al inicio o, tras >= SILENCIO s apagada."""
    a = list(alarmas)
    for j in range(max(0, o - D.H), o):
        if a[j] and j - SILENCIO >= 0 and not any(a[j - SILENCIO:j]):
            return "avisó %d s antes" % (o - j)
    if o > 0 and a[o - 1]:
        return "ya estaba encendida (no cuenta)"
    return "sin aviso previo"


def condicion(fila, srv, umbrales):
    if not fila.get("ok"):
        return True
    p95 = fila.get("lat_p95_ms")
    if p95 is None or np.isnan(p95) or p95 <= umbrales[srv]:
        return False
    if srv in D.DEPENDE:   # servicio que llama a otros: debe ser lentitud propia, no heredada
        propio = (fila.get("lat_mean_ms") or 0.0) - (fila.get("lat_dep_ms") or 0.0)
        return propio > PISO_PROPIO_MS
    return True


# ------------------------------------------------------------------ pantalla
def barra(p, n=10):
    k = int(round(p * n))
    return "█" * k + "░" * (n - k)


def pintar(t_seg, snap, probs, alarm_v7, alarm_cpu, estados, eventos, umbral, calentando):
    rojo, amar, verde, gris, fin = "\033[91m", "\033[93m", "\033[92m", "\033[90m", "\033[0m"
    L = ["\033[H\033[J",
         "DEMO v7  ·  ¿cuello de botella en los próximos 30 s?   t = %d s   umbral = %.2f" % (t_seg, umbral),
         "",
         "%-15s %5s %8s %5s  %-18s %-11s %-9s %s" % ("servicio", "CPU", "p95(ms)", "cola", "P(cuello en 30 s)",
                                                    "v7", "CPU>80%", "estado real")]
    for s in D.SERVICIOS:
        f = snap[s]
        cpu = f.get("cpu", np.nan)
        p95 = f.get("lat_p95_ms", np.nan)
        cola = f.get("cola", np.nan)
        if calentando:
            pr = gris + "calentando %2d/%d" % (CALENTAR - calentando, CALENTAR) + fin + "  "
            if CALENTAR - calentando <= 1:
                pr = gris + "esperando tráfico " + fin + "  "
        else:
            p = probs[s]
            c = rojo if p > umbral else (amar if p > umbral / 2 else verde)
            pr = c + barra(p) + " %.2f" % p + fin + "   "
        v7 = (rojo + "⚠ ALARMA" + fin + "   ") if alarm_v7.get(s) else (verde + "ok" + fin + "         ")
        cp = (rojo + "⚠ ALARMA" + fin) if alarm_cpu.get(s) else (verde + "ok" + fin + "      ")
        real = (rojo + "CUELLO DE BOTELLA" + fin) if estados[s].en_bn else (gris + "normal" + fin)
        num = lambda v, fmt: (fmt % v) if v == v else "  -"  # noqa: E731
        L.append("%-15s %5s %8s %5s  %s%s  %s  %s" % (s, num(cpu, "%4.0f%%"), num(p95, "%7.0f"),
                                                     num(cola, "%4.0f"), pr, v7, cp, real))
    L += ["", "Eventos:"] + ["  " + e for e in list(eventos)[-10:]]
    L += ["", gris + "Ctrl+C para salir" + fin]
    sys.stdout.write("\n".join(L) + "\n")
    sys.stdout.flush()


# ------------------------------------------------------------------ repeticion
def repeticion(a):
    """Reproduce una corrida grabada con las predicciones fuera de muestra de la validacion cruzada."""
    pred = pd.read_csv(os.path.join(HERE, "data", a.resultados, "predicciones.csv"))
    pred = pred[pred["run"] == a.replay]
    if pred.empty:
        raise SystemExit("La corrida %s no esta en %s/predicciones.csv" % (a.replay, a.resultados))
    cfg = json.load(open(os.path.join(HERE, "data", a.resultados, "config.json")))
    umbral = float(np.median(cfg["umbrales_por_fold"]))
    m = pd.read_csv(os.path.join(HERE, "data", a.replay, "metrics.csv"))
    t0 = json.load(open(os.path.join(HERE, "data", a.replay, "meta.json")))["t0_locust"]
    m["sec"] = np.floor(m["timestamp"] - t0).astype(int)
    met = m.groupby(["sec", "servicio"])[["cpu", "lat_p95_ms", "cola"]].mean()
    p = pred.set_index(["t_rel", "servicio"])
    secs = range(int(max(0, pred["t_rel"].min() - 5)), int(pred["t_rel"].max()) + 1)
    estados = {s: Estado() for s in D.SERVICIOS}
    eventos = collections.deque(maxlen=50)
    print("Repeticion de %s con las predicciones de la validacion cruzada (%s)" % (a.replay, a.resultados))
    time.sleep(1.5)
    for k, t in enumerate(secs):
        snap, probs, alarm_v7, alarm_cpu = {}, {}, {}, {}
        for s in D.SERVICIOS:
            row = met.loc[(t, s)] if (t, s) in met.index else None
            snap[s] = {"cpu": row["cpu"] if row is not None else np.nan,
                       "lat_p95_ms": row["lat_p95_ms"] if row is not None else np.nan,
                       "cola": row["cola"] if row is not None else np.nan}
            e = estados[s]
            hay = (t, s) in p.index
            if hay:
                probs[s] = float(p.loc[(t, s), "prob_lstm"])
            alarm_v7[s] = bool(hay and p.loc[(t, s), "alarma_lstm"])
            cpu = snap[s]["cpu"]
            alarm_cpu[s] = bool(cpu == cpu and cpu > 80.0)
            if alarm_v7[s] and (not e.al_v7 or not e.al_v7[-1]) and not e.en_bn:
                eventos.append("[t=%3d s] v7 avisa en %s (P = %.2f)" % (t, s, probs[s]))
            e.al_v7.append(alarm_v7[s])
            e.al_cpu.append(alarm_cpu[s])
            bn = bool(hay and p.loc[(t, s), "bn_ahora"])
            if bn and not e.en_bn:
                o = len(e.al_v7) - 1
                eventos.append("[t=%3d s] %s ENTRA EN CUELLO DE BOTELLA → v7: %s | CPU>80%%: %s"
                               % (t, s.upper(), lead(e.al_v7, o), lead(e.al_cpu, o)))
            elif e.en_bn and not bn:
                eventos.append("[t=%3d s] %s se recuperó" % (t, s))
            e.en_bn = bn
        calentando = 0 if probs else 1
        if a.sin_pantalla:
            while eventos:
                print("  " + eventos.popleft())
        else:
            pintar(t, snap, {s: probs.get(s, 0.0) for s in D.SERVICIOS}, alarm_v7, alarm_cpu, estados, eventos,
                   umbral, calentando)
        time.sleep(1.0 / a.velocidad)
    print("\nRepeticion terminada.")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=os.path.join(HERE, "data", "resultados_v7", "models", "lstm_final.pt"))
    ap.add_argument("--umbrales", default=os.path.join(HERE, "data", "umbrales_propio.json"))
    ap.add_argument("--umbral", type=float, default=None)
    ap.add_argument("--log", default="")
    ap.add_argument("--sin-pantalla", action="store_true")
    ap.add_argument("--duracion", type=float, default=0)
    ap.add_argument("--replay", default="")
    ap.add_argument("--velocidad", type=float, default=1.0)
    ap.add_argument("--resultados", default="resultados_v7")
    a = ap.parse_args()

    os.system("")  # activa los colores ANSI en la consola de Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # flechas y barras tambien en consolas cp1252
    except Exception:
        pass
    if a.replay:
        return repeticion(a)
    predecir, ck = cargar_modelo(a.modelo)
    umbral = a.umbral if a.umbral is not None else ck["umbral"]
    umbrales = {s: v["umbral_ms"] for s, v in json.load(open(a.umbrales))["servicios"].items()}
    print("Modelo: %s  (umbral %.2f)" % (a.modelo, umbral))

    import docker
    client = docker.from_env()
    for s in D.SERVICIOS:
        threading.Thread(target=C.stats_worker, args=(client, s), daemon=True).start()
    stop = threading.Event()
    for s, p in C.SERVICIOS.items():
        threading.Thread(target=muestreador, args=(s, p, stop), daemon=True).start()
    time.sleep(2.5)

    snaps = collections.deque(maxlen=HIST)
    estados = {s: Estado() for s in D.SERVICIOS}
    eventos = collections.deque(maxlen=50)
    log = None
    if a.log:
        log = csv.writer(open(a.log, "w", newline=""))
        log.writerow(["t", "servicio", "cpu", "lat_p95_ms", "cola", "prob_v7", "alarma_v7", "alarma_cpu",
                      "cuello_real"])
    t0 = time.time()
    k = 0
    con_trafico = 0
    sin_trafico = 0
    try:
        while True:
            k += 1
            time.sleep(max(0.0, t0 + k - time.time()))
            if a.duracion and time.time() - t0 > a.duracion:
                break
            snap = foto()
            # sin trafico no hay nada que predecir; ademas, una ventana que mezcla "sin carga" y "carga"
            # no se parece a lo entrenado (alarma falsa al arrancar). Se espera una ventana completa CON trafico.
            # Si /metrics no responde (servicio saturado) NO es falta de trafico: ese segundo queda como
            # dato faltante, igual que en el entrenamiento. Solo se reinicia con trafico medido ~0 durante 5 s.
            rps_res = snap["reservaciones"].get("rps", np.nan)
            sin_trafico = sin_trafico + 1 if (rps_res == rps_res and rps_res < 0.5) else 0
            if sin_trafico >= 5:
                con_trafico = 0
                snaps.clear()
            elif sin_trafico == 0:
                con_trafico += 1
            snaps.append(snap)
            calentando = max(0, CALENTAR - con_trafico)
            probs, alarm_v7, alarm_cpu = {}, {}, {}
            if not calentando:
                X = np.stack([entradas(snaps, s) for s in D.SERVICIOS])
                for s, p in zip(D.SERVICIOS, predecir(X)):
                    probs[s] = float(p)
            hora = time.strftime("%H:%M:%S")
            for s in D.SERVICIOS:
                e = estados[s]
                alarm_v7[s] = bool(probs.get(s, 0.0) > umbral)
                cpu = snap[s].get("cpu", np.nan)
                alarm_cpu[s] = bool(cpu == cpu and cpu > 80.0)
                if alarm_v7[s] and (not e.al_v7 or not e.al_v7[-1]) and not e.en_bn:
                    eventos.append("[%s] v7 avisa en %s (P = %.2f)" % (hora, s, probs[s]))
                e.al_v7.append(alarm_v7[s])
                e.al_cpu.append(alarm_cpu[s])
                cond = k > 10 and condicion(snap[s], s, umbrales)
                if cond:
                    e.racha += 1
                    e.calma = 0
                else:
                    e.racha = 0
                    e.calma += 1
                if not e.en_bn and e.racha >= SOSTENIDO:
                    e.en_bn = True
                    o = len(e.al_v7) - SOSTENIDO   # el episodio empezo hace SOSTENIDO - 1 s
                    eventos.append("[%s] %s ENTRA EN CUELLO DE BOTELLA → v7: %s | CPU>80%%: %s"
                                   % (hora, s.upper(), lead(e.al_v7, o), lead(e.al_cpu, o)))
                elif e.en_bn and e.calma >= SILENCIO:
                    e.en_bn = False
                    eventos.append("[%s] %s se recuperó" % (hora, s))
                if log:
                    log.writerow([round(time.time() - t0), s, cpu, snap[s].get("lat_p95_ms"), snap[s].get("cola"),
                                  probs.get(s, ""), int(alarm_v7[s]), int(alarm_cpu[s]), int(e.en_bn)])
            if a.sin_pantalla:
                linea = " | ".join("%s %s" % (s[:5], ("%.2f" % probs[s]) if s in probs else "--")
                                   for s in D.SERVICIOS)
                print("t=%3d  %s" % (k, linea))
                while eventos:
                    print("  " + eventos.popleft())
            else:
                pintar(k, snap, probs, alarm_v7, alarm_cpu, estados, eventos, umbral, calentando)
    except KeyboardInterrupt:
        pass
    stop.set()
    print("\nDemo terminada.")
    os._exit(0)   # los hilos de docker stats son bloqueantes


if __name__ == "__main__":
    main()
