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
confunde al modelo. Si el trafico se corta, vuelve a esperar. El t de la pantalla y de los eventos son
segundos desde el inicio de la carga, el mismo reloj de las graficas.

Modo repeticion (sin Docker, siempre igual; ideal para presentar):
  python ml/demo.py --replay var_inj_leve_notificaciones_r1 --velocidad 2
  Reproduce una corrida grabada con las predicciones de la VALIDACION CRUZADA (data/<resultados>/
  predicciones.csv): cada corrida la predijo un modelo que no la vio al entrenar, asi que es honesto.
  Ejemplos con v12 (en la falla gradual y la lenta, la CPU cruza el 80 % cuando el problema ya empezo; en el
  progresivo ya estaba encendida desde 63 s antes y no cuenta como aviso):
    var_inj_leve_notificaciones_r1  falla gradual en notificaciones: avisa 19 s antes (14 s antes del primer pico
                                    de latencia: anticipacion real)
    rnd_lenta_06                    falla lenta en ticket (sube en 2 min): avisa 22 s antes (3 s antes del primer pico)
    rnd_progresivo_08               crecimiento progresivo: avisa 27 s antes del cuello de botella PROPIO de
                                    reservaciones, pero 2 s despues de que su latencia pasara el umbral (estaba lento
                                    esperando a otros): no sirve como ejemplo de anticipacion (README 13.9)
  El "estado real" es la etiqueta de cuello de botella propio de esa corrida.

Modelo: por defecto el final, v12 = ensamble de 3 LSTM entrenado con 175 corridas, 95 de ellas de
calendario aleatorio, incluidas 30 de caidas en cadena (data/resultados_v12/modelo_ensamble.json): la
probabilidad es el promedio de los tres y el umbral 0.90. Si no existe, usa v11, v9, v8 y luego v7.
Para comparar con v11:
  python ml/demo.py --modelo data/resultados_v11/modelo_ensamble.json
  python ml/demo.py --replay var_inj_leve_notificaciones_r1 --resultados resultados_v11

Opciones:
  --replay RUN      corrida de data/ a reproducir (en vez de medir en vivo)
  --velocidad X     en repeticion, X segundos de la corrida por segundo real (por defecto 1)
  --resultados DIR  carpeta de data/ con predicciones.csv para --replay (por defecto resultados_v12)
  --modelo RUTA     modelo (.pt) o ensamble (.json) (por defecto el ensamble v12)
  --umbral X        corte de la alarma (por defecto el guardado en el modelo: 0.90 en v12, 0.883 en v11)
  --log RUTA.csv    guarda cada segundo: servicio, metricas, probabilidad y alarmas
  --sin-pantalla    una linea de texto por segundo en vez de la tabla (para grabar o depurar)
  --duracion S      terminar sola despues de S segundos (0 = hasta Ctrl+C)
  --exportar PUERTO publica el estado de cada servicio, la probabilidad y las alarmas para Prometheus y Grafana
                    (por defecto 9108; 0 = no publicar). Tablero: monitoreo/README.md

La deteccion del cuello de botella real usa la misma regla que el etiquetado (etiqueta propia):
p95 > umbral del servicio y, en los que llaman a otros, tiempo propio > 50 ms, durante 3 s seguidos.

Dos ajustes sobre el modelo, sin reentrenar (los mismos de ml/evidencia_vivo.py):
  - causa raiz (ml/causa_raiz.py): si un servicio y alguno de los que llama estan en alarma (en los
    ultimos 5 s), la alarma queda en el causante y el otro muestra "↪ <causa>" en vez de ALARMA.
  - probabilidad calibrada (ml/calibrar.py, data/<resultados>/calibracion.json): la barra muestra una
    probabilidad real. La alarma sigue siendo puntaje > umbral (0,90), que equivale a ~0,51 calibrado.
"""
import argparse
import collections
import csv
import glob
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
import calibrar as CAL  # noqa: E402
import causa_raiz as CR  # noqa: E402
import datos as D  # noqa: E402
import collector as C  # noqa: E402

PISO_PROPIO_MS = 50.0   # igual que etiquetar.py --piso-propio-ms
SOSTENIDO = 3           # segundos seguidos para declarar cuello de botella (etiquetar.py --sostenido)
SILENCIO = 5            # segundos apagada antes de un encendido para contar anticipacion (lead honesto)
HIST = 60               # segundos de historia guardados (ventana de 30 s + 15 s para las tendencias)
NOMBRE = "LSTM"         # como se llama el modelo en pantalla (v7, v8...): se deduce de la ruta
# modelo por defecto: el mas nuevo que exista (ensamble v12, v11, v9, v8, v7)
_CANDIDATOS = [("resultados_v12", os.path.join("resultados_v12", "modelo_ensamble.json")),
               ("resultados_v11", os.path.join("resultados_v11", "modelo_ensamble.json")),
               ("resultados_v9", os.path.join("resultados_v9", "modelo_ensamble.json")),
               ("resultados_v8", os.path.join("resultados_v8", "modelo_ensamble.json")),
               ("resultados_v7", os.path.join("resultados_v7", "models", "lstm_final.pt"))]
RESULTADOS_DEF, MODELO_DEF = next(((r, os.path.join(HERE, "data", m)) for r, m in _CANDIDATOS
                                   if os.path.exists(os.path.join(HERE, "data", m))),
                                  (_CANDIDATOS[-1][0], os.path.join(HERE, "data", _CANDIDATOS[-1][1])))


def nombre_de(texto):
    import re
    m = re.search(r"resultados_(v\d+)", texto.replace("\\", "/"))
    return m.group(1) if m else "LSTM"
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


def entradas(snaps, srv, i0=0):
    """Mismas columnas que D.cargar (ver datos.py), sobre la historia reciente; devuelve las ultimas W.
    i0 = posicion en `snaps` del primer segundo con trafico (para las tendencias largas)."""
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
    lg = D.largo(f["cpu"].values, f["rps"].values, i0)
    return np.column_stack([base, tend, falta, onehot, lg])[-D.W:]


def cargar_uno(ruta):
    import torch
    import torch.nn as nn
    ck = torch.load(ruta, weights_only=False)

    capas = ck.get("capas", 1)   # modelos anteriores al barrido no guardan capas/dropout: 1 capa, sin dropout
    D.W = ck.get("W", D.W)        # la ventana con que se entreno el modelo

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(len(ck["cols"]), ck["hidden"], num_layers=capas, batch_first=True)
            self.fc = nn.Linear(ck["hidden"], 1)

        def forward(self, x):   # en inferencia el dropout no actua, asi que no hace falta la capa
            o, _ = self.lstm(x)
            return self.fc(o[:, -1]).squeeze(-1)

    net = Net()
    net.load_state_dict(ck["state"])
    net.eval()
    mu, sd = np.array(ck["mu"], dtype=np.float32), np.array(ck["sd"], dtype=np.float32)
    norm = ck.get("norm", list(range(len(mu))))   # v7-v9 estandarizan las primeras N_CONT columnas

    def predecir(X):   # X: (servicios, W, columnas de datos.COLUMNAS)
        X = X.astype(np.float32).copy()
        X[:, :, norm] = (X[:, :, norm] - mu) / sd
        with torch.no_grad():
            return torch.sigmoid(net(torch.from_numpy(X[:, :, ck["cols"]]))).numpy()
    return predecir, ck


def cargar_modelo(ruta):
    """Un modelo (.pt) o un ensamble (.json con "modelos" y "umbral", lo escribe ensamble.py --salida):
    en el ensamble la probabilidad es el promedio de los modelos y el umbral es el del ensamble."""
    if not ruta.endswith(".json"):
        return cargar_uno(ruta)
    cfg = json.load(open(ruta))
    base = os.path.join(HERE, "data")
    partes = [cargar_uno(os.path.join(base, m)) for m in cfg["modelos"]]
    ck = dict(partes[0][1])
    ck["umbral"] = cfg["umbral"]
    ck["n_modelos"] = len(partes)

    def predecir(X):
        return np.mean([p(X) for p, _ in partes], axis=0)
    return predecir, ck


# ------------------------------------------------------------------ eventos
class Estado(object):
    """Historia de un servicio: alarmas de ambos metodos y deteccion del cuello de botella real."""

    def __init__(self):
        self.al_v7 = collections.deque(maxlen=HIST)    # alarma PROPIA del modelo (despues de causa raiz)
        self.al_raw = collections.deque(maxlen=HIST)   # alarma del modelo antes de causa raiz
        self.espera = collections.deque(maxlen=HIST)   # alarma pasada a la causa
        self.causa = ""                                # ultima causa senalada
        self.al_cpu = collections.deque(maxlen=HIST)
        self.racha = 0
        self.calma = 0
        self.en_bn = False


def paso_modelo(t, estados, alarma, probs, eventos):
    """Aplica causa raiz a las alarmas del modelo de este segundo ({servicio: puntaje > umbral}), guarda la
    historia y anota los eventos. Devuelve {servicio: causa} de las alarmas pasadas a la causa."""
    for s in D.SERVICIOS:
        estados[s].al_raw.append(bool(alarma.get(s)))
    recientes = {s: any(list(estados[s].al_raw)[-(CR.VENTANA + 1):]) for s in D.SERVICIOS}
    espera = {}
    for s in D.SERVICIOS:
        e = estados[s]
        causa = CR.raiz(s, recientes) if alarma.get(s) else None
        propia = bool(alarma.get(s)) and not causa
        if propia and (not e.al_v7 or not e.al_v7[-1]) and not e.en_bn:
            eventos.append("[t=%3d s] %s avisa en %s (P = %.2f)" % (t, NOMBRE, s, probs[s]))
        if causa and not any(list(e.espera)[-SILENCIO:]) and not e.en_bn:
            eventos.append("[t=%3d s] %s en alarma, pero espera a %s: el aviso pasa a la causa" % (t, s, causa))
        e.al_v7.append(propia)
        e.espera.append(bool(causa))
        if causa:
            e.causa = causa
            espera[s] = causa
    return espera


def lead_modelo(e, o):
    """Lead del modelo para un episodio que empieza en la posicion o: el propio o, si no aviso por si mismo
    y su alarma se paso a la causa en los 30 s previos, "avisó la causa"."""
    r = lead(e.al_v7, o)
    if not r.startswith("avisó") and any(list(e.espera)[max(0, o - D.H):o]):
        return "avisó la causa (%s)" % e.causa
    return r


class Exportador(object):
    """Publica lo que ve la demo como metricas de Prometheus (http://localhost:PUERTO/metrics), para el tablero de
    Grafana de monitoreo/. Estado por servicio: 0 normal, 1 en alarma pero espera a otro, 2 el modelo predice un
    cuello de botella, 3 cuello de botella real."""

    def __init__(self, puerto):
        from prometheus_client import CollectorRegistry, Gauge, start_http_server
        r = CollectorRegistry()
        g = lambda n, d, l=("servicio",): Gauge(n, d, list(l), registry=r)  # noqa: E731
        self.estado = g("boleteria_estado", "0 normal, 1 espera a otro, 2 el modelo predice, 3 cuello de botella")
        self.prob = g("boleteria_probabilidad", "Probabilidad calibrada de cuello de botella en 30 s")
        self.puntaje = g("boleteria_puntaje", "Puntaje del modelo sin calibrar")
        self.alarma = g("boleteria_alarma_modelo", "Alarma propia del modelo (despues de causa raiz)")
        self.causa = g("boleteria_causa", "Alarma pasada a la causa: 1 en la serie de la causa actual",
                       ("servicio", "causa"))
        self.cpu_al = g("boleteria_alarma_cpu", "Regla CPU > 80 %")
        self.real = g("boleteria_cuello_real", "Cuello de botella real (misma regla que el etiquetado)")
        self.cpu = g("boleteria_cpu", "CPU del contenedor (%)")
        self.p95 = g("boleteria_p95_ms", "Latencia p95 del ultimo segundo (ms)")
        self.cola = g("boleteria_cola", "Peticiones en curso")
        self.umbral = Gauge("boleteria_umbral", "Umbral de la alarma en la escala que se muestra", registry=r)
        self.t = Gauge("boleteria_t_carga", "Segundos desde el inicio de la carga", registry=r)
        self.calentando = Gauge("boleteria_calentando", "Segundos que faltan para predecir", registry=r)
        self.hace = g("boleteria_alarma_hace_s", "Segundos desde que se encendio la alarma propia (NaN si apagada)")
        self.ult_ini = g("boleteria_ultimo_inicio_s", "t de carga en que empezo el ultimo cuello de botella")
        self.ult_lead = g("boleteria_ultimo_lead_s", "Segundos que el modelo aviso antes del ultimo cuello de botella")
        self.ult_aviso = g("boleteria_ultimo_aviso", "Resultado del ultimo cuello de botella (en las etiquetas)",
                           ("servicio", "modelo", "cpu"))
        self.episodio = g("boleteria_episodio", "Cada cuello de botella: el valor es su t de carga de inicio",
                          ("servicio", "n", "modelo", "cpu", "sesion"))
        self.evento = g("boleteria_evento", "Cada evento de la pantalla de la demo: el valor es su t de carga",
                        ("n", "tipo", "texto", "sesion"))
        self.sesion_g = g("boleteria_sesion", "1 en la sesion actual (una por carga)", ("sesion",))
        self.orden = g("boleteria_orden", "Posicion del servicio en la cadena de llamadas")
        for i, s in enumerate(["reservaciones", "pago", "ticket", "notificaciones", "usuarios", "eventos"]):
            self.orden.labels(s).set(i)
        self.ultima, self.ult_etiq, self.historial = {}, {}, collections.deque()
        self.desde, self.ult_on = {}, {}
        self.eventos_vistos, self.n_evento = collections.deque(maxlen=300), 0
        self.eventos_pub = collections.deque()
        self.sesion = None
        self.nueva_sesion()
        start_http_server(puerto, addr="0.0.0.0", registry=r)

    def nueva_sesion(self):
        """Empieza una carga nueva: el tablero muestra solo los eventos y cuellos de botella de esta sesion."""
        if self.sesion:
            self.sesion_g.remove(self.sesion)
        self.sesion = time.strftime("%H:%M:%S")
        self.sesion_g.labels(self.sesion).set(1)
        for s, etiq in list(self.ult_etiq.items()):
            self.ult_aviso.remove(s, *etiq)
            self.ult_ini.remove(s)
            self.ult_lead.remove(s)
        self.ult_etiq, self.desde, self.ult_on = {}, {}, {}

    def eventos_nuevos(self, eventos, t):
        for texto in list(eventos):
            if texto in self.eventos_vistos:
                continue
            self.eventos_vistos.append(texto)
            self.n_evento += 1
            tipo = ("cuello" if "ENTRA EN CUELLO" in texto else "recupera" if "se recuperó" in texto else
                    "espera" if "espera a" in texto else "aviso" if "avisa en" in texto else "otro")
            cuerpo = texto.split("] ", 1)[-1]
            try:
                t_ev = int(texto.split("t=")[1].split("s]")[0])
            except (IndexError, ValueError):
                t_ev = t
            etiq = (str(self.n_evento), tipo, cuerpo, self.sesion)
            self.evento.labels(*etiq).set(t_ev)
            self.eventos_pub.append(etiq)
            if len(self.eventos_pub) > 80:   # solo los ultimos 80 eventos
                self.evento.remove(*self.eventos_pub.popleft())

    def episodio_nuevo(self, s, t_inicio, lead_s, texto_modelo, texto_cpu):
        """Un servicio entro en cuello de botella: guarda cuanto antes aviso el modelo (y la CPU)."""
        self.ult_ini.labels(s).set(t_inicio)
        self.ult_lead.labels(s).set(lead_s if lead_s is not None else float("nan"))
        if s in self.ult_etiq:
            self.ult_aviso.remove(s, *self.ult_etiq[s])
        self.ult_etiq[s] = (texto_modelo, texto_cpu)
        self.ult_aviso.labels(s, texto_modelo, texto_cpu).set(1)
        etiq = (s, str(len(self.historial) + 1), texto_modelo, texto_cpu, self.sesion)
        self.episodio.labels(*etiq).set(t_inicio)
        self.historial.append(etiq)
        if len(self.historial) > 40:
            self.episodio.remove(*self.historial.popleft())

    def publicar(self, t, snap, probs, puntajes, estados, espera, alarm_cpu, umbral, calentando, eventos=()):
        self.eventos_nuevos(eventos, t)
        self.t.set(t)
        self.umbral.set(umbral)
        self.calentando.set(calentando)
        for s in D.SERVICIOS:
            e, f = estados[s], snap.get(s, {})
            propia = bool(e.al_v7 and e.al_v7[-1])
            c = espera.get(s, "")
            self.estado.labels(s).set(3 if e.en_bn else 2 if propia else 1 if c else 0)
            self.alarma.labels(s).set(int(propia))
            if propia:   # cortes de menos de SILENCIO s no reinician la cuenta (igual que el aviso sostenido)
                if s not in self.desde or t - self.ult_on.get(s, -1e9) > SILENCIO:
                    self.desde[s] = t
                self.ult_on[s] = t
            self.hace.labels(s).set(t - self.desde[s] if propia else float("nan"))
            self.real.labels(s).set(int(e.en_bn))
            self.cpu_al.labels(s).set(int(bool(alarm_cpu.get(s))))
            for nombre, v in ((self.prob, probs.get(s)), (self.puntaje, puntajes.get(s)), (self.cpu, f.get("cpu")),
                              (self.p95, f.get("lat_p95_ms")), (self.cola, f.get("cola"))):
                nombre.labels(s).set(v if v is not None and v == v else float("nan"))
            if self.ultima.get(s) and self.ultima[s] != c:
                self.causa.remove(s, self.ultima[s])
            if c:
                self.causa.labels(s, c).set(1)
            self.ultima[s] = c


def iniciar_exportador(puerto):
    if not puerto:
        return None
    try:
        return Exportador(puerto)
    except OSError as e:
        print("No se pudo publicar en el puerto %d (%s): sigo sin Grafana" % (puerto, e))
        return None


def lead_seg(alarmas, o):
    """Lead honesto en segundos: encendido dentro de los 30 s previos al inicio o, tras >= SILENCIO s apagada;
    None si no hubo."""
    a = list(alarmas)
    for j in range(max(0, o - D.H), o):
        if a[j] and j - SILENCIO >= 0 and not any(a[j - SILENCIO:j]):
            return o - j
    return None


def lead(alarmas, o):
    """Lead honesto como texto."""
    s = lead_seg(alarmas, o)
    if s is not None:
        return "avisó %d s antes" % s
    if o > 0 and list(alarmas)[o - 1]:
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


def actualizar_real(e, cond):
    """Estado real de un servicio, con la misma regla que etiquetar.py: un cuello de botella exige SOSTENIDO
    segundos seguidos y termina tras SILENCIO segundos sin otro tramo valido (los picos mas cortos no cuentan).
    Devuelve "entra", "sale" o None."""
    e.racha = e.racha + 1 if cond else 0
    e.calma = 0 if e.racha >= SOSTENIDO else e.calma + 1
    if not e.en_bn and e.racha >= SOSTENIDO:
        e.en_bn = True
        return "entra"
    if e.en_bn and e.calma - e.racha >= SILENCIO:
        e.en_bn = False
        return "sale"
    return None


def corrida_en_curso():
    """(t0_locust, run_id) de la corrida de run_experiment.py en curso (t0 es el cero de las graficas).
    (None, None) si no hay ninguna, por ejemplo con solo trafico de fondo o con Locust lanzado a mano."""
    metas = glob.glob(os.path.join(HERE, "data", "*", "meta.json"))
    if not metas:
        return None, None
    try:
        meta = json.load(open(max(metas, key=os.path.getmtime)))
    except (OSError, ValueError):   # run_experiment.py lo esta escribiendo en este momento
        return None, None
    fin = meta.get("t0_locust", 0) + meta.get("duration", 0) + 30
    if "t0_locust" in meta and time.time() < fin:
        return meta["t0_locust"], meta.get("run_id", "")
    return None, None


def inicio_de_carga():
    return corrida_en_curso()[0]


# ------------------------------------------------------------------ pantalla
def barra(p, n=10):
    k = int(round(p * n))
    return "█" * k + "░" * (n - k)


def pintar(t_seg, snap, probs, alarm_v7, alarm_cpu, estados, eventos, umbral, calentando, espera=None,
           umbral_puntaje=None):
    """probs y umbral en la escala que se muestra (probabilidad calibrada si hay calibracion)."""
    rojo, amar, verde, gris, morado, fin = "\033[91m", "\033[93m", "\033[92m", "\033[90m", "\033[95m", "\033[0m"
    espera = espera or {}
    corte = ("umbral = %.2f (puntaje %.2f)" % (umbral, umbral_puntaje) if umbral_puntaje is not None
             else "umbral = %.2f" % umbral)
    L = ["\033[H\033[J",
         "DEMO %s  ·  ¿cuello de botella en los próximos 30 s?   t de carga = %d s   %s" % (NOMBRE, t_seg, corte),
         "",
         "%-15s %5s %8s %5s  %-18s %-17s %-9s %s" % ("servicio", "CPU", "p95(ms)", "cola", "P(cuello en 30 s)",
                                                    NOMBRE, "CPU>80%", "estado real")]
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
        if alarm_v7.get(s):
            v7 = rojo + "⚠ ALARMA" + fin + " " * 9
        elif espera.get(s):
            v7 = morado + ("↪ %s" % espera[s]).ljust(17) + fin
        else:
            v7 = verde + "ok" + fin + " " * 15
        cp = (rojo + "⚠ ALARMA" + fin) if alarm_cpu.get(s) else (verde + "ok" + fin + "      ")
        real = (rojo + "CUELLO DE BOTELLA" + fin) if estados[s].en_bn else (gris + "normal" + fin)
        num = lambda v, fmt: (fmt % v) if v == v else "  -"  # noqa: E731
        L.append("%-15s %5s %8s %5s  %s%s  %s  %s" % (s, num(cpu, "%4.0f%%"), num(p95, "%7.0f"),
                                                     num(cola, "%4.0f"), pr, v7, cp, real))
    if espera:
        L += ["", gris + "↪ <causa>: en alarma, pero espera a ese servicio; el aviso queda en la causa" + fin]
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
    cal = CAL.cargar(os.path.join(HERE, "data", a.resultados))
    m = pd.read_csv(os.path.join(HERE, "data", a.replay, "metrics.csv"))
    t0 = json.load(open(os.path.join(HERE, "data", a.replay, "meta.json")))["t0_locust"]
    m["sec"] = np.floor(m["timestamp"] - t0).astype(int)
    met = m.groupby(["sec", "servicio"])[["cpu", "lat_p95_ms", "cola"]].mean()
    p = pred.set_index(["t_rel", "servicio"])
    secs = range(int(max(0, pred["t_rel"].min() - 5)), int(pred["t_rel"].max()) + 1)
    estados = {s: Estado() for s in D.SERVICIOS}
    eventos = collections.deque(maxlen=50)
    exp = iniciar_exportador(a.exportar)
    print("Repeticion de %s con las predicciones de la validacion cruzada (%s)" % (a.replay, a.resultados))
    time.sleep(1.5)
    for k, t in enumerate(secs):
        snap, probs, puntajes, alarma, alarm_v7, alarm_cpu = {}, {}, {}, {}, {}, {}
        for s in D.SERVICIOS:
            row = met.loc[(t, s)] if (t, s) in met.index else None
            snap[s] = {"cpu": row["cpu"] if row is not None else np.nan,
                       "lat_p95_ms": row["lat_p95_ms"] if row is not None else np.nan,
                       "cola": row["cola"] if row is not None else np.nan}
            hay = (t, s) in p.index
            if hay:
                puntajes[s] = float(p.loc[(t, s), "prob_lstm"])
                probs[s] = float(CAL.aplicar(cal, puntajes[s]))
            alarma[s] = bool(hay and p.loc[(t, s), "alarma_lstm"])
        espera = paso_modelo(t, estados, alarma, probs, eventos)
        for s in D.SERVICIOS:
            e = estados[s]
            alarm_v7[s] = e.al_v7[-1]
            cpu = snap[s]["cpu"]
            alarm_cpu[s] = bool(cpu == cpu and cpu > 80.0)
            e.al_cpu.append(alarm_cpu[s])
            # estado real: bn_ahora segundo a segundo, con los cortes de menos de SILENCIO s fusionados como en
            # el etiquetado (si no, un episodio que parpadea "entra" y "se recupera" muchas veces)
            bn = bool((t, s) in p.index and p.loc[(t, s), "bn_ahora"])
            e.calma = 0 if bn else e.calma + 1
            if bn and not e.en_bn:
                e.en_bn = True
                o = len(e.al_v7) - 1
                eventos.append("[t=%3d s] %s ENTRA EN CUELLO DE BOTELLA → %s: %s | CPU>80%%: %s"
                               % (t, s.upper(), NOMBRE, lead_modelo(e, o), lead(e.al_cpu, o)))
                if exp:
                    exp.episodio_nuevo(s, t, lead_seg(e.al_v7, o), lead_modelo(e, o), lead(e.al_cpu, o))
            elif e.en_bn and e.calma >= SILENCIO:
                e.en_bn = False
                eventos.append("[t=%3d s] %s se recuperó" % (t - SILENCIO + 1, s))
        calentando = 0 if probs else 1
        if exp:
            exp.publicar(t, snap, probs, puntajes, estados, espera, alarm_cpu, float(CAL.aplicar(cal, umbral)),
                         calentando, eventos)
        if a.sin_pantalla:
            while eventos:
                print("  " + eventos.popleft())
        else:
            pintar(t, snap, {s: probs.get(s, 0.0) for s in D.SERVICIOS}, alarm_v7, alarm_cpu, estados, eventos,
                   float(CAL.aplicar(cal, umbral)), calentando, espera, umbral if cal else None)
        time.sleep(1.0 / a.velocidad)
    print("\nRepeticion terminada.")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=MODELO_DEF)
    ap.add_argument("--umbrales", default=os.path.join(HERE, "data", "umbrales_propio.json"))
    ap.add_argument("--umbral", type=float, default=None)
    ap.add_argument("--log", default="")
    ap.add_argument("--sin-pantalla", action="store_true")
    ap.add_argument("--duracion", type=float, default=0)
    ap.add_argument("--replay", default="")
    ap.add_argument("--velocidad", type=float, default=1.0)
    ap.add_argument("--resultados", default=RESULTADOS_DEF)
    ap.add_argument("--exportar", type=int, default=9108,
                    help="puerto donde publicar las metricas para Prometheus/Grafana (0 = no publicar)")
    a = ap.parse_args()

    os.system("")  # activa los colores ANSI en la consola de Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # flechas y barras tambien en consolas cp1252
    except Exception:
        pass
    global NOMBRE
    if a.replay:
        NOMBRE = nombre_de(a.resultados)
        return repeticion(a)
    NOMBRE = nombre_de(a.modelo)
    predecir, ck = cargar_modelo(a.modelo)
    global HIST, CALENTAR   # una ventana mas larga necesita mas historia (ventana + 15 s de tendencias)
    HIST = max(HIST, D.W + 20)
    if any(n in D.LARGO for n in ck.get("feats_usadas", [])):   # tendencias de 120 s + media de 10 s
        HIST = max(HIST, D.W + 120 + 10)
    CALENTAR = max(CALENTAR, D.W + 15)
    umbral = a.umbral if a.umbral is not None else ck["umbral"]
    umbrales = {s: v["umbral_ms"] for s, v in json.load(open(a.umbrales))["servicios"].items()}
    cal = CAL.cargar(a.modelo)
    umbral_cal = float(CAL.aplicar(cal, umbral))
    print("Modelo %s: %s  (umbral %.2f%s%s)" % (NOMBRE, a.modelo, umbral,
                                               ", ensamble de %d modelos" % ck["n_modelos"] if "n_modelos" in ck else "",
                                               ", probabilidad calibrada %.2f" % umbral_cal if cal else ""))

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
    exp = iniciar_exportador(a.exportar)
    log, log_f = None, None
    if a.log:
        log_f = open(a.log, "w", newline="")
        log = csv.writer(log_f)
        log.writerow(["t", "servicio", "cpu", "lat_p95_ms", "cola", "puntaje", "prob_calibrada", "alarma_modelo",
                      "alarma_propia", "espera_a", "alarma_cpu", "cuello_real"])
    t0 = time.time()
    k = 0
    con_trafico = 0
    sin_trafico = 0
    t0_carga = None
    corrida = None   # t0 de la corrida de run_experiment.py que se esta mostrando (None = sin corrida)
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
                if con_trafico and exp:   # termino una carga: la siguiente es una sesion nueva en el tablero
                    exp.nueva_sesion()
                con_trafico = 0
                t0_carga = None
                corrida = None
                snaps.clear()
            elif sin_trafico == 0:
                con_trafico += 1
            if con_trafico:
                # que corrida es (cada 2 s): con trafico de fondo (python entorno.py iniciar) no hay pausa entre
                # corridas, asi que el cambio se detecta por el meta.json de run_experiment.py. Abre una sesion
                # nueva en el tablero y reinicia el reloj, pero NO borra la historia: el modelo no vuelve a calentar.
                nueva, run_id = corrida_en_curso() if (t0_carga is None or k % 2 == 0) else (corrida, None)
                if t0_carga is None:
                    corrida, t0_carga = nueva, nueva or time.time() - con_trafico
                elif nueva != corrida:
                    corrida, t0_carga = nueva, nueva or time.time()
                    if exp:
                        exp.nueva_sesion()
                    eventos.append("[t=  0 s] ── %s ──" % ("corrida nueva: %s" % run_id if nueva
                                                         else "terminó la corrida: tráfico de fondo"))
            snaps.append(snap)
            # segundos desde el inicio de la carga, el reloj de las graficas (sin meta.json, desde que se vio trafico)
            t_carga = int(time.time() - t0_carga) if t0_carga else con_trafico
            calentando = max(0, CALENTAR - con_trafico)
            puntajes, probs, alarm_v7, alarm_cpu = {}, {}, {}, {}
            if not calentando:
                i0 = len(snaps) - con_trafico   # primer segundo con trafico (negativo si quedo fuera de la historia)
                X = np.stack([entradas(snaps, s, i0) for s in D.SERVICIOS])
                for s, p in zip(D.SERVICIOS, predecir(X)):
                    puntajes[s] = float(p)
                    probs[s] = float(CAL.aplicar(cal, float(p)))
            alarma = {s: puntajes.get(s, 0.0) > umbral for s in D.SERVICIOS}
            espera = paso_modelo(t_carga, estados, alarma, probs, eventos)
            for s in D.SERVICIOS:
                e = estados[s]
                alarm_v7[s] = e.al_v7[-1]
                cpu = snap[s].get("cpu", np.nan)
                alarm_cpu[s] = bool(cpu == cpu and cpu > 80.0)
                e.al_cpu.append(alarm_cpu[s])
                cambio = actualizar_real(e, t_carga >= 10 and condicion(snap[s], s, umbrales))
                if cambio == "entra":
                    o = len(e.al_v7) - SOSTENIDO   # el episodio empezo hace SOSTENIDO - 1 s
                    eventos.append("[t=%3d s] %s ENTRA EN CUELLO DE BOTELLA → %s: %s | CPU>80%%: %s"
                                   % (t_carga - SOSTENIDO + 1, s.upper(), NOMBRE, lead_modelo(e, o), lead(e.al_cpu, o)))
                    if exp:
                        exp.episodio_nuevo(s, t_carga - SOSTENIDO + 1, lead_seg(e.al_v7, o), lead_modelo(e, o),
                                           lead(e.al_cpu, o))
                elif cambio == "sale":
                    eventos.append("[t=%3d s] %s se recuperó" % (t_carga - SILENCIO + 1, s))
                if log:
                    log.writerow([round(time.time() - t0), s, cpu, snap[s].get("lat_p95_ms"), snap[s].get("cola"),
                                  puntajes.get(s, ""), probs.get(s, ""), int(alarma[s]), int(alarm_v7[s]),
                                  espera.get(s, ""), int(alarm_cpu[s]), int(e.en_bn)])
            if exp:
                exp.publicar(t_carga, snap, probs, puntajes, estados, espera, alarm_cpu, umbral_cal, calentando,
                             eventos)
            if log_f:
                log_f.flush()   # se sale con os._exit: sin esto se perderian los ultimos segundos
            if a.sin_pantalla:
                linea = " | ".join("%s %s" % (s[:5], ("%.2f" % probs[s]) if s in probs else "--")
                                   for s in D.SERVICIOS)
                print("t=%3d  %s" % (t_carga, linea))
                while eventos:
                    print("  " + eventos.popleft())
            else:
                pintar(t_carga, snap, probs, alarm_v7, alarm_cpu, estados, eventos, umbral_cal, calentando, espera,
                       umbral if cal else None)
    except KeyboardInterrupt:
        pass
    stop.set()
    if log_f:
        log_f.close()
    print("\nDemo terminada.")
    sys.stdout.flush()
    os._exit(0)   # los hilos de docker stats son bloqueantes


if __name__ == "__main__":
    main()
