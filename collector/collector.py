"""
Recolector de metricas: una fila por servicio y por segundo.   [COLLECTOR_V2]

Cada servicio se muestrea en su PROPIO hilo, asi un servicio saturado que no responde no retrasa
a los demas. Si el ciclo se atrasa, se saltan ticks (no hay filas "de puesta al dia" con intervalos
minusculos). El timeout del scrape es 0.9 s (< intervalo): un servicio que no responde a /metrics
en 0.9 s queda con scrape_ok=0 y el muestreo se mantiene a 1 Hz.

Columnas de metrics.csv:
  timestamp   epoch (s) en que RESPONDIO /metrics (o del intento fallido)
  servicio
  cpu         % de CPU respecto al LIMITE del contenedor (100 = usa todo su cpus)
  ram_mb      memoria en MB
  rps         requests/s entre dos scrapes consecutivos EXITOSOS
  lat_mean_ms latencia media en ese intervalo
  lat_p95_ms  latencia p95 estimada (desde buckets del histograma)
  cola        requests en curso (in-flight) en ese instante
  err_rps     respuestas 5xx por segundo
  scrape_ok   1 si /metrics respondio, 0 si no

edges.csv (junto a metrics.csv), una fila por DEPENDENCIA y muestra:
  timestamp, origen, destino, cps (llamadas/s), lat_mean_ms, lat_p95_ms, err_ps
  Latencia medida desde el servicio que llama (incluye red + espera al destino).

Las filas de distintos servicios se intercalan en el archivo (cada hilo escribe cuando termina);
quien las lea debe ordenar por servicio y timestamp.
"""
import argparse
import csv
import math
import os
import threading
import time

import requests

SERVICIOS = {
    "usuarios": 8001, "eventos": 8002, "reservaciones": 8003,
    "pago": 8004, "ticket": 8005, "notificaciones": 8006,
}
CPU_LIMIT = 1.0        # debe coincidir con `cpus` del docker-compose.yml
SCRAPE_TIMEOUT = 0.9   # segundos; menor que el intervalo de muestreo

latest = {}  # servicio -> {"cpu":..., "ram_mb":...}  (lo actualizan hilos en segundo plano)


def stats_worker(client, name):
    """Lee docker stats en streaming (~1 muestra/s) y guarda la ultima."""
    try:
        c = client.containers.get(name)
        for s in c.stats(stream=True, decode=True):
            try:
                cs, pcs = s["cpu_stats"], s["precpu_stats"]
                cpu_d = cs["cpu_usage"]["total_usage"] - pcs["cpu_usage"]["total_usage"]
                sys_d = cs["system_cpu_usage"] - pcs["system_cpu_usage"]
                ncpu = cs.get("online_cpus") or 1
                cores = (cpu_d / sys_d) * ncpu if sys_d > 0 else 0.0
                mem = s["memory_stats"]
                st = mem.get("stats", {})
                usage = mem["usage"] - st.get("inactive_file", st.get("cache", 0))
                latest[name] = {"cpu": cores / CPU_LIMIT * 100, "ram_mb": usage / 2**20}
            except (KeyError, ZeroDivisionError):
                continue
    except Exception as e:
        print(f"[stats:{name}] {e}")


def scrape(port):
    r = requests.get(f"http://localhost:{port}/metrics", timeout=SCRAPE_TIMEOUT)
    t_resp = time.time()   # instante en que se leyeron los contadores
    r.raise_for_status()
    out = {"t": t_resp, "total": 0.0, "errors": 0.0, "sum": 0.0, "count": 0.0, "buckets": {},
           "inflight": 0.0, "edges": {}}

    def edge(t):
        return out["edges"].setdefault(t, {"total": 0.0, "errors": 0.0, "sum": 0.0,
                                           "count": 0.0, "buckets": {}})
    from prometheus_client.parser import text_string_to_metric_families
    for fam in text_string_to_metric_families(r.text):
        for s in fam.samples:
            n = s.name
            if n == "http_requests_total":
                out["total"] += s.value
                if s.labels.get("status", "").startswith("5"):
                    out["errors"] += s.value
            elif n == "http_request_duration_seconds_sum":
                out["sum"] += s.value
            elif n == "http_request_duration_seconds_count":
                out["count"] += s.value
            elif n == "http_request_duration_seconds_bucket":
                le = float(s.labels["le"])
                out["buckets"][le] = out["buckets"].get(le, 0.0) + s.value
            elif n == "http_requests_in_flight":
                out["inflight"] = s.value
            elif n == "downstream_requests_total":
                e = edge(s.labels["target"])
                e["total"] += s.value
                st = s.labels.get("status", "")
                if st.startswith("5") or st == "error":
                    e["errors"] += s.value
            elif n == "downstream_request_duration_seconds_sum":
                edge(s.labels["target"])["sum"] += s.value
            elif n == "downstream_request_duration_seconds_count":
                edge(s.labels["target"])["count"] += s.value
            elif n == "downstream_request_duration_seconds_bucket":
                b = edge(s.labels["target"])["buckets"]
                le = float(s.labels["le"])
                b[le] = b.get(le, 0.0) + s.value
    return out


def quantile(buckets, q):
    """Cuantil estimado con interpolacion lineal sobre buckets acumulados."""
    les = sorted(buckets)
    if not les or buckets[les[-1]] <= 0:
        return 0.0
    target = q * buckets[les[-1]]
    prev_le, prev_c = 0.0, 0.0
    for le in les:
        c = buckets[le]
        if c >= target:
            if math.isinf(le):
                return prev_le
            if c == prev_c:
                return le
            return prev_le + (le - prev_le) * (target - prev_c) / (c - prev_c)
        prev_le, prev_c = le, c
    return prev_le


def delta_metrics(cur, prev, dt):
    """Convierte contadores acumulados en tasas del ultimo intervalo."""
    d_total = cur["total"] - prev["total"]
    d_count = cur["count"] - prev["count"]
    if d_total < 0 or d_count < 0 or dt <= 0:  # el contenedor se reinicio
        return None
    d_buckets = {le: cur["buckets"].get(le, 0) - prev["buckets"].get(le, 0) for le in cur["buckets"]}
    return {
        "rps": d_total / dt,
        "lat_mean_ms": (cur["sum"] - prev["sum"]) / d_count * 1000 if d_count > 0 else 0.0,
        "lat_p95_ms": quantile(d_buckets, 0.95) * 1000,
        "err_rps": (cur["errors"] - prev["errors"]) / dt,
    }


def sampler(name, port, args, t0, stop, lock, w, f, ew, fe):
    """Hilo de un servicio: una muestra por tick, sin que otros servicios lo retrasen."""
    prev = None
    k = 0
    while not stop.is_set():
        k += 1
        target = t0 + k * args.interval
        if time.time() > target + args.interval:      # atrasado: saltar ticks, no "ponerse al dia"
            k = int((time.time() - t0) / args.interval) + 1
            target = t0 + k * args.interval
        if stop.wait(max(0.0, target - time.time())):
            break
        if args.duration and time.time() - t0 > args.duration:
            break
        st = latest.get(name, {})
        row = {"cpu": st.get("cpu", ""), "ram_mb": st.get("ram_mb", ""), "rps": "",
               "lat_mean_ms": "", "lat_p95_ms": "", "cola": "", "err_rps": "", "scrape_ok": 0}
        edge_rows = []
        ts = time.time()
        try:
            cur = scrape(port)
            ts = cur["t"]
            row["cola"] = cur["inflight"]
            row["scrape_ok"] = 1
            if prev is not None:
                dt = cur["t"] - prev["t"]
                d = delta_metrics(cur, prev, dt)
                if d:
                    row.update(d)
                for tgt, ecur in cur["edges"].items():
                    eprev = prev["edges"].get(tgt)
                    ed = delta_metrics(ecur, eprev, dt) if eprev else None
                    if ed:
                        edge_rows.append([f"{ts:.3f}", name, tgt, ed["rps"], ed["lat_mean_ms"],
                                          ed["lat_p95_ms"], ed["err_rps"]])
            prev = cur   # solo se actualiza con scrapes exitosos: el intervalo siguiente cubre el hueco
        except Exception:
            pass
        with lock:
            w.writerow([f"{ts:.3f}", name, row["cpu"], row["ram_mb"], row["rps"], row["lat_mean_ms"],
                        row["lat_p95_ms"], row["cola"], row["err_rps"], row["scrape_ok"]])
            f.flush()
            for r in edge_rows:
                ew.writerow(r)
            if edge_rows:
                fe.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="ruta del metrics.csv")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=0, help="segundos (0 = hasta Ctrl+C)")
    ap.add_argument("--sin-docker", action="store_true", help="no leer CPU/RAM (solo para pruebas)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if not args.sin_docker:
        import docker
        client = docker.from_env()
        for n in SERVICIOS:
            threading.Thread(target=stats_worker, args=(client, n), daemon=True).start()
        time.sleep(2)

    edges_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "edges.csv")
    stop, lock = threading.Event(), threading.Lock()
    print(f"Recolectando en {args.out} (Ctrl+C para parar)")
    with open(args.out, "w", newline="") as f, open(edges_path, "w", newline="") as fe:
        w, ew = csv.writer(f), csv.writer(fe)
        w.writerow(["timestamp", "servicio", "cpu", "ram_mb", "rps", "lat_mean_ms",
                    "lat_p95_ms", "cola", "err_rps", "scrape_ok"])
        ew.writerow(["timestamp", "origen", "destino", "cps", "lat_mean_ms", "lat_p95_ms", "err_ps"])
        t0 = time.time()
        hilos = [threading.Thread(target=sampler, args=(n, p, args, t0, stop, lock, w, f, ew, fe))
                 for n, p in SERVICIOS.items()]
        for h in hilos:
            h.start()
        try:
            while any(h.is_alive() for h in hilos):
                time.sleep(0.3)
        except KeyboardInterrupt:
            stop.set()
            for h in hilos:
                h.join(timeout=3)
    print("Recoleccion terminada")
    os._exit(0)  # los hilos de docker stats son bloqueantes


if __name__ == "__main__":
    main()