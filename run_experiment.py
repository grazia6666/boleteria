"""
Corre UN experimento completo: collector + locust + inyector, con el mismo run_id.

Ejemplos:
  python run_experiment.py --scenario normal --duration 300                      # sin bottleneck
  python run_experiment.py --scenario burst --duration 600 \
      --inject pago:120:90:150 --inject pago:360:60:150
Resultado en data/<run_id>/: metrics.csv, events.csv, locust_stats.csv, meta.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=["normal", "progresivo", "burst"])
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--inject", action="append", default=[], help="servicio:inicio_s:duracion_s:cpu_load")
    ap.add_argument("--base-users", type=int, default=15)
    ap.add_argument("--peak-users", type=int, default=100)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--burst-times", default="", help="burst: inicios de las rafagas en s, ej. 70,155,290")
    ap.add_argument("--burst-len", type=int, default=25, help="burst: duracion de cada rafaga en s")
    args = ap.parse_args()

    destinos = "-".join(sorted({i.split(":")[0] for i in args.inject})) or "sin-inyeccion"
    run_id = args.run_id or f"{args.scenario}_{destinos}_{datetime.now():%Y%m%d_%H%M%S}"
    out = os.path.join(HERE, "data", run_id)
    os.makedirs(out, exist_ok=True)
    # el inyector agrega filas a events.csv: si la carpeta ya existia (run_id repetido o reintento de la
    # campana), las fallas de la corrida anterior se mezclarian con las de esta
    if os.path.exists(os.path.join(out, "events.csv")):
        os.remove(os.path.join(out, "events.csv"))
    print(f"== Run {run_id} ==")

    subprocess.run([PY, os.path.join(HERE, "loadtest", "inyector.py"), "--reset"], check=True)
    time.sleep(2)

    procs = []
    try:
        collector = subprocess.Popen([PY, os.path.join(HERE, "collector", "collector.py"),
                                      "--out", os.path.join(out, "metrics.csv"),
                                      "--duration", str(args.duration + 25)])
        procs.append(collector)
        time.sleep(5)  # unos segundos de linea base antes de la carga

        t0 = time.time()
        env = {**os.environ, "SCENARIO": args.scenario, "DURATION": str(args.duration),
               "BASE_USERS": str(args.base_users), "PEAK_USERS": str(args.peak_users),
               "BURST_TIMES": args.burst_times, "BURST_LEN": str(args.burst_len)}
        locust = subprocess.Popen(
            [PY, "-m", "locust", "-f", os.path.join(HERE, "loadtest", "locustfile.py"),
             "--headless", "--host", "http://localhost:8003",
             "--csv", os.path.join(out, "locust"), "--only-summary"], env=env)
        procs.append(locust)

        cmd = [PY, os.path.join(HERE, "loadtest", "inyector.py"), "--run-id", run_id,
               "--escenario", args.scenario, "--t0", str(t0),
               "--out", os.path.join(out, "events.csv")]
        for i in args.inject:
            cmd += ["--inject", i]
        inyector = subprocess.Popen(cmd)
        procs.append(inyector)

        json.dump({"run_id": run_id, "scenario": args.scenario, "duration": args.duration,
                   "inject": args.inject, "base_users": args.base_users,
                   "peak_users": args.peak_users, "t0_locust": t0,
                   "burst_times": args.burst_times, "burst_len": args.burst_len},
                  open(os.path.join(out, "meta.json"), "w"), indent=2)

        locust.wait()
        inyector.wait()
        time.sleep(15)  # cola residual: se sigue grabando mientras el sistema se recupera
        collector.wait(timeout=30)
    except KeyboardInterrupt:
        print("Interrumpido")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        subprocess.run([PY, os.path.join(HERE, "loadtest", "inyector.py"), "--reset"])
    print(f"Listo -> {out}")


if __name__ == "__main__":
    main()