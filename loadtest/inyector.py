"""
Inyecta cuellos de botella via /admin/load y registra los tiempos REALES en events.csv.

Uso:
  python loadtest/inyector.py --run-id r1 --escenario burst \
      --inject pago:120:90:150 --inject ticket:360:60:150:30 --out data/r1/events.csv
  formato de --inject:  servicio:inicio_s:duracion_s:cpu_load[:rampa_s]
    - inicio_s   relativo a --t0
    - duracion_s total de la inyeccion (INCLUYE la rampa)
    - rampa_s    (opcional) segundos para subir cpu_load de 0 al valor final, 1 paso por segundo.
                 Sin rampa = escalon abrupto.
  python loadtest/inyector.py --reset     # deja todos en cpu_load=0
"""
import argparse
import csv
import os
import time

import requests

PUERTOS = {"usuarios": 8001, "eventos": 8002, "reservaciones": 8003,
           "pago": 8004, "ticket": 8005, "notificaciones": 8006}


def set_load(servicio, cpu_load, intentos=1):
    """Fija cpu_load. Un servicio saturado puede tardar en responder: se reintenta hasta `intentos` veces y,
    si no responde, se avisa y se sigue (un paso perdido de una rampa lo corrige el paso siguiente)."""
    for k in range(intentos):
        try:
            r = requests.post(f"http://localhost:{PUERTOS[servicio]}/admin/load",
                              json={"cpu_load": cpu_load}, timeout=5)
            r.raise_for_status()
            return True
        except Exception as e:
            error = e
            if k + 1 < intentos:
                time.sleep(1)
    print(f"[inyector] {servicio} no respondio a /admin/load (cpu_load={cpu_load}): {error}")
    return False


def reset_all():
    for s in PUERTOS:
        set_load(s, 0, intentos=3)


def parse_inject(spec):
    p = spec.split(":")
    if len(p) not in (4, 5):
        raise SystemExit(f"--inject invalido: {spec} (usa servicio:inicio:duracion:cpu_load[:rampa])")
    servicio, ini, dur, load = p[0], float(p[1]), float(p[2]), int(p[3])
    rampa = float(p[4]) if len(p) == 5 else 0.0
    if servicio not in PUERTOS:
        raise SystemExit(f"Servicio desconocido: {servicio}")
    if rampa > dur:
        raise SystemExit(f"La rampa ({rampa}s) no puede ser mayor que la duracion ({dur}s): {spec}")
    return servicio, ini, dur, load, rampa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="run")
    ap.add_argument("--escenario", default="")
    ap.add_argument("--inject", action="append", default=[])
    ap.add_argument("--t0", type=float, default=0, help="epoch de referencia (default: ahora)")
    ap.add_argument("--out", default="events.csv")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    if args.reset:
        reset_all()
        print("Todos los servicios en cpu_load=0")
        return

    plan = sorted((parse_inject(s) for s in args.inject), key=lambda p: p[1])  # sin solapar

    t0 = args.t0 or time.time()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    nuevo = not os.path.exists(args.out)
    try:
        with open(args.out, "a", newline="") as f:
            w = csv.writer(f)
            if nuevo:
                w.writerow(["run_id", "escenario", "servicio", "t_inicio", "t_fin", "cpu_load", "rampa_s"])
            for servicio, ini, dur, load, rampa in plan:
                time.sleep(max(0.0, t0 + ini - time.time()))
                t_ini = time.time()
                if rampa > 0:
                    print(f"[inyector] {servicio} rampa 0 -> {load} en {rampa:.0f}s")
                    for k in range(1, int(rampa) + 1):
                        set_load(servicio, max(1, round(load * k / rampa)))
                        time.sleep(max(0.0, t_ini + k - time.time()))
                else:
                    set_load(servicio, load, intentos=5)
                    print(f"[inyector] {servicio} cpu_load={load} ON (escalon)")
                time.sleep(max(0.0, t_ini + dur - time.time()))
                set_load(servicio, 0, intentos=15)
                t_fin = time.time()
                print(f"[inyector] {servicio} OFF")
                w.writerow([args.run_id, args.escenario, servicio, f"{t_ini:.3f}", f"{t_fin:.3f}", load, int(rampa)])
                f.flush()
    finally:
        reset_all()


if __name__ == "__main__":
    main()