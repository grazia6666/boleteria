"""
Deja todo encendido para probar escenarios sin esperar cada vez: los 6 servicios, Grafana/Prometheus y un
trafico de fondo constante (15 usuarios, la base del escenario normal). La demo calienta UNA sola vez y
despues cada python run_experiment.py ... se ve al instante, como una sesion nueva en Grafana.

  python entorno.py iniciar                 servicios + Grafana + trafico de fondo
  python entorno.py iniciar --sin-grafana   sin Prometheus/Grafana
  python entorno.py iniciar --usuarios 10   otro nivel de trafico de fondo
  python entorno.py estado                  que esta encendido
  python entorno.py parar                   apaga el trafico de fondo (servicios y Grafana siguen)
  python entorno.py parar --todo            apaga tambien Grafana y los servicios

Despues, en otra terminal: python ml/demo.py   (calienta 60 s la primera vez; dejarla abierta)
y los escenarios de siempre, p. ej.:
  python run_experiment.py --scenario progresivo --duration 360 --peak-users 60 --run-id prueba1 --inject pago:120:120:40:90

No usar durante las campanas (run_campaign.py apaga el fondo solo): las corridas para entrenar se graban
desde reposo, como las 175 del modelo.
"""
import argparse
import os
import subprocess
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "loadtest"))
import fondo  # noqa: E402

SERVICIOS = {"usuarios": 8001, "eventos": 8002, "reservaciones": 8003, "pago": 8004, "ticket": 8005,
             "notificaciones": 8006}
MONITOREO = os.path.join(HERE, "monitoreo", "docker-compose.yml")


def compose(*args, archivo=None):
    cmd = ["docker", "compose"] + (["-f", archivo] if archivo else []) + list(args)
    return subprocess.run(cmd, cwd=HERE).returncode == 0


def responde(url, timeout=1.0):
    try:
        return requests.get(url, timeout=timeout).ok
    except requests.RequestException:
        return False


def servicios_arriba():
    return {s: responde("http://localhost:%d/health" % p) for s, p in SERVICIOS.items()}


def iniciar(a):
    print("1) servicios (docker compose up -d)")
    if not compose("up", "-d"):
        sys.exit("No se pudo levantar los servicios. ¿Docker Desktop esta abierto?")
    for _ in range(60):
        if all(servicios_arriba().values()):
            break
        time.sleep(1)
    caidos = [s for s, ok in servicios_arriba().items() if not ok]
    if caidos:
        sys.exit("Estos servicios no responden: %s" % ", ".join(caidos))
    print("   los 6 servicios responden")
    if not a.sin_grafana:
        print("2) Prometheus y Grafana")
        if not compose("up", "-d", archivo=MONITOREO):
            print("   (no se pudo levantar el monitoreo; sigo sin Grafana)")
    print("3) trafico de fondo")
    fondo.iniciar(a.usuarios)
    print("\nListo. Ahora, en otra terminal:  python ml/demo.py   (calienta 60 s una sola vez; dejala abierta)")
    if not a.sin_grafana:
        print("Grafana: http://localhost:3000   (necesita la demo abierta para las predicciones)")
    print("Despues corre tus escenarios con python run_experiment.py ... como siempre.")
    print("Para apagar el trafico de fondo: python entorno.py parar   (todo: python entorno.py parar --todo)")


def estado(_a=None):
    arriba = servicios_arriba()
    print("Servicios: %s" % ("los 6 responden" if all(arriba.values()) else
                             "no responden: " + ", ".join(s for s, ok in arriba.items() if not ok)))
    print("Grafana:   %s" % ("encendido (http://localhost:3000)" if responde("http://localhost:3000/api/health")
                             else "apagado"))
    st = fondo.activo()
    print("Fondo:     %s" % ("encendido: %d usuarios desde %s" % (st["usuarios"], st["desde"]) if st else "apagado"))
    print("Demo:      %s" % ("exportando en :9108" if responde("http://localhost:9108/metrics")
                             else "no esta abierta (python ml/demo.py)"))


def parar(a):
    fondo.parar()
    if a.todo:
        print("Apagando Grafana y Prometheus")
        compose("stop", archivo=MONITOREO)
        print("Apagando los servicios")
        compose("stop")


def main():
    ap = argparse.ArgumentParser(description="Encender, revisar o apagar el entorno de pruebas")
    ap.add_argument("accion", choices=["iniciar", "estado", "parar"])
    ap.add_argument("--usuarios", type=int, default=fondo.USUARIOS, help="usuarios del trafico de fondo")
    ap.add_argument("--sin-grafana", action="store_true")
    ap.add_argument("--todo", action="store_true", help="con parar: tambien Grafana y los servicios")
    a = ap.parse_args()
    {"iniciar": iniciar, "estado": estado, "parar": parar}[a.accion](a)


if __name__ == "__main__":
    main()
