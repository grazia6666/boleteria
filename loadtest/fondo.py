"""
Trafico de fondo: una carga baja y constante (por defecto 15 usuarios, la base del escenario normal) que queda
corriendo entre corridas. Asi la demo (python ml/demo.py) calienta UNA sola vez y Grafana siempre tiene datos:
un sistema real nunca tiene cero trafico.

  python loadtest/fondo.py iniciar [--usuarios 15]
  python loadtest/fondo.py parar
  python loadtest/fondo.py estado
(o todo junto con python entorno.py iniciar / parar)

run_experiment.py lo detecta solo: al arrancar el escenario corta el fondo (cuando el escenario ya esta
mandando pedidos, para que la demo no vea un hueco) y lo vuelve a encender unos segundos antes de que termine.
El escenario es exactamente el mismo de siempre. run_campaign.py lo apaga: las corridas para entrenar se graban
desde reposo, como las 175 del modelo.

Es un Locust sin pantalla en segundo plano (sobrevive a cerrar la terminal). Estado en data/fondo.json,
salida en data/fondo.log.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import psutil

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ESTADO = os.path.join(HERE, "data", "fondo.json")
LOG = os.path.join(HERE, "data", "fondo.log")
USUARIOS = 15
RELEVO_S = 5   # al final del escenario, el fondo se enciende RELEVO_S s antes: Locust tarda ~3 s en mandar pedidos


def _leer():
    try:
        return json.load(open(ESTADO))
    except (OSError, ValueError):
        return None


def activo():
    """Estado del fondo ({pid, usuarios, desde}) si esta corriendo; None si no."""
    st = _leer()
    if not st:
        return None
    try:
        p = psutil.Process(st["pid"])
        # el pid puede haberse reusado: tiene que ser un Locust con el locustfile del proyecto
        if p.is_running() and p.status() != psutil.STATUS_ZOMBIE and "locust" in " ".join(p.cmdline()):
            return st
    except (psutil.Error, KeyError):
        pass
    return None


def iniciar(usuarios=None, silencioso=False):
    st = activo()
    if st:
        if not silencioso:
            print("[fondo] ya estaba corriendo (%d usuarios, pid %d)" % (st["usuarios"], st["pid"]))
        return st
    usuarios = int(usuarios or USUARIOS)
    env = {**os.environ, "SCENARIO": "normal", "DURATION": str(10 ** 9), "BASE_USERS": str(usuarios)}
    cmd = [sys.executable, "-m", "locust", "-f", os.path.join(HERE, "loadtest", "locustfile.py"),
           "--headless", "--host", "http://localhost:8003", "--only-summary"]
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    salida = open(LOG, "a")
    salida.write("\n== fondo iniciado %s (%d usuarios) ==\n" % (time.strftime("%d-%m %H:%M:%S"), usuarios))
    salida.flush()
    kw = dict(env=env, cwd=HERE, stdout=salida, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if os.name == "nt":
        base = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        try:   # que no lo mate cerrar la terminal de VS Code (job object)
            p = subprocess.Popen(cmd, creationflags=base | 0x01000000, **kw)   # CREATE_BREAKAWAY_FROM_JOB
        except OSError:
            p = subprocess.Popen(cmd, creationflags=base, **kw)
    else:
        p = subprocess.Popen(cmd, start_new_session=True, **kw)
    st = {"pid": p.pid, "usuarios": usuarios, "desde": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(st, open(ESTADO, "w"))
    if not silencioso:
        print("[fondo] trafico de fondo encendido: %d usuarios (pid %d)" % (usuarios, p.pid))
    return st


def parar(silencioso=False):
    """Apaga el fondo. Devuelve su estado (para volver a encenderlo igual) o None si no corria."""
    st = activo()
    if st:
        try:
            p = psutil.Process(st["pid"])
            for h in p.children(recursive=True):
                h.terminate()
            p.terminate()
            p.wait(timeout=5)
        except psutil.TimeoutExpired:
            p.kill()
        except psutil.Error:
            pass
        if not silencioso:
            print("[fondo] trafico de fondo apagado")
    elif not silencioso:
        print("[fondo] no estaba corriendo")
    try:
        os.remove(ESTADO)
    except OSError:
        pass
    return st


def main():
    ap = argparse.ArgumentParser(description="Trafico de fondo para la demo y Grafana")
    ap.add_argument("accion", choices=["iniciar", "parar", "estado"])
    ap.add_argument("--usuarios", type=int, default=USUARIOS)
    a = ap.parse_args()
    if a.accion == "iniciar":
        iniciar(a.usuarios)
    elif a.accion == "parar":
        parar()
    else:
        st = activo()
        print("[fondo] %s" % ("encendido: %d usuarios desde %s (pid %d)" % (st["usuarios"], st["desde"], st["pid"])
                              if st else "apagado"))


if __name__ == "__main__":
    main()
