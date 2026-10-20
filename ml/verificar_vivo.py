"""
Verificacion en vivo del modelo final por escenario de carga (normal / progresivo / burst).

Corre corridas NUEVAS (que el modelo nunca vio) con la demo en vivo grabando lo que muestra, las
etiqueta con las mismas reglas del entrenamiento y mide v7 y la regla CPU > 80 % con el mismo protocolo
de ml/por_escenario.py, para comparar con las cifras de la validacion cruzada.

Uso (desde la raiz, con Docker y los servicios arriba; NO entrenar nada mientras corre):
  python ml/verificar_vivo.py correr     # 6 corridas, ~45 min; salta las que ya tienen OK
  python ml/verificar_vivo.py evaluar    # consolida, etiqueta, predice y compara (modelo v7)
  python ml/verificar_vivo.py evaluar --modelo data/resultados_v8/modelo_ensamble.json --resultados resultados_v8

Todo queda en data/vivo/ (separado del dataset de entrenamiento). Se copian ahi las corridas normal_base
para que los umbrales de latencia salgan identicos a los del entrenamiento.
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "ml"))
sys.path.insert(0, os.path.join(HERE, "collector"))
sys.path.insert(0, HERE)
import datos as D  # noqa: E402

PY = sys.executable
VIVO = os.path.join(HERE, "data", "vivo")
MODELO = os.path.join(HERE, "data", "resultados_v7", "models", "lstm_final.pt")
# (run_id, escenario, duracion, inyecciones, usuarios pico) -- mismos parametros que la campana
PLAN = [
    ("vivo_normal_inj_ticket", "normal", 360, ["ticket:60:90:150:40", "ticket:220:60:150"], 100),
    ("vivo_normal_inj_eventos", "normal", 360, ["eventos:60:90:150:40", "eventos:220:60:150"], 100),
    ("vivo_progresivo_r1", "progresivo", 360, [], 60),
    ("vivo_progresivo_r2", "progresivo", 360, [], 60),
    ("vivo_burst_inj_pago", "burst", 360, ["pago:95:75:150:30", "pago:215:75:150"], 70),
    ("vivo_burst_inj_notificaciones", "burst", 360,
     ["notificaciones:95:75:150:30", "notificaciones:215:75:150"], 70),
]


def escenario(run):
    for p, e in (("vivo_normal", "normal"), ("vivo_progresivo", "progresivo"), ("vivo_burst", "burst")):
        if run.startswith(p):
            return e
    return "otro"


def correr():
    import run_campaign as RC
    RC.preflight()
    os.makedirs(VIVO, exist_ok=True)
    for nb in ("normal_base_r1", "normal_base_r2", "normal_base_r3", "normal_base_r4"):
        if os.path.exists(os.path.join(HERE, "data", nb, "OK")) and not os.path.exists(os.path.join(VIVO, nb)):
            shutil.copytree(os.path.join(HERE, "data", nb), os.path.join(VIVO, nb))
    ref = RC.referencia_costo()
    for i, (rid, esc, dur, inj, pico) in enumerate(PLAN, 1):
        dest = os.path.join(VIVO, rid)
        if os.path.exists(os.path.join(dest, "OK")):
            print("ya existe %s, se salta" % rid)
            continue
        for d in (dest, os.path.join(HERE, "data", rid)):
            if os.path.exists(d):
                shutil.rmtree(d)
        print("\n===== [%d/%d] %s (%s) =====" % (i, len(PLAN), rid, esc), flush=True)
        RC.restart_stack()
        log_demo = os.path.join(VIVO, rid + "_demo_log.csv")
        out_demo = open(os.path.join(VIVO, rid + "_demo.txt"), "w", encoding="utf-8")
        demo = subprocess.Popen([PY, "-u", os.path.join(HERE, "ml", "demo.py"), "--sin-pantalla",
                                 "--duracion", str(dur + 40), "--log", log_demo],
                                cwd=HERE, stdout=out_demo, stderr=subprocess.STDOUT)
        time.sleep(4)
        cmd = [PY, os.path.join(HERE, "run_experiment.py"), "--scenario", esc, "--duration", str(dur),
               "--run-id", rid, "--peak-users", str(pico), "--base-users", "15"]
        for x in inj:
            cmd += ["--inject", x]
        rc = subprocess.run(cmd, cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
        try:
            demo.wait(timeout=120)
        except subprocess.TimeoutExpired:
            demo.kill()
        out_demo.close()
        if rc != 0 or not os.path.exists(os.path.join(HERE, "data", rid, "metrics.csv")):
            print("!! %s no termino bien (codigo %s)" % (rid, rc))
            continue
        shutil.move(os.path.join(HERE, "data", rid), dest)
        for f in (rid + "_demo_log.csv", rid + "_demo.txt"):
            shutil.move(os.path.join(VIVO, f), os.path.join(dest, f.replace(rid + "_", "")))
        costo = RC.cargar_corrida(dest)[2]["cpu_por_req_pre"]
        q = costo / ref if ref else float("nan")
        print("  control de calidad: %.2f x normal_base %s" % (q, "(OK)" if q <= RC.FACTOR_CONTAMINADA
                                                              else "!! CONTAMINADA, repetir"))
        if q <= RC.FACTOR_CONTAMINADA:
            open(os.path.join(dest, "OK"), "w").write("ok")
        eventos = [l.rstrip() for l in open(os.path.join(dest, "demo.txt"), encoding="utf-8") if l.startswith("  [")]
        print("  la demo en vivo mostro:")
        for e in eventos:
            print("   " + e.strip())
        time.sleep(20)


def predecir_todo(ds, ruta=None):
    import demo
    predecir, ck = demo.cargar_modelo(ruta or MODELO)
    idx = np.where(ds.valid)[0]
    prob = np.full(ds.N, np.nan)
    for i in range(0, len(idx), 2048):
        b = idx[i:i + 2048]
        X = np.stack([ds.X[j - D.W + 1:j + 1] for j in b])
        prob[b] = predecir(X)
    return prob, ck["umbral"]


def tramos_congelados(ds, nuevas):
    """Segundos a excluir porque la MAQUINA se congelo (laptop suspendida, etc.): el collector deja de
    escribir varios segundos a la vez y al volver ningun servicio responde, lo que la regla de etiquetado
    toma como cuello de botella en todos los servicios a la vez. No es un problema del sistema.
    Se detectan como saltos > 3 s entre mediciones de reservaciones y se excluyen 5 s antes y 30 s despues."""
    excl = np.zeros(ds.N, dtype=bool)
    notas = []
    for run in sorted(set(ds.run[nuevas])):
        t0 = json.load(open(os.path.join(VIVO, run, "meta.json")))["t0_locust"]
        m = pd.read_csv(os.path.join(VIVO, run, "metrics.csv"))
        ts = m[m["servicio"] == "reservaciones"]["timestamp"].sort_values()
        d = ts.diff()
        for fin, salto in zip(ts[d > 3], d[d > 3]):
            a, b = fin - salto - t0 - 5, fin - t0 + 30
            excl |= (ds.run == run) & (ds.sec >= a) & (ds.sec <= b)
            notas.append("%s: congelamiento de %.0f s en t = %.0f s (se excluye de %.0f a %.0f s)"
                         % (run, salto, fin - salto - t0, a, b))
    return excl, notas


def evaluar(modelo=None, resultados="resultados_v7"):
    import run_campaign as RC
    RC.DATA = VIVO
    RC.consolidar()
    rc = subprocess.run([PY, os.path.join(HERE, "ml", "etiquetar.py"), "--data", VIVO, "--solo-propio",
                         "--sufijo", "_propio"], cwd=HERE, stdout=subprocess.DEVNULL).returncode
    if rc != 0:
        raise SystemExit("Fallo el etiquetado de data/vivo")
    ds = D.cargar(VIVO, "_propio", "y_30s")
    prob, umbral = predecir_todo(ds, modelo)
    print("Modelo: %s (umbral %.3f) | comparado con data/%s" % (modelo or MODELO, umbral, resultados))
    nuevas = np.array([r.startswith("vivo_") for r in ds.run])
    excl, notas = tramos_congelados(ds, nuevas)
    if notas:
        print("\nTRAMOS EXCLUIDOS (la maquina se congelo, no es un problema del sistema):")
        for n in notas:
            print("  " + n)
        antes = len(ds.eps)
        ds.eps = ds.eps[~excl[ds.eps["gi"].values]].reset_index(drop=True)
        print("  episodios excluidos: %d" % (antes - len(ds.eps)))
    cub = np.isfinite(prob) & nuevas & ~excl
    lstm = cub & (prob > umbral)
    cpu = D.alarma_cpu(ds) & cub
    esc = np.array([escenario(r) for r in ds.run])
    mB = ds.valid & (ds.evalu == 1) & cub
    leads = {"LSTM": D.lead_times(ds, lstm), "CPU > 80 %": D.lead_times(ds, cpu)}
    filas = []
    for e in ("normal", "progresivo", "burst"):
        m = mB & (esc == e)
        runs = set(ds.run[(esc == e) & cub])
        for nombre, al in (("LSTM", lstm), ("CPU > 80 %", cpu)):
            r = D.prf(ds.y, al, m)
            lt = leads[nombre]
            lt = lt[lt["run"].isin(runs)]
            filas.append({"escenario": e, "metodo": nombre, "corridas": len(runs), "episodios": len(lt),
                          "f1": r["f1"], "precision": r["precision"], "recall": r["recall"],
                          "falsas_h": D.falsas_alarmas_hora(ds, al, m),
                          "avisados_%": 100 * lt["detectado"].mean() if len(lt) else np.nan,
                          "anticipacion_s": lt["lead_s"].median() if len(lt) else np.nan})
    vivo = pd.DataFrame(filas)
    vivo.to_csv(os.path.join(VIVO, "verificacion_%s.csv" % resultados.replace("resultados_", "")), index=False)
    cv = pd.read_csv(os.path.join(HERE, "data", resultados, "por_escenario_carga.csv"))
    comp = vivo.merge(cv, on=["escenario", "metodo"], suffixes=("_vivo", "_cv"))
    cols = ["escenario", "metodo", "episodios_vivo", "episodios_cv", "f1_vivo", "f1_cv", "falsas_h_vivo",
            "falsas_h_cv", "avisados_%_vivo", "avisados_%_cv", "anticipacion_s_vivo", "anticipacion_s_cv"]
    pd.set_option("display.width", 250)
    print("\nVIVO (corridas nuevas, modelo final) vs CV (80 corridas de la validacion cruzada):")
    print(comp[cols].round(2).to_string(index=False))
    print("\nEPISODIOS DE LAS CORRIDAS NUEVAS (v7):")
    lt = leads["LSTM"]
    lt = lt[lt["run"].str.startswith("vivo_")]
    print(lt[["run", "srv", "causa", "detectado", "lead_s", "alarma_previa"]].to_string(index=False))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # los eventos de la demo llevan flechas
    except Exception:
        pass
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("modo", choices=["correr", "evaluar"], nargs="?", default="evaluar")
    ap.add_argument("--modelo", default=None, help="modelo (.pt) o ensamble (.json); por defecto v7")
    ap.add_argument("--resultados", default="resultados_v7",
                    help="carpeta de data/ con por_escenario_carga.csv para comparar (la del mismo modelo)")
    a = ap.parse_args()
    if a.modo == "correr":
        correr()
    else:
        evaluar(a.modelo, a.resultados)
