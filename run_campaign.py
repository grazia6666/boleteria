"""
Campana completa: corre todos los experimentos en cadena y, al final, une los datos.
Entre corridas reinicia los contenedores (estado en memoria limpio, evita quedarse sin RAM).

Uso (desde la raiz del proyecto):
  python run_campaign.py --dry-run            # solo muestra el plan y el tiempo estimado
  python run_campaign.py --only normal_base   # prueba corta: un grupo
  python run_campaign.py                      # corre todo + consolida el dataset
  python run_campaign.py --reps 2             # mas repeticiones
  python run_campaign.py --solo-consolidar    # no corre nada, solo une lo que ya hay
  python run_campaign.py --repetir-contaminadas   # repite las corridas que no pasan el control de calidad
Control de calidad: al terminar cada corrida se mide la CPU total por request ANTES de inyectar; si
es > 1.5 x la de normal_base, la maquina estaba lenta, la corrida no se marca OK (deja un archivo
CONTAMINADA) y se repetira. La consolidacion anade cpu_por_req_pre, costo_vs_base y calidad a
resumen_corridas.csv.
Si se interrumpe (Ctrl+C), consolida lo ya terminado; al volver a correrlo SALTA las
corridas completas (marca OK en su carpeta) y repite la incompleta.

Grupos (con --reps 1):
  normal_base      carga constante, sin inyeccion (negativos limpios)             2
  normal_inj       carga constante + inyeccion en cada servicio                   6
  burst_inj        picos de trafico + inyeccion en cada servicio                  6
  progresivo_base  trafico creciente y decreciente, sin inyeccion                 2
  var_inj          inyeccion LEVE (cpu_load 90, 15 usuarios) y FUERTE (250, 20)  12
  suave_inj        inyeccion SUAVE (cpu_load 25 y 40) con rampa larga de 90 s    12
Cada corrida con inyeccion tiene 2 eventos: uno GRADUAL (rampa) y uno ABRUPTO.
suave_inj busca episodios que crezcan despacio (o que no lleguen a saturar): son los que de
verdad se pueden anticipar y dan negativos dificiles.
La severidad varia: las etiquetas se derivan de los sintomas observados, asi que algunas
inyecciones leves no saturaran (negativos dificiles) y otras si.

Salida de la consolidacion, en data/:
  dataset_metrics.csv   metricas por servicio y segundo, de todas las corridas
  dataset_edges.csv     metricas por dependencia (origen -> destino) y segundo
  resumen_corridas.csv  una fila por corrida
  Columnas anadidas: run_id, escenario, t_rel (s desde el inicio de la carga), en_carga,
  inj_servicio (ese servicio tenia inyeccion activa), inj_alguna, inj_tipo (gradual/abrupto).
  Son la verdad de lo INYECTADO, no la etiqueta de cuello de botella (esa se deriva despues).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
PY = sys.executable
PUERTOS = {"usuarios": 8001, "eventos": 8002, "reservaciones": 8003,
           "pago": 8004, "ticket": 8005, "notificaciones": 8006}
SERVICIOS = list(PUERTOS)


# ------------------------------------------------------------------ plan
def build_plan(reps, load):
    plan = []
    for r in range(1, 2 * reps + 1):
        plan.append(dict(id=f"normal_base_r{r}", group="normal_base", scenario="normal",
                         duration=300, inject=[], peak=100))
    for r in range(1, reps + 1):
        for s in SERVICIOS:
            plan.append(dict(id=f"normal_inj_{s}_r{r}", group="normal_inj", scenario="normal",
                             duration=360, peak=100,
                             inject=[f"{s}:60:90:{load}:40", f"{s}:220:60:{load}"]))
    for r in range(1, reps + 1):
        for s in SERVICIOS:
            # los picos de burst caen en 60-85, 180-205 y 300-325 s: las inyecciones los esquivan
            plan.append(dict(id=f"burst_inj_{s}_r{r}", group="burst_inj", scenario="burst",
                             duration=360, peak=70,
                             inject=[f"{s}:95:75:{load}:30", f"{s}:215:75:{load}"]))
    for r in range(1, 2 * reps + 1):
        plan.append(dict(id=f"progresivo_base_r{r}", group="progresivo_base", scenario="progresivo",
                         duration=360, inject=[], peak=60))
    for r in range(1, reps + 1):  # variantes de severidad: (nombre, cpu_load, usuarios base, rampa)
        for nombre, carga, usuarios, rampa in (("leve", 90, 15, 40), ("fuerte", 250, 20, 20)):
            for s in SERVICIOS:
                plan.append(dict(id=f"var_inj_{nombre}_{s}_r{r}", group="var_inj", scenario="normal",
                                 duration=360, peak=100, base=usuarios,
                                 inject=[f"{s}:60:90:{carga}:{rampa}", f"{s}:220:60:{carga}"]))
    for r in range(1, reps + 1):  # suaves: rampa de 90 s dentro de 120 s, luego escalon de 90 s
        for carga in (25, 40):
            for s in SERVICIOS:
                plan.append(dict(id=f"suave_inj_{carga}_{s}_r{r}", group="suave_inj", scenario="normal",
                                 duration=360, peak=100,
                                 inject=[f"{s}:60:120:{carga}:90", f"{s}:230:90:{carga}"]))
    plan += plan_aleatorio(reps)
    return plan


# ------------------------------------------------------------------ calendario aleatorio
CARGAS = (25, 40, 90, 150, 250)


def inyecciones_aleatorias(rng, duracion, n):
    """n inyecciones SIN solaparse (cada falla queda aislada en el tiempo) con orden, servicio, tipo
    (gradual o abrupta), intensidad, duracion y separacion al azar. La primera empieza despues de los 45 s
    para que el control de calidad tenga su ventana sin inyeccion."""
    out, t = [], 45 + rng.uniform(0, 40)
    for _ in range(n):
        dur = rng.randint(40, 100)
        if t + dur > duracion - 20:
            break
        s, carga = rng.choice(SERVICIOS), rng.choice(CARGAS)
        if rng.random() < 0.5:
            out.append(f"{s}:{t:.0f}:{dur}:{carga}:{rng.randint(20, min(90, dur))}")   # gradual
        else:
            out.append(f"{s}:{t:.0f}:{dur}:{carga}")                                    # abrupta
        t += dur + rng.randint(25, 90)
    return out


def rafagas_aleatorias(rng, duracion, k, largo):
    """Inicios de k rafagas a intervalos irregulares, la primera despues de los 60 s."""
    out, t = [], 60 + rng.uniform(0, 40)
    while len(out) < k and t + largo < duracion - 20:
        out.append(round(t))
        t += largo + rng.uniform(40, 120)
    return out


def fallas_simultaneas(rng, duracion, n, inicio_min, inicio_max, ventana=60):
    """n fallas en servicios DISTINTOS que se solapan: la primera empieza entre inicio_min e inicio_max y las
    demas hasta `ventana` s despues. Tipo al azar: gradual (60 %), lenta (15 %) o abrupta (25 %)."""
    out, t0 = [], rng.uniform(inicio_min, inicio_max)
    for k, s in enumerate(rng.sample(SERVICIOS, n)):
        t = t0 + (rng.uniform(0, ventana) if k else 0)
        u = rng.random()
        if u < 0.60:
            rampa = rng.choice((30, 60, 90, 150))
            d, carga, spec = rampa + rng.randint(20, 60), rng.choice((90, 150, 250)), "gradual"
        elif u < 0.75:
            rampa = rng.choice((120, 180))
            d, carga, spec = rampa + rng.randint(20, 40), rng.choice((60, 90)), "lenta"
        else:
            rampa, d, carga, spec = 0, rng.randint(40, 90), rng.choice((150, 250)), "abrupta"
        d = min(d, int(duracion - 20 - t))
        if d < 30:
            continue
        rampa = min(rampa, d)
        out.append(f"{s}:{t:.0f}:{d}:{carga}" + (f":{rampa}" if rampa else ""))
    return out


def plan_aleatorio(reps):
    """Grupos con CALENDARIO ALEATORIO: en las campanas anteriores el fallo gradual iba siempre primero y el
    abrupto ~45 s despues en el mismo servicio, y los picos de burst caian cada 120 s; el modelo aprendio
    parte de ese calendario. Aqui el orden, el momento, el servicio, el tipo, la intensidad y la duracion de
    las inyecciones, y los momentos de las rafagas, son al azar. Cada corrida usa su propia semilla (el
    nombre), asi el plan es reproducible y no cambia al subir --reps."""
    import random
    plan = []
    for i in range(1, 24 * reps + 1):
        rng = random.Random(f"rnd_normal-{i}")
        dur = rng.choice((300, 360, 420))
        plan.append(dict(id=f"rnd_normal_{i:02d}", group="rnd_normal", scenario="normal", duration=dur,
                         peak=100, base=rng.randint(12, 20),
                         inject=inyecciones_aleatorias(rng, dur, rng.choice((1, 2, 2, 3)))))
    for i in range(1, 10 * reps + 1):
        rng = random.Random(f"rnd_burst-{i}")
        dur, largo = rng.choice((360, 420)), rng.randint(15, 35)
        plan.append(dict(id=f"rnd_burst_{i:02d}", group="rnd_burst", scenario="burst", duration=dur,
                         peak=rng.randint(55, 85), base=15, burst_len=largo,
                         burst_times=rafagas_aleatorias(rng, dur, rng.choice((2, 3, 3, 4)), largo),
                         inject=inyecciones_aleatorias(rng, dur, rng.choice((0, 1, 2)))))
    for i in range(1, 6 * reps + 1):
        rng = random.Random(f"rnd_progresivo-{i}")
        dur = rng.choice((300, 360, 420))
        plan.append(dict(id=f"rnd_progresivo_{i:02d}", group="rnd_progresivo", scenario="progresivo",
                         duration=dur, peak=rng.randint(45, 90), base=15,
                         inject=inyecciones_aleatorias(rng, dur, rng.choice((0, 0, 1)))))
    # 10 corridas por grupo, sin importar --reps
    for i in range(1, 11):
        # rafagas que suben en rampa: cada velocidad 2 veces, para medir anticipacion vs. velocidad de subida
        rng = random.Random(f"rnd_rampa-{i}")
        rampa, largo = RAMPAS[(i - 1) % len(RAMPAS)], rng.randint(20, 35)
        dur = 540 if rampa >= 120 else 420   # las rafagas lentas necesitan mas corrida para que entren 2
        plan.append(dict(id=f"rnd_rampa_{i:02d}", group="rnd_rampa", scenario="burst", duration=dur,
                         peak=rng.randint(60, 85), base=15, burst_len=largo, burst_ramp=rampa,
                         burst_times=rafagas_aleatorias(rng, dur, rng.choice((2, 3)), largo + rampa),
                         inject=[]))
    for i in range(1, 11):
        # fallas lentas: el servicio tarda 2-5 min en llegar a su intensidad maxima
        rng = random.Random(f"rnd_lenta-{i}")
        dur, inj, t = 480, [], 45 + rng.uniform(0, 40)
        for _ in range(rng.choice((1, 2))):
            rampa = rng.choice((120, 180, 240, 300))
            d = rampa + rng.randint(20, 60)
            if t + d > dur - 20:
                break
            inj.append(f"{rng.choice(SERVICIOS)}:{t:.0f}:{d}:{rng.choice((60, 90, 90, 150))}:{rampa}")
            t += d + rng.randint(30, 90)
        plan.append(dict(id=f"rnd_lenta_{i:02d}", group="rnd_lenta", scenario="normal", duration=dur,
                         peak=100, base=rng.randint(12, 20), inject=inj))
    # caidas en cadena: carga que crece o rafagas + fallas, y varias fallas a la vez (10 corridas cada uno)
    for i in range(1, 11):
        rng = random.Random(f"rnd_combo_prog-{i}")
        dur = rng.choice((360, 420))
        plan.append(dict(id=f"rnd_combo_prog_{i:02d}", group="rnd_combo_prog", scenario="progresivo",
                         duration=dur, peak=rng.randint(55, 80), base=15,
                         inject=fallas_simultaneas(rng, dur, rng.choice((1, 2, 2)), 60, 180)))
    for i in range(1, 11):
        rng = random.Random(f"rnd_combo_burst-{i}")
        dur, largo, rampa = 420, rng.randint(20, 35), rng.choice((0, 0, 20, 40, 60))
        plan.append(dict(id=f"rnd_combo_burst_{i:02d}", group="rnd_combo_burst", scenario="burst", duration=dur,
                         peak=rng.randint(60, 85), base=15, burst_len=largo, burst_ramp=rampa,
                         burst_times=rafagas_aleatorias(rng, dur, rng.choice((2, 3)), largo + rampa),
                         inject=fallas_simultaneas(rng, dur, rng.choice((1, 2)), 60, 220)))
    for i in range(1, 11):
        rng = random.Random(f"rnd_simultanea-{i}")
        dur = rng.choice((360, 420))
        plan.append(dict(id=f"rnd_simultanea_{i:02d}", group="rnd_simultanea", scenario="normal", duration=dur,
                         peak=100, base=rng.randint(12, 20),
                         inject=fallas_simultaneas(rng, dur, rng.choice((2, 2, 3)), 50, 150)))
    return plan


RAMPAS = (10, 20, 40, 60, 120)   # segundos de subida de las rafagas de rnd_rampa


# ------------------------------------------------------------------ docker / salud
def wait_healthy(timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if all(requests.get(f"http://localhost:{p}/health", timeout=2).status_code == 200
                   for p in PUERTOS.values()):
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def restart_stack():
    subprocess.run(["docker", "compose", "restart"], cwd=HERE, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait_healthy():
        raise RuntimeError("Los servicios no quedaron sanos despues del reinicio")
    time.sleep(10)


def preflight():
    """Comprueba que los contenedores y el collector tienen el codigo nuevo ANTES de gastar horas."""
    print("Verificacion previa...")
    if not wait_healthy(30):
        raise SystemExit("Los servicios no responden. Levanta con: docker compose up -d --build")
    try:
        r = requests.post("http://localhost:8003/reservations", timeout=15,
                          json={"user_id": 1, "event_id": 1, "quantity": 1})
        r.raise_for_status()
        texto = requests.get("http://localhost:8003/metrics", timeout=5).text
    except Exception as e:
        raise SystemExit(f"La cadena de servicios no funciona ({e}). Revisa: docker compose logs")
    problemas = []
    if "downstream_request_duration_seconds" not in texto:
        problemas.append("Los contenedores tienen un common.py VIEJO (sin metricas de dependencias).")
    if 'le="0.15"' not in texto:
        problemas.append("Los contenedores tienen un common.py VIEJO (sin buckets finos de latencia).")
    if problemas:
        raise SystemExit("\n".join(problemas) + "\nCopia el common.py nuevo a los 6 servicios y corre: "
                         "docker compose up -d --build")
    cpath = os.path.join(HERE, "collector", "collector.py")
    if "edges.csv" not in open(cpath, encoding="utf-8").read():
        raise SystemExit("Tu collector/collector.py es la version vieja (no guarda edges.csv). Reemplazalo.")
    print("  OK: servicios, metricas de dependencias y collector actualizados\n")


# ------------------------------------------------------------------ consolidacion
FACTOR_CONTAMINADA = 1.5   # costo previo > 1.5 x el de normal_base => la maquina estaba lenta


def costo_previo(m, ev, t0):
    """Control de calidad: CPU total del sistema (suma de los 6 servicios, %) por request/s de entrada
    (rps de reservaciones), en la mediana de [15 s, primera inyeccion - 2 s] (como mucho hasta 58 s).
    No depende de la carga: si para el mismo trabajo se gasta mucha mas CPU que en normal_base, el
    host estaba ocupado o lento (otro programa, ahorro de energia, temperatura) y la corrida no es
    comparable con las demas."""
    fin = 58.0
    if len(ev):
        fin = min(fin, float(ev["t_inicio"].min() - t0) - 2.0)
    v = m[(m["t_rel"] >= 15) & (m["t_rel"] <= fin)]
    if v.empty:
        return float("nan")
    v = v.assign(sec=v["t_rel"].astype(int))
    cpu = v.groupby("sec")["cpu"].sum() / v.groupby("sec")["servicio"].nunique() * 6
    rps = v[v["servicio"] == "reservaciones"].groupby("sec")["rps"].mean()
    x = (cpu / rps).replace([float("inf")], float("nan")).dropna()
    return round(float(x.median()), 2) if len(x) else float("nan")


def referencia_costo():
    """Costo previo de las corridas normal_base completas (la linea base de la maquina)."""
    vals = []
    for d in glob.glob(os.path.join(DATA, "normal_base_r*")):
        if os.path.exists(os.path.join(d, "OK")):
            vals.append(cargar_corrida(d)[2]["cpu_por_req_pre"])
    vals = [v for v in vals if v == v]
    return sorted(vals)[len(vals) // 2] if vals else None


def cargar_corrida(d):
    import pandas as pd
    run_id = os.path.basename(d)
    meta = {}
    if os.path.exists(os.path.join(d, "meta.json")):
        meta = json.load(open(os.path.join(d, "meta.json")))
    m = pd.read_csv(os.path.join(d, "metrics.csv"))
    t0 = meta.get("t0_locust", m["timestamp"].min())
    dur = meta.get("duration", 0)

    m["run_id"] = run_id
    m["escenario"] = meta.get("scenario", "")
    m["t_rel"] = (m["timestamp"] - t0).round(3)
    m["en_carga"] = ((m["t_rel"] >= 0) & ((m["t_rel"] <= dur) if dur else True)).astype(int)
    m["inj_servicio"] = 0
    m["inj_alguna"] = 0
    m["inj_tipo"] = ""

    ev = pd.DataFrame()
    ep = os.path.join(d, "events.csv")
    if os.path.exists(ep):
        ev = pd.read_csv(ep)
        if "rampa_s" not in ev.columns:
            ev["rampa_s"] = 0
        for _, e in ev.iterrows():
            ventana = (m["timestamp"] >= e["t_inicio"]) & (m["timestamp"] <= e["t_fin"])
            propia = ventana & (m["servicio"] == e["servicio"])
            m.loc[ventana, "inj_alguna"] = 1
            m.loc[propia, "inj_servicio"] = 1
            m.loc[propia, "inj_tipo"] = "gradual" if e["rampa_s"] > 0 else "abrupto"

    edges = pd.DataFrame()
    gp = os.path.join(d, "edges.csv")
    if os.path.exists(gp):
        edges = pd.read_csv(gp)
        edges["run_id"] = run_id
        edges["escenario"] = meta.get("scenario", "")
        edges["t_rel"] = (edges["timestamp"] - t0).round(3)

    resumen = {
        "run_id": run_id, "escenario": meta.get("scenario", ""), "duracion_s": dur,
        "cpu_por_req_pre": costo_previo(m, ev, t0),
        "base_users": meta.get("base_users", ""), "peak_users": meta.get("peak_users", ""),
        "filas": len(m), "filas_scrape_fallido": int((m["scrape_ok"] == 0).sum()),
        "eventos": len(ev),
        "servicios_inyectados": ",".join(sorted(set(ev["servicio"]))) if len(ev) else "",
        "eventos_graduales": int((ev["rampa_s"] > 0).sum()) if len(ev) else 0,
        "eventos_abruptos": int((ev["rampa_s"] == 0).sum()) if len(ev) else 0,
        "filas_edges": len(edges),
    }
    return m, edges, resumen


def consolidar():
    import pandas as pd
    dirs = sorted(d for d in glob.glob(os.path.join(DATA, "*")) if os.path.isdir(d)
                  and os.path.exists(os.path.join(d, "metrics.csv"))
                  and os.path.exists(os.path.join(d, "OK")))
    if not dirs:
        print("No hay corridas completas (con OK) para consolidar.")
        return
    ms, es, rs = [], [], []
    for d in dirs:
        m, e, r = cargar_corrida(d)
        ms.append(m)
        rs.append(r)
        if len(e):
            es.append(e)
    metrics = pd.concat(ms, ignore_index=True)
    resumen = pd.DataFrame(rs)
    ref = resumen[resumen["run_id"].str.startswith("normal_base")]["cpu_por_req_pre"].median()
    resumen["costo_vs_base"] = (resumen["cpu_por_req_pre"] / ref).round(2)
    resumen["calidad"] = ["contaminada" if x > FACTOR_CONTAMINADA else "ok" for x in resumen["costo_vs_base"]]
    resumen[["run_id", "cpu_por_req_pre", "costo_vs_base", "calidad"]].to_csv(
        os.path.join(DATA, "calidad_corridas.csv"), index=False)   # lo lee entrenar.py --excluir-contaminadas
    metrics.to_csv(os.path.join(DATA, "dataset_metrics.csv"), index=False)
    resumen.to_csv(os.path.join(DATA, "resumen_corridas.csv"), index=False)
    if es:
        pd.concat(es, ignore_index=True).to_csv(os.path.join(DATA, "dataset_edges.csv"), index=False)

    print("\n================ DATASET CONSOLIDADO ================")
    print(f"Corridas incluidas: {len(dirs)}")
    print(f"Filas (servicio-segundo): {len(metrics)}  | en carga: {int(metrics['en_carga'].sum())}")
    print(f"Filas con inyeccion activa en su propio servicio: {int(metrics['inj_servicio'].sum())}")
    print(f"Eventos inyectados: {int(resumen['eventos'].sum())} "
          f"({int(resumen['eventos_graduales'].sum())} graduales, {int(resumen['eventos_abruptos'].sum())} abruptos)")
    print("\nCorridas por escenario:")
    print(resumen["escenario"].value_counts().to_string())
    cnt = {}
    for _, r in resumen.iterrows():
        for s in r["servicios_inyectados"].split(","):
            if s:
                cnt[s] = cnt.get(s, 0) + r["eventos"]
    print("\nEventos por servicio inyectado:")
    print(pd.Series(cnt).sort_index().to_string() if cnt else "(ninguno)")
    sin_edges = [r["run_id"] for _, r in resumen.iterrows() if r["filas_edges"] == 0]
    if sin_edges:
        print(f"\nAviso: {len(sin_edges)} corridas sin datos de dependencias (edges): {', '.join(sin_edges[:5])}...")
    malas = resumen[resumen["calidad"] == "contaminada"]
    print(f"\nCONTROL DE CALIDAD (CPU por request antes de inyectar, normal_base = {ref:.2f}; "
          f"contaminada si > {FACTOR_CONTAMINADA} x):")
    if len(malas):
        print(f"  !! {len(malas)} corridas CONTAMINADAS (la maquina estaba lenta; repetirlas con "
              f"--repetir-contaminadas, o excluirlas con entrenar.py --excluir-contaminadas):")
        print(malas[["run_id", "cpu_por_req_pre", "costo_vs_base"]].to_string(index=False))
    else:
        print("  todas las corridas ok")
    fallidas = int(resumen["filas_scrape_fallido"].sum())
    if fallidas:
        print(f"Aviso: {fallidas} filas con scrape_ok=0 (se conservan: el servicio no respondio a /metrics)")
    print(f"\nArchivos en {DATA}: dataset_metrics.csv, resumen_corridas.csv"
          + (", dataset_edges.csv" if es else ""))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--load", type=int, default=150, help="cpu_load final de las inyecciones")
    ap.add_argument("--only", default="", help="grupos separados por coma")
    ap.add_argument("--cooldown", type=int, default=20, help="segundos de pausa entre corridas")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--solo-consolidar", action="store_true")
    ap.add_argument("--sin-consolidar", action="store_true")
    ap.add_argument("--sin-chequeo", action="store_true", help="omitir la verificacion previa")
    ap.add_argument("--repetir-contaminadas", action="store_true",
                    help="quitar la marca OK de las corridas que no pasan el control de calidad (se repiten)")
    ap.add_argument("--aceptar-contaminadas", action="store_true",
                    help="marcar OK una corrida nueva aunque no pase el control de calidad")
    args = ap.parse_args()

    if args.solo_consolidar:
        consolidar()
        return

    ref = referencia_costo()
    if args.repetir_contaminadas:
        if ref is None:
            raise SystemExit("No hay corridas normal_base completas: no hay referencia para el control de calidad.")
        for d in sorted(glob.glob(os.path.join(DATA, "*"))):
            ok = os.path.join(d, "OK")
            if os.path.exists(ok) and not os.path.basename(d).startswith("normal_base"):
                c = cargar_corrida(d)[2]["cpu_por_req_pre"]
                if c > FACTOR_CONTAMINADA * ref:
                    os.remove(ok)
                    print(f"  se repetira {os.path.basename(d)} (costo {c:.2f} = {c / ref:.1f} x normal_base)")

    plan = build_plan(args.reps, args.load)
    if args.only:
        grupos = set(args.only.split(","))
        plan = [p for p in plan if p["group"] in grupos]
    pendientes = [p for p in plan if not os.path.exists(os.path.join(DATA, p["id"], "OK"))]

    seg = sum(p["duration"] + 25 + 10 + 10 + args.cooldown for p in pendientes)
    print(f"{len(plan)} corridas en el plan, {len(pendientes)} pendientes, tiempo estimado ~{seg/3600:.1f} h\n")
    for p in plan:
        hecho = "OK " if p not in pendientes else "   "
        print(f"  {hecho}{p['id']:<34} {p['scenario']:<11} {p['duration']}s  {' '.join(p['inject'])}")
    if args.dry_run:
        return

    if pendientes and not args.sin_chequeo:
        preflight()

    try:
        for i, p in enumerate(pendientes, 1):
            print(f"\n===== [{i}/{len(pendientes)}] {p['id']} =====")
            d = os.path.join(DATA, p["id"])
            if os.path.exists(d):
                shutil.rmtree(d)  # corrida incompleta de un intento anterior
            restart_stack()
            cmd = [PY, os.path.join(HERE, "run_experiment.py"), "--scenario", p["scenario"],
                   "--duration", str(p["duration"]), "--run-id", p["id"], "--peak-users", str(p["peak"]),
                   "--base-users", str(p.get("base", 15))]
            for inj in p["inject"]:
                cmd += ["--inject", inj]
            if p.get("burst_times"):
                cmd += ["--burst-times", ",".join(str(x) for x in p["burst_times"]),
                        "--burst-len", str(p["burst_len"])]
                if p.get("burst_ramp"):
                    cmd += ["--burst-ramp", str(p["burst_ramp"])]
            rc = subprocess.run(cmd, cwd=HERE).returncode
            if rc == 0 and os.path.exists(os.path.join(d, "metrics.csv")):
                c = cargar_corrida(d)[2]["cpu_por_req_pre"]
                if ref and c > FACTOR_CONTAMINADA * ref and not args.aceptar_contaminadas:
                    open(os.path.join(d, "CONTAMINADA"), "w").write("%.2f vs %.2f\n" % (c, ref))
                    print(f"!! {p['id']} NO pasa el control de calidad: CPU por request {c:.2f} = "
                          f"{c / ref:.1f} x normal_base. La maquina esta lenta u ocupada (cierra otros "
                          f"programas, conecta el cargador); se repetira la proxima vez")
                else:
                    open(os.path.join(d, "OK"), "w").write("ok")
                    if p["id"].startswith("normal_base"):
                        ref = referencia_costo()
            else:
                print(f"!! {p['id']} no termino bien (codigo {rc}); se repetira la proxima vez")
            time.sleep(args.cooldown)
        print("\nCampana terminada")
    except KeyboardInterrupt:
        print("\nInterrumpido: se consolida lo que ya esta completo")

    if not args.sin_consolidar:
        consolidar()


if __name__ == "__main__":
    main()