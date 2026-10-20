"""
LSTM que predice si un servicio tendra cuello de botella en los proximos 30 s, evaluado con
validacion cruzada AGRUPADA POR CORRIDA y comparado contra el baseline.

Uso (desde la raiz del proyecto, despues de etiquetar.py):
  python ml/entrenar.py
  python ml/entrenar.py --folds 4 --epocas 15 --sufijo _piso50
  python ml/entrenar.py --criterio f1                       # umbral como antes (maximiza F1)
  python ml/entrenar.py --fpr-max 0.02                      # F0.5 con tope de falsas alarmas
  python ml/entrenar.py --sin-latencia --salida resultados_sin_lat   # ablacion sin lat_*
  python ml/entrenar.py --objetivo y_inicio_30s --contexto --tendencias --salida resultados_v3
  python ml/entrenar.py --contexto --tendencias --excluir-contaminadas --salida resultados_v4_limpio

Protocolo (nada de esto se ajusta mirando el test):
  - Cada corrida completa va a UN solo fold (estratificado por grupo), asi ventanas casi
    identicas nunca quedan a ambos lados de la division.
  - En cada fold: ~15% de las corridas de entrenamiento se reservan como validacion; con ellas
    se elige la epoca (early stopping) y el UMBRAL de decision: el que maximiza F-beta (beta=0.5
    por defecto, pesa mas la precision) en la vista B, opcionalmente con fpr <= --fpr-max.
  - --sin-latencia quita lat_mean_ms y lat_p95_ms de las entradas, para medir cuanto del
    resultado viene de leer la latencia (de la que sale la etiqueta) y cuanto de cpu/cola/rps.
  - --objetivo y_inicio_30s predice que EMPIEZA un cuello de botella en los proximos 30 s y entrena
    (y valida) solo con segundos sin cuello de botella: deja de premiar la alarma que sigue
    encendida mientras el problema dura. Se predice igual sobre todas las muestras validas.
  - --contexto anade cpu/cola/lat_p95 de los servicios a los que llama (*_dep) y --tendencias
    el cambio en 5 y 15 s de cpu, cola, rps, lat_p95 (y cpu_dep con --contexto).
  - La normalizacion usa solo estadisticas del entrenamiento.
  - Perdida BCE con pos_weight (clases desbalanceadas).
  - Cada corrida se predice exactamente una vez (cuando su fold es el de prueba); las metricas
    se calculan sobre esas predicciones agrupadas, y tambien por fold (media y desviacion).

Salidas en data/<--salida>/ (resultados por defecto): metricas_A.csv, metricas_B.csv, por_escenario.csv, lead_time_*.csv,
  por_fold.csv, predicciones.csv, comparacion.png, ejemplo_timeline.png  y  models/lstm_final.pt
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datos as D  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def asignar_folds(ds, k, seed):
    rng = np.random.RandomState(seed)
    runs = pd.DataFrame({"run": ds.run, "grupo": ds.grupo}).drop_duplicates()
    fold, c = {}, 0
    for g, sub in runs.groupby("grupo"):
        r = sorted(sub["run"])
        rng.shuffle(r)
        for x in r:
            fold[x] = c % k
            c += 1
    return fold


def split_val(ds, train_runs, seed):
    rng = np.random.RandomState(seed)
    val = set()
    g_of = dict(zip(ds.run, ds.grupo))
    por_g = {}
    for r in train_runs:
        por_g.setdefault(g_of[r], []).append(r)
    for g, rs in por_g.items():
        rs = sorted(rs)
        rng.shuffle(rs)
        if len(rs) >= 4:
            val.update(rs[:max(1, int(round(0.15 * len(rs))))])
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--sufijo", default="")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epocas", type=int, default=15)
    ap.add_argument("--paciencia", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", type=float, default=80.0)
    ap.add_argument("--criterio", choices=["fbeta", "f1"], default="fbeta",
                    help="metrica que maximiza el umbral en validacion (vista B)")
    ap.add_argument("--beta", type=float, default=0.5, help="beta de F-beta (<1 favorece la precision)")
    ap.add_argument("--fpr-max", type=float, default=None,
                    help="solo umbrales con fpr <= este valor en validacion (vista B)")
    ap.add_argument("--sin-latencia", action="store_true", help="quitar las entradas de latencia (lat_*)")
    ap.add_argument("--objetivo", choices=["y_30s", "y_inicio_30s"], default="y_30s")
    ap.add_argument("--contexto", action="store_true", help="anadir metricas de las dependencias (*_dep)")
    ap.add_argument("--tendencias", action="store_true", help="anadir cambios en 5 y 15 s (d5_*, d15_*)")
    ap.add_argument("--excluir-contaminadas", action="store_true",
                    help="dejar fuera las corridas marcadas en data/calidad_corridas.csv")
    ap.add_argument("--salida", default="resultados", help="subcarpeta de --data para los resultados")
    ap.add_argument("--sin-final", action="store_true", help="no entrenar/guardar el modelo final")
    # --- opciones para el analisis de hiperparametros (los valores por defecto reproducen v7)
    ap.add_argument("--seed-folds", type=int, default=None,
                    help="semilla del reparto en folds (por defecto = --seed); fijarla permite comparar "
                         "variantes con las mismas corridas de prueba")
    ap.add_argument("--balanceo", choices=["pos_weight", "ninguno", "submuestreo"], default="pos_weight",
                    help="pos_weight: pesa los positivos; submuestreo: en cada epoca usa todos los positivos y "
                         "--ratio negativos por positivo, elegidos al azar")
    ap.add_argument("--ratio", type=float, default=3.0, help="negativos por positivo con --balanceo submuestreo")
    ap.add_argument("--parada", choices=["loss", "loss_pond", "ap"], default="loss",
                    help="metrica de validacion para la parada temprana: loss (BCE sin ponderar, v7), "
                         "loss_pond (BCE con el mismo pos_weight del entrenamiento) o ap (average precision)")
    ap.add_argument("--capas", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--wd", type=float, default=0.0, help="weight decay (regularizacion L2)")
    ap.add_argument("--ventana", type=int, default=30, help="segundos de historia que ve el modelo")
    a = ap.parse_args()
    D.W = a.ventana

    import torch
    import torch.nn as nn
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    out = os.path.join(a.data, a.salida)
    os.makedirs(os.path.join(out, "models"), exist_ok=True)

    excluidas = D.corridas_contaminadas(a.data) if a.excluir_contaminadas else []
    if excluidas:
        print("Excluidas por control de calidad (%d): %s" % (len(excluidas), ", ".join(excluidas)))
    ds = D.cargar(a.data, a.sufijo, a.objetivo, excluidas)
    print("Muestras validas: %d | corridas: %d | episodios: %d | objetivo: %s"
          % (ds.valid.sum(), len(set(ds.run)), len(ds.eps), a.objetivo))
    offs = torch.arange(-D.W + 1, 1)

    def usar(n):
        if a.sin_latencia and "lat_" in n:
            return False
        if n.endswith("_dep") and not a.contexto:
            return False
        if n.startswith("d") and n[1:].split("_")[0].isdigit() and not a.tendencias:
            return False
        return True
    feats_usadas = [n for n in D.CONT if usar(n)]
    cols = [D.CONT.index(n) for n in feats_usadas] + list(range(D.N_CONT, D.NF))
    print("Entradas: %s + falta + servicio (%d columnas)" % (", ".join(feats_usadas), len(cols)))
    # con y_inicio_30s se entrena y valida solo en segundos sin cuello de botella
    apto = (ds.evalu == 1) if a.objetivo == "y_inicio_30s" else np.ones(ds.N, dtype=bool)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(len(cols), a.hidden, num_layers=a.capas, batch_first=True,
                                dropout=a.dropout if a.capas > 1 else 0.0)
            self.drop = nn.Dropout(a.dropout)
            self.fc = nn.Linear(a.hidden, 1)

        def forward(self, x):
            o, _ = self.lstm(x)
            return self.fc(self.drop(o[:, -1])).squeeze(-1)

    def normalizar(train_mask):
        mu = ds.X[train_mask, :D.N_CONT].mean(0)
        sd = ds.X[train_mask, :D.N_CONT].std(0) + 1e-6
        Xn = ds.X.copy()
        Xn[:, :D.N_CONT] = (Xn[:, :D.N_CONT] - mu) / sd
        return torch.from_numpy(np.ascontiguousarray(Xn[:, cols])), mu, sd

    def predecir(model, Xt, idx):
        model.eval()
        ps = []
        with torch.no_grad():
            for i in range(0, len(idx), 2048):
                b = torch.from_numpy(idx[i:i + 2048])
                ps.append(torch.sigmoid(model(Xt[b[:, None] + offs])).numpy())
        return np.concatenate(ps) if ps else np.array([])

    def average_precision(p, y):
        orden = np.argsort(-p)
        ys = y[orden]
        tp = np.cumsum(ys)
        return float((tp[ys == 1] / (np.where(ys == 1)[0] + 1)).mean()) if ys.sum() else 0.0

    def entrenar_modelo(Xt, tr_idx, va_idx):
        model = Net()
        opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
        ytr_np = ds.y[tr_idx].astype(np.float32)
        ytr = torch.from_numpy(ytr_np)
        pos = float(ytr.sum())
        pw = torch.tensor([(len(ytr) - pos) / max(pos, 1.0)])
        lf = nn.BCEWithLogitsLoss(pos_weight=pw if a.balanceo == "pos_weight" else None)
        idx_pos, idx_neg = np.where(ytr_np == 1)[0], np.where(ytr_np == 0)[0]
        mejor, mejor_estado, sin_mejora = np.inf, None, 0
        for ep in range(a.epocas):
            model.train()
            if a.balanceo == "submuestreo":
                n_neg = min(len(idx_neg), int(a.ratio * len(idx_pos)))
                perm = np.random.permutation(np.concatenate([idx_pos, np.random.choice(idx_neg, n_neg, False)]))
            else:
                perm = np.random.permutation(len(tr_idx))
            tot = 0.0
            for i in range(0, len(perm), a.batch):
                sel = perm[i:i + a.batch]
                b = torch.from_numpy(tr_idx[sel])
                loss = lf(model(Xt[b[:, None] + offs]), ytr[sel])
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(sel)
            if len(va_idx):
                pv = predecir(model, Xt, va_idx)
                yv_np = ds.y[va_idx].astype(np.float32)
                yv = torch.from_numpy(yv_np)
                pc = torch.from_numpy(pv).clamp(1e-6, 1 - 1e-6)
                if a.parada == "loss":
                    lv = float(nn.functional.binary_cross_entropy(pc, yv))
                elif a.parada == "loss_pond":
                    w = torch.where(yv == 1, pw, torch.ones(1))
                    lv = float(nn.functional.binary_cross_entropy(pc, yv, weight=w))
                else:   # cuanto mayor la AP mejor: se minimiza 1 - AP
                    lv = 1.0 - average_precision(pv, yv_np)
            else:
                lv = tot / len(perm)
            print("    epoca %2d  loss_train=%.4f  val(%s)=%.4f" % (ep + 1, tot / len(perm), a.parada, lv))
            if lv < mejor - 1e-4:
                mejor = lv
                mejor_estado = {k: v.clone() for k, v in model.state_dict().items()}
                sin_mejora = 0
            else:
                sin_mejora += 1
                if sin_mejora >= a.paciencia:
                    break
        if mejor_estado is not None:
            model.load_state_dict(mejor_estado)
        return model

    def elegir_umbral(p, idx):
        """Maximiza --criterio en la vista B de validacion. Con --fpr-max solo considera umbrales que
        lo cumplen; si ninguno lo cumple, usa el de menor fpr."""
        mask_b = (ds.evalu[idx] == 1)
        if mask_b.sum() == 0 or ds.y[idx][mask_b].sum() == 0:
            return 0.5
        mb = np.zeros(ds.N, dtype=bool)
        mb[idx[mask_b]] = True
        cands = []
        for t in np.arange(0.05, 0.96, 0.05):
            al = np.zeros(ds.N, dtype=bool)
            al[idx] = p > t
            m = D.prf(ds.y, al, mb, beta=a.beta)
            cands.append((float(t), m[a.criterio], m["fpr"]))
        ok = [c for c in cands if c[1] == c[1]]
        if a.fpr_max is not None:
            dentro = [c for c in ok if c[2] <= a.fpr_max]
            if not dentro:
                print("    aviso: ningun umbral cumple fpr <= %g; se usa el de menor fpr" % a.fpr_max)
                return min(cands, key=lambda c: (c[2], -c[0]))[0]
            ok = dentro
        return max(ok, key=lambda c: c[1])[0] if ok else 0.5

    # ------------------------------------------------------------ validacion cruzada
    fold_de = asignar_folds(ds, a.folds, a.seed if a.seed_folds is None else a.seed_folds)
    fold_fila = np.array([fold_de[r] for r in ds.run])
    prob = np.full(ds.N, np.nan)
    alarma = np.zeros(ds.N, dtype=bool)
    umbrales, filas_fold = [], []
    t0 = time.time()
    for k in range(a.folds):
        test_runs = {r for r, f in fold_de.items() if f == k}
        train_runs = [r for r, f in fold_de.items() if f != k]
        val_runs = split_val(ds, train_runs, (a.seed if a.seed_folds is None else a.seed_folds) + k)
        es_te = np.isin(ds.run, list(test_runs))
        es_va = np.isin(ds.run, list(val_runs))
        es_tr = ~es_te & ~es_va
        tr_idx = np.where(es_tr & ds.valid & apto)[0]
        va_idx = np.where(es_va & ds.valid & apto)[0]
        te_idx = np.where(es_te & ds.valid)[0]
        print("\nFold %d/%d  train=%d corridas (%d muestras)  val=%d  test=%d (%d muestras)  positivos train=%.1f%%"
              % (k + 1, a.folds, len(train_runs) - len(val_runs), len(tr_idx), len(val_runs), len(test_runs),
                 len(te_idx), 100 * ds.y[tr_idx].mean()))
        Xt, mu, sd = normalizar(es_tr)
        model = entrenar_modelo(Xt, tr_idx, va_idx)
        pv = predecir(model, Xt, va_idx)
        thr = elegir_umbral(pv, va_idx) if len(va_idx) else 0.5
        umbrales.append(thr)
        pt = predecir(model, Xt, te_idx)
        prob[te_idx] = pt
        alarma[te_idx] = pt > thr
        mb = np.zeros(ds.N, dtype=bool)
        mb[te_idx] = ds.evalu[te_idx] == 1
        for nombre, al in (("LSTM", alarma), ("threshold CPU", D.alarma_cpu(ds, a.cpu))):
            m = D.prf(ds.y, al, mb, beta=a.beta)
            filas_fold.append({"fold": k + 1, "metodo": nombre, "umbral": thr if nombre == "LSTM" else a.cpu,
                               "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                               "fbeta": m["fbeta"], "fpr": m["fpr"],
                               "falsas_alarmas_h": D.falsas_alarmas_hora(ds, al, mb), "n": m["n"]})
        print("    umbral elegido en validacion: %.2f" % thr)
    print("\nValidacion cruzada: %.0f s" % (time.time() - t0))

    # ------------------------------------------------------------ comparacion
    cubierto = np.zeros(ds.N, dtype=bool)
    cubierto[np.isfinite(prob)] = True
    alarmas = {"LSTM": alarma,
               "threshold CPU>%g%%" % a.cpu: D.alarma_cpu(ds, a.cpu) & cubierto,
               "persistencia": D.alarma_persistencia(ds) & cubierto}
    rep = D.reporte(ds, alarmas, cubierto)
    D.imprimir_reporte(ds, rep, "LSTM vs BASELINE (validacion cruzada por corrida, predicciones agrupadas)")

    pf = pd.DataFrame(filas_fold)
    print("\nPOR FOLD (vista B) - media +- desviacion:")
    resumen_fold = pf.groupby("metodo")[["precision", "recall", "f1", "fbeta", "falsas_alarmas_h"]].agg(
        ["mean", "std"]).round(3)
    print(resumen_fold.to_string())

    mB = ds.valid & (ds.evalu == 1) & cubierto
    filas_esc = []
    for g in sorted(set(ds.grupo)):
        for nombre, al in alarmas.items():
            mg = mB & (ds.grupo == g)
            m = D.prf(ds.y, al, mg, beta=a.beta)
            filas_esc.append({"escenario": g, "metodo": nombre, "precision": m["precision"],
                              "recall": m["recall"], "f1": m["f1"], "fbeta": m["fbeta"],
                              "falsas_alarmas_h": D.falsas_alarmas_hora(ds, al, mg), "n": m["n"]})
    esc = pd.DataFrame(filas_esc).round(3)
    print("\nPOR GRUPO DE CORRIDAS (vista B):")
    print(esc.pivot(index="escenario", columns="metodo", values="f1").to_string())

    lead_por_grupo = {}
    for nombre, df in rep["lead"].items():
        print("\nLEAD TIME por grupo - %s:" % nombre)
        print(resumen_lead_grupo(df) if len(df) else "(vacio)")
        lead_por_grupo[nombre] = df

    # ------------------------------------------------------------ guardar
    rep["A"].to_csv(os.path.join(out, "metricas_A.csv"))
    rep["B"].to_csv(os.path.join(out, "metricas_B.csv"))
    esc.to_csv(os.path.join(out, "por_escenario.csv"), index=False)
    pf.to_csv(os.path.join(out, "por_fold.csv"), index=False)
    for nombre, df in rep["lead"].items():
        df.to_csv(os.path.join(out, "lead_time_%s.csv" % nombre.split()[0]), index=False)
    pd.DataFrame({"run": ds.run, "servicio": ds.srv, "t_rel": ds.sec, "prob_lstm": prob,
                  "alarma_lstm": alarma, "cpu": ds.cpu, "bn_ahora": ds.bn, "y_30s": ds.y_30s, "y_objetivo": ds.y,
                  "evaluable": ds.evalu})[cubierto].to_csv(os.path.join(out, "predicciones.csv"), index=False)
    json.dump({"umbrales_por_fold": umbrales, "parametros": vars(a), "excluidas": excluidas}, open(os.path.join(out, "config.json"), "w"), indent=2)

    graficas(ds, rep, prob, alarma, a, out)

    # ------------------------------------------------------------ modelo final
    if not a.sin_final:
        print("\nEntrenando el modelo final con todas las corridas (para la demo)...")
        val_runs = split_val(ds, sorted(set(ds.run)), a.seed + 99)
        es_va = np.isin(ds.run, list(val_runs))
        Xt, mu, sd = normalizar(~es_va)
        tr_idx = np.where(~es_va & ds.valid & apto)[0]
        va_idx = np.where(es_va & ds.valid & apto)[0]
        model = entrenar_modelo(Xt, tr_idx, va_idx)
        thr = float(np.median(umbrales))
        torch.save({"state": model.state_dict(), "mu": mu.tolist(), "sd": sd.tolist(), "umbral": thr,
                    "W": D.W, "H": D.H, "hidden": a.hidden, "capas": a.capas, "dropout": a.dropout, "columnas": D.COLUMNAS, "servicios": D.SERVICIOS,
                    "cols": cols, "feats_usadas": feats_usadas, "objetivo": a.objetivo,
                    "dependencias": D.DEPENDE},
                   os.path.join(out, "models", "lstm_final.pt"))
        print("Guardado: %s (umbral = mediana de los folds = %.2f)" % (os.path.join(out, "models", "lstm_final.pt"), thr))
    print("\nResultados en %s" % out)


def resumen_lead_grupo(df):
    return D.resumen_lead(df, "grupo").to_string()


def graficas(ds, rep, prob, alarma, a, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metodos = [m for m in rep["lead"] if len(rep["lead"][m])]
    causas = sorted(set().union(*[set(rep["lead"][m]["causa"]) for m in metodos])) if metodos else []
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    mB = rep["B"]
    x = np.arange(3)
    for i, m in enumerate(mB.index):
        ax[0].bar(x + i * 0.27, [mB.loc[m, "precision"], mB.loc[m, "recall"], mB.loc[m, "f1"]], 0.27, label=m)
    ax[0].set_xticks(x + 0.27)
    ax[0].set_xticklabels(["Precision", "Recall", "F1"])
    ax[0].set_title("Anticipacion (vista B)")
    ax[0].set_ylim(0, 1)
    ax[0].legend(fontsize=8)
    for i, m in enumerate(metodos):
        r = D.resumen_lead(rep["lead"][m], "causa").reindex(causas)
        ax[1].bar(np.arange(len(causas)) + i * 0.38, r["detectados_%"].fillna(0), 0.38, label=m)
        ax[2].bar(np.arange(len(causas)) + i * 0.38, r["lead_mediana_s"].fillna(0), 0.38, label=m)
    for k, t in ((1, "Episodios detectados antes de empezar (%)"), (2, "Prediction Lead Time, mediana (s)")):
        ax[k].set_xticks(np.arange(len(causas)) + 0.19)
        ax[k].set_xticklabels(causas, rotation=20, fontsize=8)
        ax[k].set_title(t)
        ax[k].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "comparacion.png"), dpi=130)
    plt.close(fig)

    # timeline de un episodio gradual con ventana previa larga
    e = ds.eps[(ds.eps["causa"] == "inyeccion_gradual") & (ds.eps["previo_libre_s"] >= 40)]
    e = e[np.isfinite(prob[e["gi"].values])] if len(e) else e
    if not len(e):
        return
    r = e.iloc[0]
    s0 = ds.sstart[r.gi]
    lo, hi = max(r.gi - 90, s0), min(r.gi + 60, np.where(ds.run == r.run)[0].max())
    sl = slice(lo, hi)
    t = ds.sec[sl]
    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(t, ds.cpu[sl], color="tab:orange", label="CPU %")
    ax[0].axhline(a.cpu, color="gray", ls="--", lw=1, label="threshold %g%%" % a.cpu)
    ax[0].set_ylabel("CPU (%)")
    ax[0].legend(fontsize=8)
    ax[1].plot(t, np.exp(ds.X[sl, 4]) - 1, color="tab:blue")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("latencia p95 (ms)")
    ax[2].plot(t, prob[sl], color="tab:green", label="P(cuello de botella en 30 s)")
    ax[2].fill_between(t, 0, 1, where=ds.bn[sl] == 1, color="red", alpha=0.15, label="cuello de botella real")
    ax[2].set_ylabel("probabilidad LSTM")
    ax[2].set_xlabel("segundos desde el inicio de la carga")
    ax[2].legend(fontsize=8)
    for x_ in ax:
        x_.axvline(ds.sec[r.gi], color="red", ls=":", lw=1)
    fig.suptitle("%s - %s (inicio del episodio en la linea roja)" % (r.run, r.srv))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ejemplo_timeline.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()