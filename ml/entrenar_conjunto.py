"""
v13: un LSTM que ve los 6 servicios a la vez.

v12 mira cada servicio por separado (con un resumen de los servicios a los que llama). v13 recibe, en cada
segundo, los ultimos 30 s de los 6 servicios juntos, asi puede ver que un servicio de abajo se esta cayendo
mientras decide sobre los que lo llaman (caidas en cadena). Dos formas:
  --forma conjunta      una muestra por segundo con los 6 servicios y 6 salidas (128 unidades)
  --forma por_servicio  una muestra por servicio y segundo, como v12, con los 6 servicios en la entrada mas
                        el indicador del servicio que se predice (64 unidades, como v12)

Todo lo demas es igual que v12 (entrenar.py --sufijo _propio --contexto --tendencias --seed-folds 42), para
que la comparacion sea justa: mismos datos y etiqueta (y_30s propia), las mismas 21 senales por servicio
(las de v12 sin el indicador de servicio), mismos folds por corrida, mismas corridas de validacion para la
parada temprana y el umbral (F0,5 en la vista B), normalizacion solo con el entrenamiento, BCE con
pos_weight, Adam 0,001, lote 256, 15 epocas, paciencia 3.

Uso (desde la raiz, despues de etiquetar.py --solo-propio --sufijo _propio):
  python ml/entrenar_conjunto.py --seed 1 --salida resultados_v13/s1
  python ml/entrenar_conjunto.py --seed 1 --salida resultados_v13/prueba --solo-fold 0 --sin-final   # prueba corta
  python ml/ensamble.py --dirs resultados_v13/s1 resultados_v13/s2 resultados_v13/s3 --salida resultados_v13
Salidas en data/<--salida>/: predicciones.csv y config.json con el mismo formato que entrenar.py (sirven para
ensamble.py, por_escenario.py y evaluar_externo.py) y models/conjunto_final.pt (entrenado con todas las
corridas). La demo todavia no sabe usar este modelo.
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
from entrenar import asignar_folds, split_val  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = len(D.SERVICIOS)
COLS = [D.COLUMNAS.index(n) for n in D.CONT] + [D.N_CONT]   # las 20 continuas de v12 + "falta"


def alinear(ds):
    """Une los 6 servicios de cada corrida por segundo. F[t, s] = fila de ds del servicio s en ese segundo
    (solo segundos que tienen los 6). pos = segundos seguidos hasta t dentro de la corrida (una ventana de W s
    es valida si pos >= W - 1)."""
    df = pd.DataFrame({"run": ds.run, "srv": ds.srv, "sec": ds.sec, "i": np.arange(ds.N)})
    F, runs, pos = [], [], []
    for run, g in df.groupby("run", sort=True):
        piv = g.pivot(index="sec", columns="srv", values="i").reindex(columns=D.SERVICIOS).dropna()
        n = len(piv)
        corte = np.r_[True, np.diff(piv.index.values) != 1]
        pos.append(np.arange(n) - np.maximum.accumulate(np.where(corte, np.arange(n), 0)))
        F.append(piv.values.astype(np.int64))
        runs += [run] * n
    return np.vstack(F), np.array(runs, dtype=object), np.concatenate(pos)


def muestras(F, M, pos, forma):
    """Indices de las muestras: un segundo (conjunta) o un par segundo*S + servicio (por_servicio)."""
    anclas = np.where((pos >= D.W - 1) & M.any(1))[0]
    if forma == "conjunta":
        return anclas
    a, s = np.nonzero(M[anclas])
    return anclas[a] * S + s


def segundo_de(idx, forma):
    return idx if forma == "conjunta" else idx // S


def normalizar(ds, filas_train, mu=None, sd=None):
    """Las columnas de COLS normalizadas con la media y desviacion del entrenamiento (como entrenar.py)."""
    if mu is None:
        Xtr = ds.X[filas_train][:, D.NORM]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xn = ds.X.copy()
    Xn[:, D.NORM] = (Xn[:, D.NORM] - np.asarray(mu, dtype=np.float32)) / np.asarray(sd, dtype=np.float32)
    return Xn[:, COLS], mu, sd


def tensor_6(Xc, F):
    import torch
    return torch.from_numpy(np.ascontiguousarray(Xc[F].reshape(len(F), -1).astype(np.float32)))


def crear_red(n_in, hidden, n_out):
    import torch.nn as nn

    class Red(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, n_out)

        def forward(self, x):
            o, _ = self.lstm(x)
            return self.fc(o[:, -1])
    return Red()


def entradas(G, idx, forma):
    """Ventanas de W s de los 6 servicios; en por_servicio se agrega el indicador del servicio que se predice."""
    import torch
    offs = torch.arange(-D.W + 1, 1)
    b = torch.from_numpy(np.asarray(segundo_de(idx, forma)))
    x = G[b[:, None] + offs]
    if forma == "conjunta":
        return x
    oh = torch.eye(S)[torch.from_numpy(np.asarray(idx % S))]
    return torch.cat([x, oh[:, None, :].expand(-1, D.W, -1)], 2)


def predecir(model, G, idx, forma):
    """Probabilidades: (len(idx), 6) en conjunta, (len(idx),) en por_servicio."""
    import torch
    model.eval()
    ps = []
    with torch.no_grad():
        for i in range(0, len(idx), 1024):
            p = torch.sigmoid(model(entradas(G, idx[i:i + 1024], forma))).numpy()
            ps.append(p if forma == "conjunta" else p[:, 0])
    return np.concatenate(ps) if ps else np.zeros((0, S) if forma == "conjunta" else 0)


def a_filas(p, idx, F, M, N, forma):
    """Lleva las predicciones a las filas de ds; NaN donde no hay."""
    out = np.full(N, np.nan)
    if forma == "conjunta":
        r, m = F[idx], M[idx]
        out[r[m]] = p[m]
    else:
        out[F.reshape(-1)[idx]] = p
    return out


def objetivo(Y, M, idx, forma):
    if forma == "conjunta":
        return Y[idx], M[idx].astype(np.float32)
    return Y.reshape(-1)[idx][:, None], np.ones((len(idx), 1), dtype=np.float32)


def elegir_umbral(ds, prob, filas, beta):
    """Igual que entrenar.py: el umbral (0,05 a 0,95) que maximiza F-beta en la vista B de validacion."""
    mb = np.zeros(ds.N, dtype=bool)
    mb[filas] = ds.evalu[filas] == 1
    if mb.sum() == 0 or ds.y[mb].sum() == 0:
        return 0.5
    cands = []
    for t in np.arange(0.05, 0.96, 0.05):
        al = np.zeros(ds.N, dtype=bool)
        al[filas] = prob[filas] > t
        cands.append((float(t), D.prf(ds.y, al, mb, beta=beta)["fbeta"]))
    ok = [c for c in cands if c[1] == c[1]]
    return max(ok, key=lambda c: c[1])[0] if ok else 0.5


def entrenar_red(G, Y, M, tr, va, a):
    import torch
    import torch.nn as nn
    n_out = S if a.forma == "conjunta" else 1
    model = crear_red(G.shape[1] + (0 if a.forma == "conjunta" else S), a.hidden, n_out)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    ytr, mtr = objetivo(Y, M, tr, a.forma)
    pos = (ytr * mtr).sum(0)
    neg = ((1 - ytr) * mtr).sum(0)
    lf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / np.maximum(pos, 1.0), dtype=torch.float32),
                              reduction="none")
    ytr, mtr = torch.from_numpy(ytr), torch.from_numpy(mtr)
    yva, mva = objetivo(Y, M, va, a.forma)
    mva = mva.astype(bool)
    mejor, mejor_estado, sin_mejora = np.inf, None, 0
    for ep in range(a.epocas):
        model.train()
        perm = np.random.permutation(len(tr))
        tot, peso = 0.0, 0.0
        for i in range(0, len(perm), a.batch):
            sel = perm[i:i + a.batch]
            m = mtr[sel]
            loss = (lf(model(entradas(G, tr[sel], a.forma)), ytr[sel]) * m).sum() / m.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * float(m.sum())
            peso += float(m.sum())
        pv = predecir(model, G, va, a.forma).reshape(len(va), -1)
        pc = np.clip(pv[mva], 1e-6, 1 - 1e-6)
        yv = yva[mva]
        lv = float(-(yv * np.log(pc) + (1 - yv) * np.log(1 - pc)).mean())   # BCE sin ponderar, como v12
        print("    epoca %2d  loss_train=%.4f  val(loss)=%.4f" % (ep + 1, tot / max(peso, 1.0), lv), flush=True)
        if lv < mejor - 1e-4:
            mejor, sin_mejora = lv, 0
            mejor_estado = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= a.paciencia:
                break
    if mejor_estado is not None:
        model.load_state_dict(mejor_estado)
    return model, mejor


def predecir_corridas(ds, ruta):
    """Puntaje del modelo final guardado (models/conjunto_final.pt) para todas las filas de ds; NaN donde no hay
    ventana completa de los 6 servicios. Sirve para evaluar corridas que el modelo no vio."""
    import torch
    ck = torch.load(ruta, weights_only=False)
    D.W = ck["W"]
    forma = ck["forma"]
    F, _, pos = alinear(ds)
    M = ds.valid[F]
    Xc, _, _ = normalizar(ds, None, ck["mu"], ck["sd"])
    G = tensor_6(Xc, F)
    model = crear_red(G.shape[1] + (0 if forma == "conjunta" else S), ck["hidden"], S if forma == "conjunta" else 1)
    model.load_state_dict(ck["state"])
    idx = muestras(F, M, pos, forma)
    return a_filas(predecir(model, G, idx, forma), idx, F, M, ds.N, forma), ck["umbral"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--salida", required=True, help="subcarpeta de --data para los resultados")
    ap.add_argument("--forma", choices=["por_servicio", "conjunta"], default="por_servicio")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--seed-folds", type=int, default=42, help="el mismo reparto en folds que v12")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epocas", type=int, default=15)
    ap.add_argument("--paciencia", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=None, help="por defecto 64 (por_servicio, como v12) o 128 (conjunta)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--hilos", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="hilos de CPU (por defecto la mitad, para que la laptop siga usable)")
    ap.add_argument("--solo-fold", type=int, default=-1, help="correr solo ese fold (prueba corta)")
    ap.add_argument("--sin-final", action="store_true", help="no entrenar el modelo final con todas las corridas")
    a = ap.parse_args()
    if a.hidden is None:
        a.hidden = 128 if a.forma == "conjunta" else 64

    import torch
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.hilos)
    out = os.path.join(a.data, a.salida)
    os.makedirs(os.path.join(out, "models"), exist_ok=True)

    ds = D.cargar(a.data, "_propio", "y_30s")
    F, run_f, pos_f = alinear(ds)
    M = ds.valid[F]
    Y = ds.y[F].astype(np.float32)
    idx_all = muestras(F, M, pos_f, a.forma)
    run_m = run_f[segundo_de(idx_all, a.forma)]
    print("Forma %s | muestras validas v12: %d | muestras: %d | corridas: %d | %d unidades"
          % (a.forma, ds.valid.sum(), len(idx_all), len(set(run_f)), a.hidden), flush=True)

    fold_de = asignar_folds(ds, a.folds, a.seed_folds)
    prob = np.full(ds.N, np.nan)
    alarma = np.zeros(ds.N, dtype=bool)
    umbrales, val_loss = [], []
    t0 = time.time()
    for k in range(a.folds):
        if 0 <= a.solo_fold != k:
            umbrales.append(float("nan"))
            continue
        test = {r for r, f in fold_de.items() if f == k}
        train = [r for r, f in fold_de.items() if f != k]
        val = split_val(ds, train, a.seed_folds + k)
        es_te, es_va = np.isin(run_m, list(test)), np.isin(run_m, list(val))
        tr, va, te = idx_all[~es_te & ~es_va], idx_all[es_va], idx_all[es_te]
        print("\nFold %d/%d  train=%d corridas (%d muestras)  val=%d  test=%d (%d muestras)"
              % (k + 1, a.folds, len(train) - len(val), len(tr), len(val), len(test), len(te)), flush=True)
        Xc, _, _ = normalizar(ds, ~np.isin(ds.run, list(test)) & ~np.isin(ds.run, list(val)))
        G = tensor_6(Xc, F)
        model, lv = entrenar_red(G, Y, M, tr, va, a)
        val_loss.append(lv)
        pv = a_filas(predecir(model, G, va, a.forma), va, F, M, ds.N, a.forma)
        thr = elegir_umbral(ds, pv, np.where(np.isfinite(pv))[0], a.beta)
        pt = a_filas(predecir(model, G, te, a.forma), te, F, M, ds.N, a.forma)
        ok = np.isfinite(pt)
        prob[ok], alarma[ok] = pt[ok], pt[ok] > thr
        umbrales.append(thr)
        mb = ok & (ds.evalu == 1)
        for nombre, al in (("v13", alarma), ("CPU > 80 %", D.alarma_cpu(ds) & ok)):
            m = D.prf(ds.y, al, mb)
            print("    %-10s P %.3f  R %.3f  F1 %.3f  falsas/h %.2f" % (nombre, m["precision"], m["recall"], m["f1"],
                                                                         D.falsas_alarmas_hora(ds, al, mb)))
        print("    mejor val(loss) %.4f | umbral elegido en validacion: %.2f" % (lv, thr), flush=True)
    print("\nValidacion cruzada: %.0f s" % (time.time() - t0))

    cub = np.isfinite(prob)
    pd.DataFrame({"run": ds.run, "servicio": ds.srv, "t_rel": ds.sec, "prob_lstm": prob, "alarma_lstm": alarma,
                  "cpu": ds.cpu, "bn_ahora": ds.bn, "y_30s": ds.y_30s, "y_objetivo": ds.y,
                  "evaluable": ds.evalu})[cub].to_csv(os.path.join(out, "predicciones.csv"), index=False)
    param = dict(vars(a), sufijo="_propio", objetivo="y_30s", ventana=D.W)
    json.dump({"umbrales_por_fold": umbrales, "val_loss_por_fold": val_loss, "parametros": param, "excluidas": []},
              open(os.path.join(out, "config.json"), "w"), indent=2)

    if not a.sin_final and a.solo_fold < 0:
        print("\nEntrenando el modelo final con todas las corridas...", flush=True)
        val = split_val(ds, sorted(set(ds.run)), a.seed + 99)
        es_va = np.isin(run_m, list(val))
        Xc, mu, sd = normalizar(ds, ~np.isin(ds.run, list(val)))
        model, _ = entrenar_red(tensor_6(Xc, F), Y, M, idx_all[~es_va], idx_all[es_va], a)
        torch.save({"state": model.state_dict(), "mu": np.asarray(mu).tolist(), "sd": np.asarray(sd).tolist(),
                    "cols": COLS, "servicios": D.SERVICIOS, "hidden": a.hidden, "W": D.W, "forma": a.forma,
                    "umbral": float(np.nanmedian(umbrales))},
                   os.path.join(out, "models", "conjunto_final.pt"))
        print("Guardado: %s (umbral = mediana de los folds = %.2f)"
              % (os.path.join(out, "models", "conjunto_final.pt"), np.nanmedian(umbrales)))
    print("\nResultados en %s" % out)


if __name__ == "__main__":
    main()
