#!/bin/bash
# Ablacion pedida en la retroalimentacion: el MISMO LSTM de v12 con y sin las entradas de dependencias.
#   v12 (ya entrenado)  = --contexto --tendencias
#   v12_sindep          = --tendencias          (sin cpu_dep, cola_dep, lat_p95_dep, d5_cpu_dep, d15_cpu_dep)
# Mismas 175 corridas de data/, mismas semillas (1-3), mismos folds (--seed-folds 42) -> ensamble.
# Replica en data_thr/ (88 corridas grabadas aparte): thr_base (ya entrenado) contra thr_sindep.
# Despues: python ml/comparar_dependencias.py
# Uso (desde la raiz; no grabar corridas mientras corre): bash ml/pipeline_ablacion_dep.sh
cd "$(dirname "$0")/.." || exit 1
L=data/pipeline_ablacion_dep.log
paso() { echo "[$(date +%H:%M)] $*" | tee -a $L; }

for DATA in data data_thr; do
  if [ "$DATA" = data ]; then V=v12_sindep; OPT=""; else V=thr_sindep; OPT="--data data_thr"; fi
  paso "entrenamiento $V ($DATA)"
  for s in 1 2 3; do
    python -u ml/entrenar.py $OPT --sufijo _propio --tendencias --seed-folds 42 --seed $s --salida resultados_$V/s$s \
      > data/entrenar_${V}_s$s.log 2>&1 || { paso "FALLO $V semilla $s"; exit 1; }
    paso "   $V semilla $s lista"
  done
  python ml/ensamble.py $OPT --dirs resultados_$V/s1 resultados_$V/s2 resultados_$V/s3 --salida resultados_$V \
    --salida-csv resultados_$V/ensamble_metricas.csv > data/ensamble_$V.log 2>&1 || { paso "FALLO ensamble $V"; exit 1; }
done
paso "LISTO"
