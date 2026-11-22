#!/bin/bash
# v15: el freno de CPU de Docker como entrada (datos.FRENO), sobre la campana data_thr/ (88 corridas grabadas con el
# recolector nuevo). Para separar el efecto de la senal del de las corridas, se entrenan sobre LAS MISMAS corridas:
#   thr_base = configuracion de v12 (contexto + tendencias)      thr_freno = la misma + --freno
# 3 semillas cada uno, mismos folds -> ensambles. Despues: python ml/comparar_freno.py
# Uso (desde la raiz, despues de la campana; NO grabar corridas mientras corre): bash ml/pipeline_thr.sh
cd "$(dirname "$0")/.." || exit 1
DATA=data_thr
L=data/pipeline_thr.log
paso() { echo "[$(date +%H:%M)] $*" | tee -a $L; }

paso "1/3 etiquetado de $DATA"
python ml/etiquetar.py --data $DATA --solo-propio --sufijo _propio > data/etiquetar_thr.log 2>&1 || { paso "FALLO etiquetar"; exit 1; }

BASE="--data $DATA --sufijo _propio --contexto --tendencias --seed-folds 42"
for variante in thr_base thr_freno; do
  extra=""
  [ "$variante" = "thr_freno" ] && extra="--freno"
  paso "2/3 entrenamiento $variante"
  for s in 1 2 3; do
    python -u ml/entrenar.py $BASE $extra --seed $s --salida resultados_$variante/s$s > data/entrenar_${variante}_s$s.log 2>&1 \
      || { paso "FALLO $variante semilla $s"; exit 1; }
    paso "   $variante semilla $s lista"
  done
  python ml/ensamble.py --data $DATA --dirs resultados_$variante/s1 resultados_$variante/s2 resultados_$variante/s3 \
    --salida resultados_$variante --salida-csv resultados_$variante/ensamble_metricas.csv > data/ensamble_$variante.log 2>&1 \
    || { paso "FALLO ensamble $variante"; exit 1; }
done
paso "3/3 LISTO"
