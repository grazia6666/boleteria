#!/bin/bash
# v14: senales de tiempo propio (datos.PROPIO) + 29 corridas mas de fallas en carga normal y progresiva.
#   etiquetado con sufijo propio (no toca el dataset de v12)
#   -> v12 reentrenado con las 204 corridas (referencia justa) y v14 (= v12 + --propio), 3 semillas cada uno,
#      mismos folds -> ensambles
# Uso (desde la raiz, despues de la campana; NO grabar corridas mientras corre): bash ml/pipeline_v14.sh
cd "$(dirname "$0")/.." || exit 1
L=data/pipeline_v14.log
paso() { echo "[$(date +%H:%M)] $*" | tee -a $L; }
SUF=_propio_v14

paso "1/3 etiquetado ($SUF)"
python ml/etiquetar.py --solo-propio --sufijo $SUF > data/etiquetar_v14.log 2>&1 || { paso "FALLO etiquetar"; exit 1; }

BASE="--sufijo $SUF --contexto --tendencias --seed-folds 42"
for variante in v14base v14; do
  extra=""
  [ "$variante" = "v14" ] && extra="--propio"
  paso "2/3 entrenamiento $variante"
  for s in 1 2 3; do
    python -u ml/entrenar.py $BASE $extra --seed $s --salida resultados_$variante/s$s > data/entrenar_${variante}_s$s.log 2>&1 \
      || { paso "FALLO $variante semilla $s"; exit 1; }
    paso "   $variante semilla $s lista"
  done
  python ml/ensamble.py --dirs resultados_$variante/s1 resultados_$variante/s2 resultados_$variante/s3 \
    --salida resultados_$variante --salida-csv resultados_$variante/ensamble_metricas.csv > data/ensamble_$variante.log 2>&1 \
    || { paso "FALLO ensamble $variante"; exit 1; }
done
paso "3/3 LISTO"
