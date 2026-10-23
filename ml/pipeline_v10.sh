#!/bin/bash
# v10: tendencias largas (--largo) y/o mas peso a los avisos tempranos (--peso-temprano), contra v9.
# Mismos folds que v9 (--seed-folds 42), 3 semillas por variante, ensamble y prueba en las corridas rnd_*.
# Uso (desde la raiz, sin campana activa): bash ml/pipeline_v10.sh   -> registro en data/pipeline_v10.log
cd "$(dirname "$0")/.." || exit 1
L=data/pipeline_v10.log
paso() { echo "[$(date +%H:%M)] $*" | tee -a $L; }
BASE="--sufijo _propio --contexto --tendencias --seed-folds 42"

for v in "v10a:--largo" "v10b:--largo --peso-temprano 3" "v10c:--peso-temprano 3"; do
  n=${v%%:*}; extra=${v#*:}
  paso "$n ($extra)"
  for s in 1 2 3; do
    python -u ml/entrenar.py $BASE $extra --seed $s --salida resultados_$n/s$s > data/entrenar_${n}_s$s.log 2>&1 \
      || { paso "FALLO $n semilla $s"; exit 1; }
    paso "   $n semilla $s lista"
  done
  python ml/ensamble.py --dirs resultados_$n/s1 resultados_$n/s2 resultados_$n/s3 --salida resultados_$n \
    --salida-csv resultados_$n/ensamble_metricas.csv > data/ensamble_$n.log 2>&1
  python ml/evaluar_externo.py --resultados resultados_$n --nombre $n > data/externo_$n.log 2>&1
done
paso "LISTO"
