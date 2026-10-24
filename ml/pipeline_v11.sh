#!/bin/bash
# Mas datos de calendario aleatorio y reentrenamiento:
#   campana (rnd_progresivo hasta 12, rnd_rampa: rafagas que suben en 10-120 s, rnd_lenta: fallas de 2-5 min)
#   -> etiquetado -> v9 en las corridas nuevas que nunca vio (prueba honesta)
#   -> v11 (config de v9), v11l (+ --largo) y v11lx (--largo sin las corridas de calendario fijo, que
#      contienen el atajo "despues de una falla viene otra"), 3 semillas cada uno, mismos folds.
# Uso (desde la raiz, Docker arriba, CARGADOR conectado, laptop libre): bash ml/pipeline_v11.sh
cd "$(dirname "$0")/.." || exit 1
L=data/pipeline_v11.log
paso() { echo "[$(date +%H:%M)] $*" | tee -a $L; }
GRUPOS=rnd_progresivo,rnd_rampa,rnd_lenta

paso "1/5 campana aleatoria ($GRUPOS)"
python -u run_campaign.py --only $GRUPOS --reps 2 >> data/campana_v11.log 2>&1 || { paso "FALLO la campana"; exit 1; }
paso "   segunda pasada: repite las que no pasaron el control de calidad"
python -u run_campaign.py --only $GRUPOS --reps 2 >> data/campana_v11.log 2>&1 || { paso "FALLO la segunda pasada"; exit 1; }
docker compose stop > /dev/null 2>&1

paso "2/5 etiquetado"
python ml/etiquetar.py --solo-propio --sufijo _propio > data/etiquetar_v11_propio.log 2>&1 || { paso "FALLO etiquetar"; exit 1; }

paso "3/5 prueba de fuego: v9 en las corridas nuevas que nunca vio"
python ml/evaluar_externo.py --modelo data/resultados_v9/modelo_ensamble.json --nuevas-para resultados_v9 \
  --nombre v9_nuevas > data/externo_v9_nuevas.log 2>&1

paso "4/5 entrenamiento"
BASE="--sufijo _propio --contexto --tendencias --seed-folds 42"
FIJOS=normal_inj,burst_inj,var_inj,suave_inj
for v in "v11:" "v11l:--largo" "v11lx:--largo --excluir-grupos $FIJOS"; do
  n=${v%%:*}; extra=${v#*:}
  paso "   $n ($extra)"
  for s in 1 2 3; do
    python -u ml/entrenar.py $BASE $extra --seed $s --salida resultados_$n/s$s > data/entrenar_${n}_s$s.log 2>&1 \
      || { paso "FALLO $n semilla $s"; exit 1; }
    paso "      $n semilla $s lista"
  done
  python ml/ensamble.py --dirs resultados_$n/s1 resultados_$n/s2 resultados_$n/s3 --salida resultados_$n \
    --salida-csv resultados_$n/ensamble_metricas.csv > data/ensamble_$n.log 2>&1
  python ml/evaluar_externo.py --resultados resultados_$n --nombre $n > data/externo_$n.log 2>&1
done

paso "5/5 tablas por escenario"
for n in v11 v11l v11lx; do python ml/por_escenario.py --resultados resultados_$n > data/por_escenario_$n.log 2>&1; done
paso "LISTO"
