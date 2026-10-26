#!/bin/bash
# Caidas en cadena: carga que crece o rafagas + fallas, y varias fallas a la vez.
#   campana (rnd_combo_prog, rnd_combo_burst, rnd_simultanea: 30 corridas, ~4 h)
#   -> etiquetado -> v11 en las corridas nuevas que nunca vio (prueba honesta)
#   -> v12 (config de v11) con todas las corridas, 3 semillas, mismos folds -> tablas
# Uso (desde la raiz, Docker arriba, CARGADOR conectado, laptop libre): bash ml/pipeline_v12.sh
cd "$(dirname "$0")/.." || exit 1
L=data/pipeline_v12.log
paso() { echo "[$(date +%H:%M)] $*" | tee -a $L; }
GRUPOS=rnd_combo_prog,rnd_combo_burst,rnd_simultanea

paso "1/5 campana de caidas en cadena ($GRUPOS)"
python -u run_campaign.py --only $GRUPOS >> data/campana_v12.log 2>&1 || { paso "FALLO la campana"; exit 1; }
paso "   segunda pasada: repite las que no pasaron el control de calidad"
python -u run_campaign.py --only $GRUPOS >> data/campana_v12.log 2>&1 || { paso "FALLO la segunda pasada"; exit 1; }
docker compose stop > /dev/null 2>&1

paso "2/5 etiquetado"
python ml/etiquetar.py --solo-propio --sufijo _propio > data/etiquetar_v12_propio.log 2>&1 || { paso "FALLO etiquetar"; exit 1; }

paso "3/5 prueba de fuego: v11 en las corridas nuevas que nunca vio"
python ml/evaluar_externo.py --modelo data/resultados_v11/modelo_ensamble.json --nuevas-para resultados_v11 \
  --nombre v11_nuevas > data/externo_v11_nuevas.log 2>&1

paso "4/5 entrenamiento de v12"
BASE="--sufijo _propio --contexto --tendencias --seed-folds 42"
for s in 1 2 3; do
  python -u ml/entrenar.py $BASE --seed $s --salida resultados_v12/s$s > data/entrenar_v12_s$s.log 2>&1 \
    || { paso "FALLO v12 semilla $s"; exit 1; }
  paso "   v12 semilla $s lista"
done
python ml/ensamble.py --dirs resultados_v12/s1 resultados_v12/s2 resultados_v12/s3 --salida resultados_v12 \
  --salida-csv resultados_v12/ensamble_metricas.csv > data/ensamble_v12.log 2>&1
python ml/evaluar_externo.py --resultados resultados_v12 --nombre v12 > data/externo_v12.log 2>&1

paso "5/5 tablas por escenario"
python ml/por_escenario.py --resultados resultados_v12 > data/por_escenario_v12.log 2>&1
paso "LISTO"
