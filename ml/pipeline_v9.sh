#!/bin/bash
# Cadena completa de v9 (calendario aleatorio). Retomable: la campana salta las corridas con OK.
# Uso, desde la raiz del proyecto con Docker Desktop abierto y "docker compose up -d":
#   bash ml/pipeline_v9.sh
# Pasos: campana rnd_* (40 corridas, ~4.9 h) -> etiquetado -> v8 en corridas nuevas -> v9 (3 semillas,
# 120 corridas) -> ensamble y tablas -> ventana de 60 s con calendario aleatorio. Registro: data/pipeline_v9.log
cd "C:/Users/Personal/Desktop/boleteria_nube" || exit 1
L=data/pipeline_v9.log
paso() { echo "[$(date +%H:%M)] $*" | tee -a $L; }

paso "1/7 campana aleatoria (40 corridas)"
python -u run_campaign.py --only rnd_normal,rnd_burst,rnd_progresivo >> data/campana_rnd.log 2>&1 || { paso "FALLO la campana"; exit 1; }
docker compose stop > /dev/null 2>&1

paso "2/7 etiquetado (cuello de botella propio, 120 corridas)"
python ml/etiquetar.py --solo-propio --sufijo _propio > data/etiquetar_v9_propio.log 2>&1 || { paso "FALLO etiquetar"; exit 1; }

paso "3/7 prueba de fuego: v8 en las corridas aleatorias que nunca vio"
python ml/evaluar_externo.py --modelo data/resultados_v8/modelo_ensamble.json --nombre v8 > data/externo_v8.log 2>&1

paso "4/7 v9: 3 LSTM con las 120 corridas"
for s in 1 2 3; do
  python -u ml/entrenar.py --sufijo _propio --contexto --tendencias --seed-folds 42 --seed $s --salida resultados_v9/s$s > data/entrenar_v9_s$s.log 2>&1 || { paso "FALLO v9 semilla $s"; exit 1; }
  paso "   v9 semilla $s lista"
done

paso "5/7 ensamble v9 y tablas"
python ml/ensamble.py --dirs resultados_v9/s1 resultados_v9/s2 resultados_v9/s3 --salida resultados_v9 --salida-csv resultados_v9/ensamble_metricas.csv > data/ensamble_v9.log 2>&1
python ml/por_escenario.py --resultados resultados_v9 > data/por_escenario_v9.log 2>&1
python ml/postproceso.py --sufijo _propio --resultados resultados_v9 > data/postproceso_v9.log 2>&1
python ml/evaluar_externo.py --resultados resultados_v9 --nombre v9 > data/externo_v9.log 2>&1

paso "6/7 ventana de 60 s con calendario aleatorio (para ver si ahora si ayuda)"
for s in 1 2 3; do
  python -u ml/entrenar.py --sufijo _propio --contexto --tendencias --seed-folds 42 --seed $s --ventana 60 --sin-final --salida resultados_v9w60/s$s > data/entrenar_v9w60_s$s.log 2>&1 || { paso "FALLO ventana 60 semilla $s"; break; }
done
python ml/ensamble.py --dirs resultados_v9w60/s1 resultados_v9w60/s2 resultados_v9w60/s3 --salida resultados_v9w60 > data/ensamble_v9w60.log 2>&1
python ml/evaluar_externo.py --resultados resultados_v9w60 --nombre v9_ventana60 > data/externo_v9w60.log 2>&1

paso "7/7 LISTO"
