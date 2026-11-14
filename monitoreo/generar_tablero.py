"""Genera el tablero de Grafana (monitoreo/grafana/dashboards/boleteria.json), armado como la pantalla de
python ml/demo.py: una tabla con todos los servicios y los eventos narrados.
Uso (desde la raiz): python monitoreo/generar_tablero.py [--salida RUTA]   (Grafana lo recarga solo en ~10 s)

  1. Resumen: t de carga, si el modelo ya predice, cuantos servicios estan en cuello de botella y en cuantos
     predice el modelo.
  2. Ahora: un renglon por servicio (en el orden de la cadena) con su estado en color, la probabilidad en barra,
     CPU, regla CPU > 80 %, p95, cola, hace cuanto se encendio la alarma y que aviso en su ultimo cuello de botella.
  3. Eventos de esta carga (los mismos de la demo) | probabilidad en el tiempo.
  4. Estado en el tiempo.
  5. Detalle del sistema (plegado): latencia, CPU, espera a otros servicios, cola, trafico, errores, historial y
     alertas de Prometheus."""
import argparse
import json
import os

AQUI = os.path.dirname(os.path.abspath(__file__))
DS = {"type": "prometheus", "uid": "prometheus"}
J = '{job="$job"}'
CADENA = ["reservaciones", "pago", "ticket", "notificaciones", "usuarios", "eventos"]
COLOR = {"ticket": "#1F5FBF", "pago": "#008300", "reservaciones": "#6250D6", "notificaciones": "#1BAF7A",
         "usuarios": "#EDA100", "eventos": "#E87BA4"}
VERDE, MORADO, NARANJA, ROJO, GRIS = "#2F7D4F", "#7B5FC0", "#E07B1F", "#C0322B", "#5B6475"
ESTADOS = [(0, "normal", VERDE), (1, "↪ espera a otro", MORADO), (2, "⚠ EL MODELO PREDICE", NARANJA),
           (3, "✖ CUELLO DE BOTELLA", ROJO)]
MAPEO_ESTADO = [{"type": "value", "options": {str(v): {"text": t, "color": c, "index": v}}} for v, t, c in ESTADOS]
SIN_VALOR = [{"type": "special", "options": {"match": "nan", "result": {"text": "—", "index": 20}}},
             {"type": "special", "options": {"match": "null", "result": {"text": "—", "index": 21}}}]
TIPOS = [("aviso", "⚠ el modelo avisa", NARANJA), ("cuello", "✖ cuello de botella", ROJO),
         ("recupera", "✓ se recuperó", VERDE), ("espera", "↪ espera a otro", MORADO), ("otro", "·", GRIS)]


def umbrales(*pasos):
    return {"mode": "absolute", "steps": [{"color": c, "value": v} for v, c in pasos]}


# el color del estado sale del propio numero (0..3): mas robusto que el color de los mapeos
UMBRAL_ESTADO = umbrales((None, VERDE), (1, MORADO), (2, NARANJA), (3, ROJO))


def instantanea(ref, expr):
    return {"datasource": DS, "refId": ref, "expr": expr, "format": "table", "instant": True}


def columna(nombre, *props):
    return {"matcher": {"id": "byName", "options": nombre}, "properties": list(props)}


def prop(i, v):
    return {"id": i, "value": v}


def serie(titulo, expr, x, y, w, h, unidad="none", leyenda="{{servicio}}", umbral=None, minimo=None, maximo=None,
          log=False, descripcion=""):
    custom = {"lineWidth": 2, "fillOpacity": 0, "showPoints": "never", "spanNulls": True,
              "thresholdsStyle": {"mode": "dashed" if umbral is not None else "off"}}
    if log:
        custom["scaleDistribution"] = {"type": "log", "log": 10}
    pasos = [(None, "transparent")] + ([(umbral, ROJO)] if umbral is not None else [])
    defaults = {"unit": unidad, "custom": custom, "thresholds": umbrales(*pasos), "color": {"mode": "palette-classic"}}
    if minimo is not None:
        defaults["min"] = minimo
    if maximo is not None:
        defaults["max"] = maximo
    return {"type": "timeseries", "title": titulo, "description": descripcion, "datasource": DS,
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [{"datasource": DS, "refId": "A", "expr": expr, "legendFormat": leyenda}],
            "fieldConfig": {"defaults": defaults, "overrides": [
                columna(s, prop("color", {"mode": "fixed", "fixedColor": c})) for s, c in COLOR.items()]},
            "options": {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                        "tooltip": {"mode": "multi", "sort": "desc"}}}


def estadistica(titulo, expr, x, w, mapeos, pasos, unidad="none", sin_valor="sin demo"):
    return {"type": "stat", "title": titulo, "datasource": DS, "gridPos": {"x": x, "y": 0, "w": w, "h": 4},
            "targets": [{"datasource": DS, "refId": "A", "expr": expr, "instant": True}],
            "fieldConfig": {"defaults": {"unit": unidad, "mappings": mapeos, "noValue": sin_valor,
                                         "color": {"mode": "thresholds"}, "thresholds": umbrales(*pasos)},
                            "overrides": []},
            "options": {"colorMode": "background", "graphMode": "none", "textMode": "value",
                        "justifyMode": "center", "text": {"valueSize": 30},
                        "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False}}}


AHORA = [("A", "max by (servicio) (boleteria_orden%s)"), ("B", "max by (servicio) (boleteria_estado%s)"),
         ("C", "max by (servicio) (boleteria_probabilidad%s)"), ("D", "max by (servicio) (boleteria_cpu%s)"),
         ("E", "max by (servicio) (boleteria_alarma_cpu%s)"), ("F", "max by (servicio) (boleteria_p95_ms%s)"),
         ("G", "max by (servicio) (boleteria_cola%s)"), ("H", "max by (servicio) (boleteria_alarma_hace_s%s)"),
         ("I", "max by (servicio, causa) (boleteria_causa%s)"),
         ("K", "max by (servicio, modelo) (boleteria_ultimo_aviso%s)")]
NOMBRES = {"Value #A": "orden", "servicio": "Servicio", "Value #B": "Estado", "Value #C": "P(cuello en 30 s)",
           "Value #D": "CPU", "Value #E": "Regla CPU > 80 %", "Value #F": "p95", "Value #G": "Cola",
           "Value #H": "Alarma encendida hace", "causa": "Espera a", "modelo": "Último cuello: el modelo"}

paneles = [
    # 1. resumen
    estadistica("Segundos de carga", "boleteria_t_carga" + J, 0, 5, [], [(None, GRIS)], unidad="suffix: s"),
    estadistica("Modelo", "boleteria_calentando" + J, 5, 6,
                [{"type": "value", "options": {"0": {"text": "prediciendo", "color": VERDE, "index": 0}}},
                 {"type": "range", "options": {"from": 1, "to": 1000,
                                               "result": {"text": "calentando", "color": NARANJA, "index": 1}}}],
                [(None, GRIS)]),
    estadistica("Servicios en cuello de botella", "count(boleteria_estado%s == 3) or vector(0)" % J, 11, 7, [],
                [(None, VERDE), (1, ROJO)], sin_valor="0"),
    estadistica("Servicios donde el modelo predice", "count(boleteria_estado%s == 2) or vector(0)" % J, 18, 6, [],
                [(None, VERDE), (1, NARANJA)], sin_valor="0"),
    # 2. ahora
    {"type": "table", "title": "Ahora: cada servicio, en el orden de la cadena de llamadas", "datasource": DS,
     "description": "Estado: verde normal; naranja el modelo predice un cuello de botella en los proximos 30 s; rojo "
                    "cuello de botella real; morado en alarma pero espera a otro servicio (el aviso queda en la "
                    "causa). Alarma encendida hace: segundos desde que el modelo encendio la alarma. Ultimo cuello: "
                    "cuanto antes aviso el modelo la ultima vez que el servicio entro en cuello de botella.",
     "gridPos": {"x": 0, "y": 4, "w": 24, "h": 8},
     "targets": [instantanea(r, e % J) for r, e in AHORA],
     "transformations": [
         {"id": "joinByField", "options": {"byField": "servicio", "mode": "outer"}},
         {"id": "filterFieldsByName", "options": {"include": {
             "pattern": "^(servicio|Value #[A-H]|causa|modelo)$"}}},
         {"id": "organize", "options": {
             "indexByName": {"Value #A": 0, "servicio": 1, "Value #B": 2, "Value #C": 3, "Value #D": 4,
                             "Value #E": 5, "Value #F": 6, "Value #G": 7, "Value #H": 8, "causa": 9, "modelo": 10},
             "renameByName": NOMBRES}},
         {"id": "sortBy", "options": {"sort": [{"field": "orden", "desc": False}]}}],
     "fieldConfig": {"defaults": {"custom": {"align": "left", "cellOptions": {"type": "auto"}},
                                  "mappings": SIN_VALOR}, "overrides": [
         columna("orden", prop("custom.hidden", True)),
         columna("Servicio", prop("custom.width", 150)),
         columna("Estado", prop("mappings", MAPEO_ESTADO), prop("custom.cellOptions", {"type": "color-background"}),
                 prop("color", {"mode": "thresholds"}), prop("thresholds", UMBRAL_ESTADO), prop("custom.width", 220)),
         columna("P(cuello en 30 s)", prop("unit", "percentunit"), prop("min", 0), prop("max", 1),
                 prop("decimals", 0), prop("custom.width", 260),
                 prop("custom.cellOptions", {"type": "gauge", "mode": "basic", "valueDisplayMode": "text"}),
                 prop("color", {"mode": "thresholds"}),
                 prop("thresholds", umbrales((None, VERDE), (0.255, NARANJA), (0.51, ROJO)))),
         columna("CPU", prop("unit", "percent"), prop("decimals", 0), prop("custom.width", 80),
                 prop("custom.cellOptions", {"type": "color-text"}), prop("color", {"mode": "thresholds"}),
                 prop("thresholds", umbrales((None, "text"), (80, ROJO)))),
         columna("Regla CPU > 80 %", prop("custom.width", 150),
                 prop("mappings", [{"type": "value", "options": {
                     "0": {"text": "ok", "color": VERDE, "index": 0},
                     "1": {"text": "⚠ ALARMA", "color": ROJO, "index": 1}}}]),
                 prop("custom.cellOptions", {"type": "color-text"})),
         columna("p95", prop("unit", "ms"), prop("decimals", 0), prop("custom.width", 90)),
         columna("Cola", prop("decimals", 0), prop("custom.width", 70)),
         columna("Alarma encendida hace", prop("unit", "s"), prop("decimals", 0), prop("custom.width", 170),
                 prop("mappings", SIN_VALOR)),
         columna("Espera a", prop("custom.width", 130), prop("custom.cellOptions", {"type": "color-text"}),
                 prop("color", {"mode": "fixed", "fixedColor": MORADO})),
         columna("Último cuello: el modelo", prop("custom.cellOptions", {"type": "color-text"}),
                 prop("color", {"mode": "fixed", "fixedColor": "#1F5FBF"}))]},
     "options": {"showHeader": True, "cellHeight": "sm", "footer": {"show": False}}},
    # 3. eventos y probabilidad
    {"type": "table", "title": "Eventos de esta carga (el más reciente arriba)", "datasource": DS,
     "description": "Los mismos mensajes de la pantalla de python ml/demo.py.",
     "gridPos": {"x": 0, "y": 12, "w": 12, "h": 11},
     "targets": [instantanea("A", "last_over_time(boleteria_evento%s[30m]) * on(sesion) group_left() "
                                  "boleteria_sesion%s" % (J, J))],
     "transformations": [
         {"id": "organize", "options": {
             "excludeByName": {"Time": True, "instance": True, "job": True, "__name__": True, "n": True,
                               "sesion": True},
             "indexByName": {"Value": 0, "tipo": 1, "texto": 2},
             "renameByName": {"Value": "t", "tipo": "Qué pasó", "texto": "Detalle"}}},
         {"id": "sortBy", "options": {"sort": [{"field": "t", "desc": True}]}}],
     "fieldConfig": {"defaults": {"custom": {"align": "left"}}, "overrides": [
         columna("t", prop("unit", "suffix: s"), prop("decimals", 0), prop("custom.width", 70)),
         columna("Detalle", prop("custom.cellOptions", {"type": "auto", "wrapText": True})),
         columna("Qué pasó", prop("custom.width", 175), prop("custom.cellOptions", {"type": "color-background"}),
                 prop("mappings", [{"type": "value", "options": {k: {"text": t, "color": c, "index": i}}}
                                   for i, (k, t, c) in enumerate(TIPOS)]))]},
     "options": {"showHeader": True, "cellHeight": "sm", "footer": {"show": False}}},
    serie("Probabilidad de cuello de botella en los próximos 30 s", "boleteria_probabilidad" + J, 12, 12, 12, 11,
          unidad="percentunit", umbral=0.51, minimo=0, maximo=1,
          descripcion="Probabilidad calibrada. La linea roja punteada es el umbral de la alarma (0,51)."),
    # 4. estado en el tiempo
    {"type": "state-timeline",
     "title": "Estado en el tiempo · verde normal · naranja el modelo predice · rojo cuello de botella · morado "
              "espera a otro", "datasource": DS,
     "gridPos": {"x": 0, "y": 23, "w": 24, "h": 8},
     "targets": [{"datasource": DS, "refId": chr(65 + i), "expr": 'boleteria_estado{job="$job",servicio="%s"}' % s,
                  "legendFormat": s} for i, s in enumerate(CADENA)],
     "fieldConfig": {"defaults": {"custom": {"fillOpacity": 85, "lineWidth": 0}, "color": {"mode": "thresholds"},
                                  "thresholds": UMBRAL_ESTADO, "mappings": MAPEO_ESTADO}, "overrides": []},
     "options": {"showValue": "never", "mergeValues": True, "rowHeight": 0.85, "alignValue": "left",
                 "legend": {"displayMode": "list", "placement": "bottom", "showLegend": False},
                 "tooltip": {"mode": "single"}}},
]
detalle = [
    {"type": "table", "title": "Historial: cada cuello de botella de esta carga y cuánto antes avisó cada método",
     "datasource": DS, "gridPos": {"x": 0, "y": 33, "w": 14, "h": 9},
     "targets": [instantanea("A", "last_over_time(boleteria_episodio%s[1h]) * on(sesion) group_left() "
                                  "boleteria_sesion%s" % (J, J))],
     "transformations": [{"id": "organize", "options": {
         "excludeByName": {"Time": True, "instance": True, "job": True, "__name__": True, "n": True, "sesion": True},
         "indexByName": {"Value": 0, "servicio": 1, "modelo": 2, "cpu": 3},
         "renameByName": {"Value": "Empezó en", "servicio": "Servicio", "modelo": "Modelo", "cpu": "CPU > 80 %"}}},
         {"id": "sortBy", "options": {"sort": [{"field": "Empezó en", "desc": True}]}}],
     "fieldConfig": {"defaults": {"custom": {"align": "left"}}, "overrides": [
         columna("Empezó en", prop("unit", "s"), prop("decimals", 0), prop("custom.width", 110))]},
     "options": {"showHeader": True, "cellHeight": "sm", "footer": {"show": False}}},
    {"type": "table", "title": "Alertas activas (Prometheus)", "datasource": DS,
     "gridPos": {"x": 14, "y": 33, "w": 10, "h": 9},
     "targets": [instantanea("A", 'ALERTS{alertstate="firing"}')],
     "transformations": [{"id": "organize", "options": {
         "excludeByName": {"Time": True, "Value": True, "alertstate": True, "__name__": True, "job": True,
                           "instance": True},
         "indexByName": {"alertname": 0, "servicio": 1, "causa": 2, "severidad": 3},
         "renameByName": {"alertname": "Alerta", "servicio": "Servicio", "causa": "Espera a",
                          "severidad": "Severidad"}}}],
     "fieldConfig": {"defaults": {"custom": {"align": "left"}}, "overrides": []},
     "options": {"showHeader": True, "cellHeight": "sm", "footer": {"show": False}}},
    serie("Latencia p95 (ms)",
          'boleteria_p95_ms{job="$job"} or on(servicio) (1000 * histogram_quantile(0.95, sum by (le, servicio) '
          "(rate(http_request_duration_seconds_bucket[5s]))))", 0, 42, 12, 8, unidad="ms", log=True),
    serie("CPU por servicio", "boleteria_cpu" + J, 12, 42, 12, 8, unidad="percent", umbral=80, minimo=0, maximo=110),
    serie("Tiempo esperando a otros servicios (% del tiempo de cada petición)",
          'clamp_max(100 * sum by (service) (rate(downstream_request_duration_seconds_sum[5s]) / '
          'rate(downstream_request_duration_seconds_count[5s])) / (sum by (service) '
          '(rate(http_request_duration_seconds_sum[5s])) / sum by (service) '
          '(rate(http_request_duration_seconds_count[5s]))), 100)',
          0, 50, 12, 8, unidad="percent", leyenda="{{service}}", minimo=0, maximo=100,
          descripcion="Cerca de 100 %: lento porque espera al que llama. Bajo: su lentitud es trabajo propio."),
    serie("Peticiones en curso (cola)", 'boleteria_cola{job="$job"} or on(servicio) http_requests_in_flight',
          12, 50, 12, 8),
    serie("Tráfico: peticiones por segundo", "sum by (service) (rate(http_requests_total[5s]))", 0, 58, 12, 8,
          unidad="reqps", leyenda="{{service}}"),
    serie("Errores por segundo (5xx)", 'sum by (service) (rate(http_requests_total{status=~"5.."}[5s]))',
          12, 58, 12, 8, unidad="reqps", leyenda="{{service}}"),
]
paneles.append({"type": "row", "title": "Detalle del sistema", "collapsed": True,
                "gridPos": {"x": 0, "y": 31, "w": 24, "h": 1}, "panels": detalle})
n = 1
for p in paneles + detalle:
    p["id"] = n
    n += 1

tablero = {"uid": "boleteria", "title": "Boletería: ¿se está muriendo algo?", "tags": ["boleteria"],
           "timezone": "browser", "refresh": "1s", "schemaVersion": 39, "version": 4, "editable": True,
           "time": {"from": "now-10m", "to": "now"},
           "timepicker": {"refresh_intervals": ["1s", "5s", "10s", "30s"]},
           "templating": {"list": [{"name": "job", "type": "custom", "hide": 2, "query": "modelo,modelo_prueba",
                                    "current": {"text": "modelo", "value": "modelo"},
                                    "options": [{"text": "modelo", "value": "modelo", "selected": True},
                                                {"text": "modelo_prueba", "value": "modelo_prueba",
                                                 "selected": False}]}]},
           "panels": paneles, "annotations": {"list": []}}
ap = argparse.ArgumentParser()
ap.add_argument("--salida", default=os.path.join(AQUI, "grafana", "dashboards", "boleteria.json"))
salida = ap.parse_args().salida
os.makedirs(os.path.dirname(os.path.abspath(salida)), exist_ok=True)
json.dump(tablero, open(salida, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(salida)
