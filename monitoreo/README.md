# Tablero en vivo: Prometheus + Grafana

Muestra, segundo a segundo, si algún servicio se está muriendo y si el modelo lo vio venir. Es opcional y va aparte
de los servicios: solo se enciende cuando se quiere mirar.

## Uso

```powershell
docker compose up -d                                        # los 6 servicios
docker compose -f monitoreo/docker-compose.yml up -d        # Prometheus y Grafana (la primera vez descarga ~500 MB)
python ml/demo.py                                           # el modelo; publica sus predicciones en el puerto 9108
# en otra terminal, la carga, por ejemplo:
python run_experiment.py --scenario progresivo --duration 360 --peak-users 60 --run-id vivo_grafana_1 --inject notificaciones:100:90:90:60
```

- **Grafana:** http://localhost:3000. Abre directo el tablero "Boletería: ¿se está muriendo algo?". Para mirar no pide usuario; para editar, `admin` / `admin`.
- **Prometheus:** http://localhost:9090. La pestaña **Alerts** lista las alertas activas.
- **Sin Docker ni carga:** `python ml/demo.py --replay rnd_combo_prog_01` también publica, así que el tablero muestra una corrida grabada. Los paneles de tráfico, errores y espera vienen de los servicios y quedan vacíos.
- **Sin la demo:** siguen funcionando la latencia, la cola, el tráfico, los errores, el tiempo de espera y la alerta `ServicioNoResponde`, porque salen de los servicios. El estado, la probabilidad, la CPU y las alertas del modelo necesitan `python ml/demo.py`. La demo se puede dejar en segundo plano, sin pantalla: `Start-Process python -ArgumentList "ml/demo.py","--sin-pantalla" -WindowStyle Hidden`. Para detenerla, cierra Python desde el Administrador de tareas.
- **Apagar:** `docker compose -f monitoreo/docker-compose.yml down`. Hazlo antes de grabar campañas: Prometheus pide `/metrics` a cada servicio cada segundo y suma carga.

## El tablero

Está armado como la pantalla de `python ml/demo.py`:

| Panel | Qué muestra |
| --- | --- |
| **Resumen** (arriba) | Segundos de carga, si el modelo ya predice o está calentando, cuántos servicios están en cuello de botella y en cuántos predice el modelo |
| **Ahora** | Un renglón por servicio, en el orden de la cadena. Columnas: **estado** en color (verde normal, naranja "⚠ EL MODELO PREDICE", rojo "✖ CUELLO DE BOTELLA", morado "espera a otro"), **probabilidad** en barra, CPU, regla CPU > 80 %, p95, cola, **hace cuánto se encendió la alarma**, a qué servicio espera y **qué avisó el modelo en su último cuello de botella** ("avisó 14 s antes", "avisó la causa (ticket)", "sin aviso previo") |
| **Eventos de esta carga** | Los mismos mensajes de la demo, con color por tipo, el más reciente arriba: el modelo avisa, entra en cuello de botella (con cuánto antes avisó cada método), espera a otro, se recuperó |
| Probabilidad | La del modelo por servicio en el tiempo; la línea roja punteada es el umbral (0,51) |
| Estado en el tiempo | Los mismos colores como franjas en el tiempo |
| Detalle del sistema (plegado) | Historial de cuellos de botella, alertas de Prometheus, latencia, CPU, tiempo esperando a otros servicios, cola, tráfico y errores |

- **Eventos e historial de una sola carga:** muestran solo la carga actual. Cuando una carga termina, la demo empieza una sesión nueva.
- **Una sola demo a la vez:** solo la primera que arranca publica en el puerto 9108. Las demás avisan "No se pudo publicar en el puerto 9108".
- **Vuelta a la vista normal:** si se hace zoom sin querer, http://localhost:3000/d/boleteria?from=now-10m&to=now&refresh=1s.

**Para probar el tablero sin tocar la demo principal:** lanza una segunda demo con `python ml/demo.py --replay rnd_combo_prog_01 --exportar 9109` y abre http://localhost:3000/d/boleteria?var-job=modelo_prueba. Las capturas se sacan con el contenedor `grafana-renderer`:
```
curl -u admin:admin -o tablero.png "http://localhost:3000/render/d/boleteria?var-job=modelo_prueba&width=1700&height=1450&kiosk"
```

## Archivos

- `docker-compose.yml`: Prometheus 2.53 y Grafana 11.2, con 1 CPU y 512 MB cada uno, en la red de los servicios, más el renderizador de capturas de Grafana.
- `prometheus.yml`: lee cada segundo los 6 servicios y la demo (`host.docker.internal:9108`).
- `alertas.yml`: las reglas de alerta.
- `grafana/`: la fuente de datos y el tablero, que se cargan solos. El tablero se regenera con `python monitoreo/generar_tablero.py`.

La demo publica estas métricas (`python ml/demo.py --exportar 0` lo desactiva):
- `boleteria_estado`, `boleteria_probabilidad`, `boleteria_puntaje`
- `boleteria_alarma_modelo`, `boleteria_causa`, `boleteria_alarma_cpu`, `boleteria_cuello_real`
- `boleteria_cpu`, `boleteria_p95_ms`, `boleteria_cola`, `boleteria_umbral`, `boleteria_calentando`
- `boleteria_alarma_hace_s`, `boleteria_ultimo_inicio_s`, `boleteria_ultimo_lead_s`, `boleteria_ultimo_aviso`, `boleteria_episodio`
