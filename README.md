# Boletería en la Nube: documentación del proyecto

Proyecto del primer parcial: *Microservice Bottleneck Prediction*. Este documento explica la estructura del proyecto, qué hace cada archivo, qué métricas se recolectan (y cómo se calculan), qué genera cada experimento y qué limitaciones tienen los datos.

---

## 1. Estructura del proyecto

```
boleteria_nube/
├── docker-compose.yml          # levanta los 6 servicios con límites de CPU/RAM
├── requirements-tools.txt      # dependencias de las herramientas (en tu máquina, no en Docker)
├── run_experiment.py           # corre UN experimento completo
├── run_campaign.py             # corre VARIOS experimentos en cadena
│
├── servicios/                  # el backend (corre dentro de Docker)
│   ├── usuarios/
│   ├── eventos/
│   ├── reservaciones/
│   ├── pago/
│   ├── ticket/
│   └── notificaciones/
│       ├── Dockerfile
│       ├── requirements.txt
│       └── app/
│           ├── __init__.py
│           ├── common.py       # idéntico en los 6: métricas, carga simulada
│           └── main.py         # endpoints propios de cada servicio
│
├── collector/
│   └── collector.py            # mide el sistema 1 vez por segundo
│
├── loadtest/
│   ├── locustfile.py           # genera tráfico (3 escenarios)
│   └── inyector.py             # provoca cuellos de botella controlados
│
├── ml/
│   └── graficar_corrida.py     # grafica una corrida para validarla
│
└── data/                       # resultados: una carpeta por corrida
    ├── dataset_metrics.csv     # (lo genera run_campaign al final) todas las corridas unidas
    ├── dataset_edges.csv
    ├── resumen_corridas.csv
    └── <run_id>/
        ├── metrics.csv
        ├── edges.csv           # métricas por dependencia (origen → destino)
        ├── events.csv          # solo si hubo inyección
        ├── locust_stats.csv (y otros locust_*.csv)
        ├── meta.json
        ├── grafica.png         # si la graficaste
        └── OK                  # lo pone run_campaign al terminar bien
```

### Qué corre dónde

| Parte | Dónde corre | Cómo se inicia |
|---|---|---|
| `servicios/*` | Contenedores Docker | `docker compose up -d --build` |
| `collector`, `loadtest`, `run_*.py`, `ml/` | Tu máquina (Python) | `python run_experiment.py ...` |

Los scripts de tu máquina hablan con los servicios por `localhost` y leen las estadísticas de los contenedores a través del SDK de Docker. Por eso Docker Desktop debe estar abierto.

### Qué aún no existe

Estos archivos quedan para la siguiente etapa: `ml/etiquetar.py` (generar las etiquetas), `ml/entrenar.py` (LSTM) y `ml/evaluar.py` (Precision, Recall, F1, Lead Time y comparación contra el baseline).

---

## 2. La aplicación: 6 microservicios

Una boletería simulada. Cada reserva recorre una cadena de llamadas síncronas:

```
                         ┌──────────┐
                    ┌───►│ usuarios │
                    │    └──────────┘
┌───────────────┐   │    ┌──────────┐
│ reservaciones │───┼───►│ eventos  │
└───────────────┘   │    └──────────┘
                    │    ┌──────────┐    ┌──────────┐    ┌────────────────┐
                    └───►│   pago   │───►│  ticket  │───►│ notificaciones │
                         └──────────┘    └──────────┘    └────────────────┘
```

| Servicio | Puerto | Llama a | Rol |
|---|---|---|---|
| usuarios | 8001 | n/a | Valida que el usuario exista |
| eventos | 8002 | n/a | Descuenta boletos (con lock) y devuelve el precio |
| reservaciones | 8003 | usuarios, eventos, pago | Orquestador: recibe la petición del cliente |
| pago | 8004 | ticket | Registra el cobro |
| ticket | 8005 | notificaciones | Genera el código del boleto |
| notificaciones | 8006 | n/a | Final de la cadena |

Cada petición `POST /reservations` pasa **una vez por cada uno de los 6 servicios**, así que en carga normal todos reciben aproximadamente los mismos requests por segundo.

### `docker-compose.yml`

- Mapea `8001..8006` hacia el puerto interno 8000 de cada contenedor.
- **`cpus: "1.0"` y `mem_limit: 256m`** por servicio. Estos límites son los que hacen que un servicio se sature de forma realista, y el collector usa el límite de CPU como referencia (100% = usa su cupo completo).
- `container_name` igual al nombre del servicio, porque el collector busca los contenedores por ese nombre.
- Variables de entorno: `SERVICE_NAME`, `BASE_DELAY_MS`, `CPU_LOAD` y las URLs de los servicios que cada uno llama.

### `common.py` (idéntico en los 6 servicios)

Es la capa de medición y experimentación, separada de la lógica de negocio.

| Elemento | Función |
|---|---|
| `config` | `base_delay_ms` y `cpu_load` actuales. Se leen del entorno al arrancar y se pueden cambiar en caliente |
| `simulate_work()` | Espera `base_delay_ms` y ejecuta un bucle de `cpu_load × 10.000` iteraciones. **Es lo que fabrica el cuello de botella** |
| `call_service()` | Llamada HTTP a otro servicio; los fallos se devuelven como 502 |
| `create_app()` | Crea la app con el middleware de métricas y los endpoints de utilidad |

**Métricas Prometheus que expone cada servicio en `GET /metrics`:**

| Métrica | Tipo | Etiquetas | Qué mide |
|---|---|---|---|
| `http_requests_total` | Counter | service, method, path, status | Requests atendidos |
| `http_request_duration_seconds` | Histogram | service, path | Latencia de cada request |
| `http_requests_in_flight` | Gauge | service | Requests que están siendo procesados ahora mismo |

- Los **buckets** del histograma van de 5 ms a 30 s (20 valores: 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 0.8, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 30). Son finos entre 100 ms y 4 s a propósito, que es donde se decide si un servicio está degradado.
- El middleware **no mide** `/metrics` ni `/health`.
- La etiqueta `path` usa la plantilla de la ruta (`/users/{user_id}`) y no el valor real, para no generar miles de series.
- La latencia de un servicio **incluye el tiempo de espera de los servicios que llama**. Por eso la latencia de `reservaciones` es la suma de toda la cadena.
- `http_requests_in_flight` es el proxy de **longitud de cola**.

**Endpoints que reciben todos los servicios:**

| Endpoint | Función |
|---|---|
| `GET /health` | Responde `{"status":"ok"}` |
| `GET /metrics` | Métricas en formato Prometheus |
| `GET /admin/load` | Muestra `base_delay_ms` y `cpu_load` actuales |
| `POST /admin/load` | Cambia `cpu_load` y/o `base_delay_ms` sin reiniciar |

---

## 3. Las herramientas de experimentación

### 3.1 `collector/collector.py`: qué se mide

Cada segundo escribe **una fila por servicio** en `metrics.csv`. Toma los datos de dos fuentes: Docker (CPU y RAM) y el `/metrics` de cada servicio (todo lo demás).

| Columna | Descripción | Cómo se calcula |
|---|---|---|
| `timestamp` | Instante del muestreo (epoch, segundos) | Reloj de la máquina, con ticks alineados cada 1 s |
| `servicio` | Nombre del servicio | n/a |
| `cpu` | CPU en % **respecto al límite del contenedor** (100 = usa todo su `cpus`) | Núcleos usados = (Δ uso CPU / Δ CPU del sistema) × nº de CPUs; luego ÷ `CPU_LIMIT` × 100 |
| `ram_mb` | Memoria en MB | `usage` menos la caché de archivos inactiva (`inactive_file` o `cache`) |
| `rps` | Requests por segundo | Δ de `http_requests_total` ÷ Δ de tiempo |
| `lat_mean_ms` | Latencia media del último intervalo | Δ de `_sum` ÷ Δ de `_count` × 1000. Vale 0 si no hubo requests |
| `lat_p95_ms` | Latencia p95 del último intervalo | Estimada con los Δ de los buckets del histograma, con interpolación lineal |
| `cola` | Requests en curso (longitud de cola) | Valor instantáneo de `http_requests_in_flight` |
| `err_rps` | Respuestas 5xx por segundo | Δ de `http_requests_total` con `status` 5xx ÷ Δ de tiempo |
| `scrape_ok` | 1 si `/metrics` respondió, 0 si no | Timeout de 1.5 s |

**Cómo funciona por dentro:**

- La lectura de Docker usa un hilo por contenedor con `stats(stream=True)`, que entrega una muestra cada ~1 s. El collector siempre toma la más reciente.
- Los 6 `/metrics` se consultan **en paralelo** para no perder el ritmo.
- Todas las tasas son **diferencias entre dos lecturas consecutivas**, es decir, del último intervalo y no acumuladas desde el arranque.
- Si un contador baja (contenedor reiniciado), esa fila queda con las columnas de tasas vacías en lugar de un valor falso.
- `CPU_LIMIT = 1.0` está escrito en el script y **debe coincidir** con `cpus` del `docker-compose.yml`.

**Correspondencia con lo que pide el PDF:**

| Pide el PDF | Columna |
|---|---|
| CPU | `cpu` |
| RAM | `ram_mb` |
| Latencia | `lat_mean_ms`, `lat_p95_ms` |
| Requests/s | `rps` |
| Longitud de cola | `cola` |

### 3.2 `loadtest/locustfile.py`: el tráfico

Cada usuario virtual hace un `POST /reservations` (usuario 1, evento aleatorio entre 1 y 3, cantidad 1), espera entre 0.3 y 1.0 s y repite. Todo el tráfico entra por `reservaciones` (puerto 8003).

El escenario se elige con la variable `SCENARIO`:

| Escenario | Comportamiento | Parámetros (por defecto) |
|---|---|---|
| `normal` | Usuarios constantes | `BASE_USERS=15` |
| `progresivo` | Sube de 5 a `PEAK_USERS` durante el 65% del tiempo y vuelve a bajar a 5 | `PEAK_USERS` |
| `burst` | Carga base con picos súbitos | `BASE_USERS=15`, `PEAK_USERS`, picos de `BURST_LEN=25` s cada `BURST_EVERY=120` s desde `BURST_START=60` s |

Con 360 s, los picos del burst caen en los segundos 60-85, 180-205 y 300-325.

**Importante:** Locust funciona en **circuito cerrado**. Cada usuario espera la respuesta antes de enviar la siguiente petición. Por eso la cola de un servicio no puede superar el número de usuarios activos, y cuando el sistema se satura, `rps` se estanca en vez de seguir creciendo.

### 3.3 `loadtest/inyector.py`: cuellos de botella controlados

Cambia `cpu_load` de un servicio mediante `POST /admin/load` en momentos programados, y guarda los **tiempos reales** de inicio y fin en `events.csv`. Esos tiempos son la referencia para etiquetar.

**Formato:** `--inject servicio:inicio_s:duracion_s:cpu_load[:rampa_s]`

| Parte | Significado |
|---|---|
| `inicio_s` | Segundos desde el inicio de la carga |
| `duracion_s` | Duración total de la inyección (**incluye** la rampa) |
| `cpu_load` | Valor final de la carga |
| `rampa_s` | Opcional. Segundos para subir de 0 al valor final, un paso por segundo. Sin rampa, es un escalón abrupto |

Ejemplos:

- `pago:60:60:150` → escalón: a los 60 s sube a 150 de golpe, durante 60 s.
- `pago:60:90:150:40` → rampa: sube de 0 a 150 entre los 60 y 100 s, y se mantiene hasta los 150 s.

Reglas: las inyecciones de una misma corrida se ejecutan de forma secuencial (no pueden solaparse), y al terminar, o si se interrumpe, **todos los servicios vuelven a `cpu_load=0`**. `python loadtest/inyector.py --reset` lo hace a mano.

**`events.csv`:**

| Columna | Descripción |
|---|---|
| `run_id`, `escenario` | Identifican la corrida |
| `servicio` | Servicio inyectado |
| `t_inicio`, `t_fin` | Epoch real del inicio y del fin de la inyección |
| `cpu_load` | Valor final de la carga |
| `rampa_s` | 0 si fue abrupta, o los segundos de rampa |

### 3.4 `run_experiment.py`: una corrida completa

Coordina las tres herramientas con el mismo `run_id`:

1. Pone todos los servicios en `cpu_load=0`.
2. Arranca el collector y espera 5 s (línea base en reposo).
3. Arranca Locust (modo sin interfaz) y el inyector al mismo tiempo.
4. Cuando Locust termina, espera 15 s más para registrar la recuperación.
5. Detiene todo y vuelve a resetear la carga.

```powershell
python run_experiment.py --scenario normal --duration 300
python run_experiment.py --scenario burst --duration 360 --inject pago:95:75:150:30 --peak-users 70
```

| Argumento | Por defecto | Descripción |
|---|---|---|
| `--scenario` | (obligatorio) | `normal`, `progresivo` o `burst` |
| `--duration` | 600 | Segundos de carga |
| `--inject` | ninguno | Repetible, formato del inyector |
| `--base-users` | 15 | Usuarios base |
| `--peak-users` | 100 | Usuarios en el pico |
| `--run-id` | automático | Por defecto: `escenario_servicio_fecha` |

La carpeta `data/<run_id>/` contiene al terminar:

| Archivo | Contenido |
|---|---|
| `metrics.csv` | Las métricas por servicio y segundo (tabla de la sección 3.1) |
| `events.csv` | Inyecciones reales (solo si hubo) |
| `locust_stats.csv`, `locust_stats_history.csv`, `locust_failures.csv`, ... | Medición **del lado del cliente**: latencia vista por el usuario y fallos |
| `meta.json` | Escenario, duración, usuarios, inyecciones y `t0_locust` (el instante en que empezó la carga) |

Una corrida de `--duration D` genera unas `6 × (D + 25)` filas en `metrics.csv`.

### 3.5 `run_campaign.py`: muchas corridas en cadena

Ejecuta un plan de corridas llamando a `run_experiment.py`, para generar el dataset completo sin supervisión.

```powershell
python run_campaign.py --dry-run             # solo ver el plan y el tiempo
python run_campaign.py --only normal_base    # un grupo
python run_campaign.py                       # todo
```

**Plan por defecto (28 corridas, unas 3.3 h):**

| Grupo | Corridas | Contenido |
|---|---|---|
| `normal_base` | 2 | Carga constante, 300 s, sin inyección |
| `normal_inj` | 6 | Carga constante, 360 s, inyección en cada servicio |
| `burst_inj` | 6 | Picos de tráfico (70 usuarios), 360 s, inyección en cada servicio |
| `progresivo_base` | 2 | Tráfico creciente y decreciente (60 usuarios), 360 s, sin inyección |
| `var_inj` | 12 | Inyección **leve** (`cpu_load` 90, 15 usuarios) y **fuerte** (250, 20 usuarios) en cada servicio, 360 s |

Cada corrida con inyección contiene **dos eventos**: uno **gradual** (rampa de 40 s o 30 s) y uno **abrupto**, con `cpu_load` 150.

**Comportamiento:**

- **Reinicia los contenedores** antes de cada corrida (`docker compose restart`). Esto evita que las listas en memoria agoten los 256 MB, y cada corrida parte de un estado limpio.
- **Reanuda:** al terminar bien, deja un archivo `OK` en la carpeta de la corrida. Si se interrumpe y se vuelve a ejecutar, se salta las corridas con `OK` y repite la incompleta.
- **Opciones:** `--reps N` (más repeticiones), `--load N` (cpu_load de las inyecciones), `--only grupo1,grupo2`, `--cooldown N` (pausa entre corridas).

### 3.6 `ml/graficar_corrida.py`: validar una corrida

```powershell
python ml/graficar_corrida.py                          # última corrida
python ml/graficar_corrida.py data\<run_id> --smooth 3 --log --show
```

Genera `grafica.png` con 4 paneles (latencia p95, cola, CPU y requests/s, una línea por servicio), las inyecciones sombreadas en rojo y el inicio y fin de la carga en líneas grises. Si hay inyecciones, también imprime una tabla "fuera vs dentro" para el servicio inyectado y para `reservaciones`.

Opciones: `--smooth N` (media móvil de N segundos), `--log` (latencia en escala logarítmica), `--show` (abre la ventana además de guardar).

---

## 4. Referencia rápida de comandos

```powershell
# Levantar / reconstruir los servicios
docker compose up -d --build

# Probar un flujo a mano
curl -X POST http://localhost:8003/reservations -H "Content-Type: application/json" -d '{"user_id":1,"event_id":1,"quantity":2}'

# Un experimento
python run_experiment.py --scenario normal --duration 150 --inject pago:40:90:150:40

# Una campaña
python run_campaign.py --dry-run
python run_campaign.py

# Ver los resultados
python ml/graficar_corrida.py data\<run_id> --smooth 3
```

---

## 5. Métricas de evaluación (etapa siguiente)

Las que pide el PDF. Se calcularán igual para el baseline y para el LSTM, sobre las mismas corridas de prueba.

| Métrica | Pregunta que responde |
|---|---|
| Precision | De las alarmas emitidas, ¿cuántas correspondían a un cuello de botella real? |
| Recall | De los cuellos de botella reales, ¿cuántos se detectaron? |
| F1 | Equilibrio entre Precision y Recall |
| Prediction Lead Time | ¿Cuántos segundos antes del inicio del cuello de botella se emitió la primera alarma correcta? |

Aún falta definir (y justificar antes de comparar):

- **Qué es un cuello de botella en el instante t.** Una propuesta es un criterio de latencia, por ejemplo p95 por encima de un umbral sostenido varios segundos, idéntico para todas las corridas. Un umbral de cola fijo no sirve, porque con Locust en circuito cerrado la cola está acotada por el número de usuarios.
- **Qué cuenta como alarma correcta.** Por ejemplo, la alarma emitida dentro de los 30 s previos al inicio del evento. Las alarmas fuera de esa ventana son falsos positivos.

El **baseline** es la regla `CPU > 80%`. Los datos muestran que falla en un caso importante: cuando `pago` se satura, `reservaciones` se degrada pero su CPU **baja** (está esperando), así que la regla nunca la detecta.

---

## 6. Limitaciones conocidas de los datos

- **Todo corre en una sola máquina.** Locust, el collector, Docker y los servicios compiten por la misma CPU. Eso añade ruido y hace que cada corrida dependa del equipo. Mantén el PC sin otras tareas pesadas durante las corridas.
- **Locust en circuito cerrado:** la cola nunca supera el número de usuarios activos y el `rps` se estanca al saturarse.
- **La latencia p95 es una estimación.** Sale de un histograma con buckets, no de las muestras individuales, así que su precisión depende de la anchura del bucket donde cae.
- **Filas con `scrape_ok=0`:** cuando un servicio está muy saturado puede no responder a `/metrics` a tiempo. Esas filas son información (el servicio dejó de responder), no errores. No las borres sin pensar.
- **Segundos en reposo:** al inicio y al final de cada corrida hay filas con `rps=0` y latencia 0. Conviene filtrarlas o recortarlas al analizar.
- **Pico final tras Locust:** al parar, la cola acumulada se vacía de golpe y aparece un pico de `rps` y latencia. No es comportamiento normal; recórtalo o márcalo al etiquetar.
- **El estado vive en memoria y se pierde al reiniciar.** Es intencional: cada corrida parte de cero.
- **`cpu_load=150` para todos los servicios** es un valor de partida validado solo en `pago`. Los demás servicios pueden requerir ajuste.
- **Una corrida tiene pocos eventos.** Para entrenar y evaluar hacen falta varias repeticiones, y la división entre entrenamiento y prueba debe hacerse **por corrida**, no por filas al azar.
- **Python 3.8** funciona para las herramientas actuales, pero ya no tiene soporte oficial.


---

## 7. Actualización: interacción entre servicios y consolidación del dataset

### 7.1 Métricas de dependencias (`edges.csv`)

Cada servicio que llama a otro mide ahora **cada arista** de la cadena, desde el lado de quien llama. `common.py` expone dos métricas nuevas:

| Métrica | Tipo | Etiquetas | Qué mide |
|---|---|---|---|
| `downstream_request_duration_seconds` | Histogram | service, target | Latencia de cada llamada a otro servicio (red + espera al destino) |
| `downstream_requests_total` | Counter | service, target, status | Llamadas realizadas y su resultado (`error` si no hubo respuesta) |

El collector las convierte en `edges.csv`, una fila por dependencia y segundo:

| Columna | Descripción |
|---|---|
| `timestamp` | Instante del muestreo |
| `origen`, `destino` | Servicio que llama y servicio llamado |
| `cps` | Llamadas por segundo |
| `lat_mean_ms`, `lat_p95_ms` | Latencia de la llamada, media y p95 |
| `err_ps` | Llamadas fallidas (5xx o sin respuesta) por segundo |

Las **5 aristas** de la aplicación son: `reservaciones → usuarios`, `reservaciones → eventos`, `reservaciones → pago`, `pago → ticket` y `ticket → notificaciones`.

Para qué sirven:

- **Localizar la causa.** La latencia de un servicio incluye la espera a los que llama. Comparar la latencia propia del servicio con la de sus aristas permite separar el tiempo que pasa dentro del servicio de la propagación desde un destino lento.
- **Segundo parcial.** Son las relaciones entre servicios que usará la Temporal GNN, con una característica medida por cada arista.

Si un servicio no responde a `/metrics` en un segundo dado, ese segundo no genera filas de aristas para él.

### 7.2 Verificación previa de `run_campaign.py`

Antes de empezar, la campaña comprueba que los contenedores y el collector tienen el código actualizado. Hace una reserva de prueba y revisa que `/metrics` de `reservaciones` incluya las métricas de dependencias y los buckets finos de latencia, y que `collector.py` escriba `edges.csv`. Si algo está desactualizado, se detiene con un mensaje claro y no gasta horas generando datos incompletos. Se puede omitir con `--sin-chequeo`.

### 7.3 Consolidación automática

Al terminar (o al interrumpirse con Ctrl+C), `run_campaign.py` une todas las corridas completas (las que tienen el archivo `OK`) y genera en `data/`:

| Archivo | Contenido |
|---|---|
| `dataset_metrics.csv` | Métricas por servicio y segundo de todas las corridas |
| `dataset_edges.csv` | Métricas por dependencia de todas las corridas |
| `resumen_corridas.csv` | Una fila por corrida: escenario, duración, filas, eventos, servicios inyectados |

Columnas que se añaden a `dataset_metrics.csv`:

| Columna | Significado |
|---|---|
| `run_id`, `escenario` | Identifican la corrida |
| `t_rel` | Segundos desde el inicio de la carga de Locust |
| `en_carga` | 1 si la fila cae dentro de la ventana de carga |
| `inj_servicio` | 1 si **ese** servicio tenía una inyección activa en ese instante |
| `inj_alguna` | 1 si había alguna inyección activa en la corrida |
| `inj_tipo` | `gradual`, `abrupto` o vacío |

Importante: estas columnas son la verdad de **lo que se inyectó**. No son la etiqueta de cuello de botella, que se deriva de los síntomas (latencia, cola) en el paso de etiquetado. Un servicio puede estar saturado sin tener inyección propia (por ejemplo `reservaciones` cuando se satura `pago`), y una inyección leve puede no producir saturación.

También se puede consolidar sin correr nada: `python run_campaign.py --solo-consolidar`.

### 7.4 Opciones de `run_campaign.py`

| Opción | Efecto |
|---|---|
| `--dry-run` | Muestra el plan y el tiempo estimado, sin ejecutar |
| `--only g1,g2` | Corre solo esos grupos |
| `--reps N` | Repite el plan N veces |
| `--load N` | `cpu_load` de las inyecciones de los grupos `normal_inj` y `burst_inj` |
| `--cooldown N` | Pausa entre corridas (20 s por defecto) |
| `--solo-consolidar` | Solo une los datos ya existentes |
| `--sin-consolidar` | No consolida al final |
| `--sin-chequeo` | Omite la verificación previa |