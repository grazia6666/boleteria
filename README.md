# Guía del proyecto: variables, campañas y arquitecturas

Proyecto P1 — *Microservice Bottleneck Prediction* (Sistemas en la Nube).
Esta guía explica **qué medimos, por qué, cómo se genera cada dato, qué parámetros tiene cada experimento y cómo funciona cada arquitectura** (los microservicios, el pipeline de datos y el modelo). Está escrita para consultarla cuando te pierdas: cada sección es independiente.

---

## Índice

1. [La idea completa en un minuto](#1-la-idea-completa-en-un-minuto)
2. [Arquitectura 1: los 6 microservicios](#2-arquitectura-1-los-6-microservicios)
3. [Conceptos previos (glosario corto)](#3-conceptos-previos-glosario-corto)
4. [Variables que recolectamos](#4-variables-que-recolectamos)
5. [Qué archivos genera cada corrida](#5-qué-archivos-genera-cada-corrida)
6. [Parámetros de los experimentos y las campañas](#6-parámetros-de-los-experimentos-y-las-campañas)
7. [Etiquetado: cómo decidimos que hay un cuello de botella](#7-etiquetado-cómo-decidimos-que-hay-un-cuello-de-botella)
8. [Arquitectura 2: el pipeline de datos y código](#8-arquitectura-2-el-pipeline-de-datos-y-código)
9. [Arquitectura 3: el baseline y el modelo LSTM](#9-arquitectura-3-el-baseline-y-el-modelo-lstm)
10. [Métricas de evaluación](#10-métricas-de-evaluación)
11. [Comandos en orden](#11-comandos-en-orden)
12. [Limitaciones que debes conocer](#12-limitaciones-que-debes-conocer)

---

## 1. La idea completa en un minuto

1. Tenemos **6 microservicios** (una boletería) que se llaman entre sí.
2. Les mandamos **tráfico simulado** (Locust) y, a propósito, **les provocamos cuellos de botella** (inyector) en momentos conocidos.
3. Un **recolector** mide cada servicio cada segundo (CPU, RAM, latencia, peticiones por segundo, cola, errores) y guarda todo en CSV.
4. Un script de **etiquetado** decide, mirando la latencia, en qué instantes cada servicio estaba en cuello de botella, y marca los **30 segundos anteriores** como "se viene un cuello de botella".
5. Comparamos dos formas de predecirlo:
   - **Baseline:** una regla fija ("alarma si CPU > 80 %").
   - **LSTM:** una red neuronal que mira los últimos 30 s de métricas y estima la probabilidad de que haya un cuello de botella en los próximos 30 s.
6. Medimos con **Precision, Recall, F1 y Prediction Lead Time** (cuántos segundos antes avisamos).

```
tráfico (Locust) ──► microservicios ──► recolector ──► CSV ──► etiquetado ──► baseline / LSTM ──► métricas
                          ▲
                inyector (provoca cuellos de botella)
```

---

## 2. Arquitectura 1: los 6 microservicios

Todos son aplicaciones **FastAPI** (Python) en contenedores **Docker**, orquestados con `docker-compose.yml`. Cada contenedor tiene un límite de **1 CPU y 256 MB de RAM**. Ese límite importa: el "100 % de CPU" que medimos es relativo a esa 1 CPU, y por eso un servicio puede saturarse de verdad.

### 2.1 Quién llama a quién

```
                        ┌──────────► usuarios        (8001)
 Locust ──► reservaciones (8003) ────► eventos         (8002)
 (tráfico)              └──────────► pago            (8004) ──► ticket (8005) ──► notificaciones (8006)
```

- Locust **solo** habla con `reservaciones`: cada reserva (`POST /reservations`) dispara llamadas en cadena.
- `reservaciones` llama a `usuarios`, `eventos` y `pago`.
- `pago` llama a `ticket`; `ticket` llama a `notificaciones`.
- **La cadena más larga tiene 4 servicios:** `reservaciones → pago → ticket → notificaciones`.

### 2.2 Qué hace cada servicio

| Servicio | Puerto (host) | Rol en la boletería | Endpoints principales | Llama a |
| --- | --- | --- | --- | --- |
| `usuarios` | 8001 | Registro y consulta de usuarios | `POST/GET /users`, `GET /users/{id}` | nadie |
| `eventos` | 8002 | Catálogo de eventos y entradas disponibles | `GET/POST /events`, `GET /events/{id}`, `POST /events/{id}/reserve` | nadie |
| `reservaciones` | 8003 | **Orquestador**: crea la reserva | `POST/GET /reservations`, `GET /reservations/{id}` | usuarios, eventos, pago |
| `pago` | 8004 | Procesa el pago | `POST/GET /payments` | ticket |
| `ticket` | 8005 | Genera el ticket | `POST/GET /tickets` | notificaciones |
| `notificaciones` | 8006 | Envía la notificación final | `POST/GET /notifications` | nadie |

Dentro de la red de Docker cada servicio escucha en el puerto 8000; los puertos 8001–8006 son los que se ven desde tu PC.

### 2.3 El archivo `common.py` (idéntico en los 6)

Es el "motor común" de todos los servicios. Aporta:

| Pieza | Qué hace | Por qué nos sirve |
| --- | --- | --- |
| `config` (`base_delay_ms`, `cpu_load`) | Parámetros de comportamiento modificables en vivo | Permiten provocar cuellos de botella |
| `simulate_work()` | Espera `base_delay_ms` (por defecto 5 ms) y luego hace un bucle de CPU de `cpu_load × 10.000` iteraciones | Con `cpu_load=0` el servicio es rápido; al subirlo consume CPU real y se satura |
| `call_service()` | Llama a otro servicio, mide cuánto tardó esa llamada y devuelve 502 si falla | Genera las métricas de **aristas** (dependencias) |
| Middleware de métricas | Cuenta peticiones, mide su duración y cuántas hay en curso | Genera las variables que recolectamos |
| `GET /health` | Responde si el servicio está vivo | Verificación previa de la campaña |
| `GET /metrics` | Expone las métricas en formato Prometheus | De aquí lee el recolector |
| `GET/POST /admin/load` | Lee o cambia `cpu_load` y `base_delay_ms` | **Es el mecanismo de inyección de cuellos de botella** |

### 2.4 Las métricas que expone cada servicio (Prometheus)

| Métrica | Tipo | Etiquetas | Qué cuenta |
| --- | --- | --- | --- |
| `http_requests_total` | Counter | service, method, path, status | Peticiones atendidas (acumulado) |
| `http_request_duration_seconds` | Histogram | service, path | Latencia de cada petición, en cubetas |
| `http_requests_in_flight` | Gauge | service | Peticiones que están siendo atendidas **ahora** |
| `downstream_request_duration_seconds` | Histogram | service, target | Latencia de las llamadas a otros servicios |
| `downstream_requests_total` | Counter | service, target, status | Cuántas llamadas a otros servicios y con qué resultado |

Las cubetas (*buckets*) del histograma son finas, de 5 ms a 30 s: `0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 0.8, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 30` segundos. Eso permite estimar percentiles con razonable precisión alrededor de 100–400 ms, que es donde vive nuestro umbral.

> **Counter vs Gauge vs Histogram.** Un *counter* solo crece (hay que restar dos lecturas para obtener una tasa). Un *gauge* sube y baja (valor instantáneo). Un *histogram* cuenta cuántas observaciones cayeron en cada rango; de ahí salen medias y percentiles.

---

## 3. Conceptos previos (glosario corto)

| Concepto | Explicación sencilla |
| --- | --- |
| **Cuello de botella** | Un servicio no da abasto: sus peticiones empiezan a tardar mucho más de lo normal (o dejan de responder). |
| **Saturación** | La CPU (o el recurso) está al límite; es una causa típica de cuello de botella. |
| **Propagación** | Un servicio lento hace que quien lo llama también se vuelva lento, aunque su CPU esté tranquila. |
| **Latencia** | Cuánto tarda una petición en responderse. |
| **p95** | El valor por debajo del cual está el 95 % de las latencias. Muestra las peticiones lentas que la media oculta. |
| **Throughput / rps** | Peticiones por segundo que atiende el servicio. |
| **Cola** | Peticiones esperando o en curso. Aquí usamos *peticiones en vuelo* como aproximación. |
| **Ventana** | Los últimos *W* segundos de métricas que ve el modelo (aquí W = 30 s). |
| **Horizonte** | Cuánto hacia el futuro predecimos (aquí H = 30 s). |
| **Episodio** | Un tramo continuo de cuello de botella en un servicio (de inicio a fin). |
| **Inyección** | Provocar un cuello de botella a propósito con `/admin/load`. |
| **Escenario** | El patrón de tráfico: normal, progresivo o ráfaga. |
| **Corrida (run)** | Un experimento completo con un escenario y unas inyecciones; deja su carpeta en `data/<run_id>/`. |
| **Campaña** | El conjunto de corridas que se ejecutan en cadena (`run_campaign.py`). |

---

## 4. Variables que recolectamos

El recolector (`collector/collector.py`) toma **una muestra por segundo y por servicio**. Mezcla dos fuentes:

- **Docker** (`docker stats`): CPU y RAM del contenedor.
- **Prometheus `/metrics`** de cada servicio: peticiones, latencia, cola, errores.

### 4.1 `metrics.csv`: una fila por servicio y por segundo

| Columna | Unidad | Qué es | Cómo se calcula | Cómo leerla |
| --- | --- | --- | --- | --- |
| `timestamp` | s (época) | Momento de la muestra | Reloj del recolector | Sirve para alinear con eventos |
| `servicio` | texto | Cuál de los 6 | Nombre del contenedor | Entrada del modelo (one-hot) |
| `cpu` | % | Uso de CPU del contenedor | `docker stats`, relativo al límite de 1 CPU | 100 % = saturado. Sube por carga o por inyección |
| `ram_mb` | MB | Memoria usada | `docker stats` | Cambia poco aquí; útil para detectar fugas |
| `rps` | req/s | Peticiones atendidas por segundo | Δ`http_requests_total` / Δtiempo | Tráfico que recibe el servicio. Puede dar saltos por muestreo irregular |
| `lat_mean_ms` | ms | Latencia media en el intervalo | Δsuma de duraciones / Δnúmero de peticiones | Sube cuando el servicio se degrada |
| `lat_p95_ms` | ms | Latencia p95 en el intervalo | Se interpola dentro de las cubetas del histograma | Es la **señal principal de cuello de botella** (el etiquetado se basa en ella) |
| `cola` | peticiones | Peticiones en curso | Gauge `http_requests_in_flight` | Si crece, el servicio no da abasto. Es el mejor indicador adelantado |
| `err_rps` | err/s | Respuestas con error por segundo | Δ de peticiones con estado 5xx o `error` | Fallos visibles (por ejemplo 502 por una dependencia caída) |
| `scrape_ok` | 0 / 1 | ¿Respondió `/metrics` a tiempo? | 1 si hubo respuesta antes de 0,9 s | `0` durante saturación fuerte; esas filas traen latencias vacías |

**Detalles que conviene recordar**

- El p95 es una **estimación** a partir de cubetas, no el percentil exacto.
- Cuando `scrape_ok = 0` el servicio estaba tan saturado que no contestó a la medición; **eso mismo es señal de cuello de botella** y el etiquetado lo trata como tal.
- En las corridas ya hechas (recolector anterior) el muestreo era irregular; `ml/datos.py` las remuestrea a una rejilla de 1 s.

### 4.2 `edges.csv`: una fila por dependencia y por segundo

Mide cada **arista** origen → destino (por ejemplo `reservaciones → pago`).

| Columna | Unidad | Qué es |
| --- | --- | --- |
| `timestamp` | s | Momento |
| `origen`, `destino` | texto | Quién llama y a quién |
| `cps` | llamadas/s | Llamadas por segundo por esa arista |
| `lat_mean_ms`, `lat_p95_ms` | ms | Latencia de esa llamada vista **desde el origen** (incluye red y espera) |
| `err_ps` | err/s | Llamadas fallidas por segundo |

Sirve para ver **dónde** se propaga un problema y, en P2, para modelos sobre grafos. En P1 el LSTM usa solo `metrics.csv`.

### 4.3 `events.csv`: qué inyectamos y cuándo (verdad conocida)

| Columna | Qué es |
| --- | --- |
| `run_id` | Corrida |
| `escenario` | normal / progresivo / burst |
| `servicio` | Servicio al que se inyectó carga |
| `t_inicio`, `t_fin` | Instante real de inicio y fin (época, en segundos) |
| `cpu_load` | Intensidad aplicada (multiplica el bucle de CPU) |
| `rampa_s` | Segundos que tardó en llegar a la intensidad final. `0` = salto abrupto; `>0` = subida gradual |

> **Ojo:** `events.csv` dice lo que *provocamos*, no lo que *ocurrió*. La etiqueta de cuello de botella se deriva después de los síntomas observados.

### 4.4 Columnas que añade la consolidación (`dataset_metrics.csv`)

| Columna | Significado |
| --- | --- |
| `run_id` | Identificador de la corrida |
| `escenario` | Escenario de tráfico |
| `t_rel` | Segundos desde el inicio de la carga |
| `en_carga` | 1 si ya había tráfico en ese instante |
| `inj_servicio` | 1 si **ese** servicio tenía inyección activa |
| `inj_alguna` | 1 si algún servicio tenía inyección activa |
| `inj_tipo` | `gradual` o `abrupto` |

### 4.5 Lo que **no** medimos (y por qué)

- **Ancho de banda de red, disco:** en este sistema no son el factor limitante.
- **Trazas distribuidas (ids de petición):** quedan para P2.
- **Tráfico real:** todo es simulado y usa siempre el mismo usuario (`user_id=1`).

---

## 5. Qué archivos genera cada corrida

En `data/<run_id>/`:

| Archivo | Contenido |
| --- | --- |
| `metrics.csv` | Variables por servicio y segundo (sección 4.1) |
| `edges.csv` | Variables por dependencia (sección 4.2) |
| `events.csv` | Inyecciones reales (sección 4.3) |
| `locust_*.csv` | Estadísticas de Locust: lo que vieron los usuarios simulados (peticiones, fallos, tiempos) |
| `meta.json` | Parámetros usados: escenario, duración, usuarios, inyecciones |
| `OK` | Marca de corrida completa (la campaña salta las que ya la tienen) |

Y tras consolidar, en `data/`: `dataset_metrics.csv`, `dataset_edges.csv` y `resumen_corridas.csv` (una fila por corrida).

---

## 6. Parámetros de los experimentos y las campañas

### 6.1 Los escenarios de tráfico (`loadtest/locustfile.py`)

Locust crea **usuarios virtuales en bucle cerrado**: cada usuario hace una petición, espera la respuesta y repite. Un usuario genera unas 1,25 req/s. Si el sistema se lentifica, los usuarios esperan más y generan menos tráfico (por eso el sistema "se frena solo").

| Escenario | Cómo es la carga | Para qué sirve |
| --- | --- | --- |
| **normal** | Usuarios constantes (`--base-users`, 15 por defecto) | Línea base y base para inyectar cuellos de botella |
| **progresivo** | Crece de 5 usuarios hasta `--peak-users` (60) y vuelve a bajar, en forma de triángulo (≈65 % del tiempo subiendo) | Saturación natural y gradual |
| **burst** | Carga base con **ráfagas** a `--peak-users` (70): la primera a los 60 s, luego cada 120 s, de 25 s de duración | Picos bruscos de tráfico |

Parámetros de entorno de Locust: `SCENARIO`, `DURATION`, `BASE_USERS`, `PEAK_USERS`, `BURST_START=60`, `BURST_EVERY=120`, `BURST_LEN=25`.

### 6.2 El inyector (`loadtest/inyector.py`)

Formato: `--inject servicio:inicio:duracion:cpu_load[:rampa]`

| Parte | Significado | Ejemplo `pago:60:90:150:40` |
| --- | --- | --- |
| `servicio` | A quién se inyecta | `pago` |
| `inicio` | Segundo (desde el inicio de la corrida) en que empieza | 60 |
| `duracion` | Cuánto dura | 90 s |
| `cpu_load` | Intensidad final (× 10.000 iteraciones por petición) | 150 |
| `rampa` (opcional) | Segundos para subir de 0 a `cpu_load` (1 escalón por segundo). Sin rampa = salto directo | 40 s |

Al terminar, el inyector **restablece** el servicio (`--reset`). También escribe `events.csv` con los tiempos reales.

### 6.3 Una corrida (`run_experiment.py`)

Lanza a la vez el recolector, Locust y el inyector, y guarda todo en `data/<run_id>/`.

| Parámetro | Defecto | Significado |
| --- | --- | --- |
| `--scenario` | (obligatorio) | normal / progresivo / burst |
| `--duration` | 600 | Duración en segundos |
| `--inject` | ninguno | Inyecciones (se puede repetir) |
| `--base-users` | 15 | Usuarios base |
| `--peak-users` | 100 | Usuarios en el pico |
| `--run-id` | auto | Nombre de la carpeta |

Ejemplo: `python run_experiment.py --scenario normal --duration 150 --inject pago:40:60:150`

### 6.4 La campaña (`run_campaign.py`)

Ejecuta todas las corridas en cadena. Entre una y otra **reinicia los contenedores** (estado limpio) y espera `--cooldown` s.

| Parámetro | Defecto | Significado |
| --- | --- | --- |
| `--reps` | 1 | Repeticiones de cada combinación (`--reps 2` duplica la campaña) |
| `--load` | 150 | `cpu_load` final de las inyecciones en `normal_inj` y `burst_inj` |
| `--only` | todos | Grupos a correr, separados por coma |
| `--cooldown` | 20 | Pausa entre corridas (s) |
| `--dry-run` | no | Muestra el plan y el tiempo estimado sin ejecutar |
| `--solo-consolidar` | no | No corre nada; solo une lo ya hecho |
| `--sin-consolidar` | no | Corre pero no consolida al final |
| `--sin-chequeo` | no | Omite la verificación previa |

La **verificación previa** comprueba que los 6 servicios responden, que una reserva atraviesa toda la cadena y que los contenedores tienen el `common.py` nuevo (métricas de dependencias y cubetas finas). Así no se pierden horas con código viejo.

Si interrumpes con Ctrl+C, consolida lo terminado; al reanudar, **salta** las corridas con marca `OK` y repite la incompleta.

### 6.5 El plan de la campaña (`--reps 1` = 28 corridas, ≈ 3,3 h)

| Grupo | Corridas | Escenario | Duración | Usuarios | Inyección |
| --- | --- | --- | --- | --- | --- |
| `normal_base` | 2 | normal | 300 s | 15 | Ninguna (negativos limpios; de aquí salen los umbrales) |
| `normal_inj` | 6 (una por servicio) | normal | 360 s | 15 | Gradual `s:60:90:150:40` (rampa de 40 s) y abrupta `s:220:60:150` |
| `burst_inj` | 6 (una por servicio) | burst | 360 s | 15 base, pico 70 | Gradual `s:95:75:150:30` y abrupta `s:215:75:150`, esquivando las ráfagas (que caen en 60–85, 180–205 y 300–325 s) |
| `progresivo_base` | 2 | progresivo | 360 s | 5 → 60 → 5 | Ninguna (saturación natural de `reservaciones`) |
| `var_inj` | 12 (6 leves + 6 fuertes) | normal | 360 s | leve 15 / fuerte 20 | Leve: `cpu_load` 90, rampa 40 s. Fuerte: `cpu_load` 250, rampa 20 s. Cada una con evento gradual (60 s) y abrupto (220 s) |

Cada corrida con inyección contiene **dos eventos**: uno gradual (con rampa) y uno abrupto (salto). Por eso en los resultados aparecen las causas `inyeccion_gradual` e `inyeccion_abrupto`.

**Por qué hay un grupo de cada tipo**

- `normal_base`, `progresivo_base`: ejemplos **sin inyección**, para que el modelo aprenda cómo es lo normal y no solo lo anómalo.
- `normal_inj`, `burst_inj`: cuellos de botella provocados con tráfico estable y con ráfagas (más ruido).
- `var_inj`: variedad de severidad, para no aprender un único patrón.

**Pendiente de la campaña ampliada:** `--reps 2` (56 corridas, ≈ 6,6 h), recolector nuevo con muestreo regular y un grupo de inyecciones **suaves** (`cpu_load` ≈ 25–40) que se degraden despacio y sean anticipables.

---

## 7. Etiquetado: cómo decidimos que hay un cuello de botella

Script: `ml/etiquetar.py`. Opera por servicio y por corrida.

### 7.1 Definición

1. **Umbral de latencia por servicio** = `max(3 × mediana de lat_p95 en corridas normal_base, 150 ms)`.
   Resultado real: 150 ms para `eventos`, `notificaciones`, `usuarios`, `ticket` y `pago`; 292,5 ms para `reservaciones` (es más lento por naturaleza, porque orquesta).
2. Un instante está **en cuello de botella** (`bn_ahora = 1`) si `lat_p95 > umbral` **o** el servicio no respondió a `/metrics` (`scrape_ok = 0`).
3. Se exige que dure **al menos 3 s seguidos** (`--sostenido`) para ignorar picos aislados.
4. Los episodios separados por **menos de 5 s** se fusionan en uno (`--fusion`).
5. Se ignoran los **primeros 10 s** de cada corrida (arranque, `--ignorar-inicio`).

### 7.2 Parámetros

| Parámetro | Defecto | Efecto |
| --- | --- | --- |
| `--umbral-mult` | 3 | Cuántas veces la latencia normal se considera cuello de botella |
| `--piso-ms` | 150 | Umbral mínimo (evita falsos positivos por ruido en servicios muy rápidos) |
| `--sostenido` | 3 s | Duración mínima de un episodio |
| `--fusion` | 5 s | Separación máxima para unir episodios |
| `--horizonte` | 30 s | Cuánto antes del episodio se marca "se viene" |
| `--ignorar-inicio` | 10 s | Segundos descartados al inicio |
| `--seed` | 42 | Semilla del reparto entre entrenamiento y prueba |
| `--sufijo` | vacío | Sufijo para guardar variantes sin pisar los archivos |

### 7.3 Columnas de `dataset_etiquetado.csv` (las más importantes)

| Columna | Significado |
| --- | --- |
| `bn_ahora` | 1 si en este instante ya hay cuello de botella |
| `ep_id`, `ep_causa` | Episodio al que pertenece y su causa |
| `y_30s` | **Etiqueta objetivo:** 1 si habrá cuello de botella en los próximos 30 s |
| `y_inicio_30s` | 1 si un episodio **nuevo empieza** en los próximos 30 s |
| `utilizable` | 1 si hay carga, pasó el arranque y el horizonte completo cabe en la corrida |
| `evaluable` | 1 si es utilizable **y** `bn_ahora = 0` (donde se evalúa la anticipación) |
| `umbral_ms` | Umbral aplicado a ese servicio |
| `split`, `grupo` | Reparto de entrenamiento/prueba y grupo de la campaña |

### 7.4 Causas de episodio

| Causa | Qué es |
| --- | --- |
| `inyeccion_gradual` | Provocado por una inyección con rampa en ese servicio |
| `inyeccion_abrupto` | Provocado por una inyección en salto |
| `propagacion` | El servicio se degrada porque lo hace otro del que depende (o que depende de él) |
| `natural` | Aparece por carga de tráfico (ráfagas o saturación) sin inyección en ese servicio |

En esta campaña: 148 episodios = 25 graduales + 24 abruptos + 76 de propagación + 23 naturales.

### 7.5 Vistas de evaluación

- **Vista A (utilizable):** todos los instantes utilizables. Incluye los que ya están en cuello de botella; el modelo podría acertar solo "copiando" la situación actual.
- **Vista B (evaluable):** solo instantes **sin** cuello de botella ahora. Ahí acertar significa **anticipar**. Es la vista principal para el informe.

---

## 8. Arquitectura 2: el pipeline de datos y código

```
boleteria_nube/
├── docker-compose.yml              ← levanta los 6 servicios (1 CPU, 256 MB cada uno)
├── servicios/<servicio>/           ← Dockerfile, requirements.txt, app/{common,main}.py
├── collector/collector.py          ← mide CPU/RAM (docker stats) y /metrics, cada 1 s
├── loadtest/
│   ├── locustfile.py               ← escenarios de tráfico
│   └── inyector.py                 ← provoca cuellos de botella
├── run_experiment.py               ← UNA corrida (recolector + locust + inyector)
├── run_campaign.py                 ← TODAS las corridas + consolidación
├── ml/
│   ├── graficar_corrida.py         ← gráfica de una corrida (4 paneles)
│   ├── etiquetar.py                ← pone etiquetas y episodios
│   ├── datos.py                    ← carga, ventanas y funciones de métricas
│   ├── baseline.py                 ← umbral CPU>80 % y persistencia
│   └── entrenar.py                 ← LSTM con validación cruzada
└── data/
    ├── <run_id>/                   ← una carpeta por corrida
    ├── dataset_metrics.csv, dataset_edges.csv, resumen_corridas.csv
    ├── dataset_etiquetado.csv, episodios.csv, umbrales.json
    └── resultados/                 ← tablas, gráficas y modelo entrenado
```

Flujo de datos:

```
 run_campaign.py ──► data/<run>/*.csv ──► consolidar ──► dataset_metrics.csv
                                                              │
                                                  etiquetar.py │
                                                              ▼
                                                 dataset_etiquetado.csv + episodios.csv
                                                              │
                                  ┌───────────────────────────┴─────────────┐
                              baseline.py                              entrenar.py
                            (umbral, persistencia)                       (LSTM)
                                                                              │
                                                                data/resultados/*.csv, *.png
```

---

## 9. Arquitectura 3: el baseline y el modelo LSTM

### 9.1 Qué entra al modelo

Para cada servicio y cada segundo *t*, el modelo ve una **ventana de 30 s** (de *t-29* a *t*). Cada segundo es un vector de **14 valores**:

| Parte | Cantidad | Detalle |
| --- | --- | --- |
| Métricas | 7 | `cpu, ram_mb, rps, lat_mean_ms, lat_p95_ms, cola, err_rps` (estandarizadas con la media y desviación del entrenamiento) |
| Dato faltante | 1 | 1 si ese segundo no tenía medición |
| Servicio | 6 | One-hot del servicio (cuál de los 6) |

Salida: **una probabilidad** (0 a 1) de que haya cuello de botella en los próximos 30 s.

### 9.2 ¿Qué es un LSTM, en simple?

Una red neuronal para **series de tiempo**. Lee la ventana segundo a segundo y mantiene una "memoria" interna que decide qué recordar y qué olvidar. Al final del segundo 30 resume lo visto, y una capa lineal convierte ese resumen en la probabilidad. Así puede captar **tendencias** (la cola sube, la latencia se acelera) que un umbral instantáneo no ve.

```
ventana 30 s × 14 valores ─► LSTM (1 capa, 64 unidades) ─► último paso ─► capa lineal ─► sigmoide ─► P(cuello de botella en 30 s)
```

### 9.3 Hiperparámetros (`ml/entrenar.py`)

| Parámetro | Defecto | Significado |
| --- | --- | --- |
| `--folds` | 4 | Número de particiones de la validación cruzada |
| `--epocas` | 15 | Máximo de pasadas por los datos |
| `--paciencia` | 3 | Épocas sin mejorar antes de parar (*early stopping*) |
| `--hidden` | 64 | Unidades del LSTM |
| `--batch` | 256 | Ejemplos por actualización |
| `--lr` | 0,001 | Tasa de aprendizaje |
| `--seed` | 42 | Semilla |
| `--cpu` | 80 | Umbral del baseline de CPU |
| `--sin-final` | no | No entrena el modelo final (más rápido) |

Pérdida: entropía cruzada binaria con `pos_weight`, porque los instantes "peligrosos" son pocos frente a los normales.

### 9.4 Cómo se evita hacer trampa (protocolo)

- **Validación cruzada agrupada por corrida:** ninguna corrida aparece a la vez en entrenamiento y prueba. Si no, el modelo memorizaría la corrida.
- Dentro de cada fold, ~15 % de las corridas de entrenamiento se usan como **validación** para decidir cuándo parar y **qué umbral de decisión** usar (el que maximiza F1 en vista B, probando de 0,05 a 0,95).
- La **normalización** se calcula solo con datos de entrenamiento.
- Se reúnen las predicciones de los 4 folds y se evalúa todo junto, además de por fold.
- Al final se entrena un **modelo final** con todas las corridas (`models/lstm_final.pt`), con el umbral = mediana de los umbrales de los folds.

### 9.5 Los métodos que se comparan

| Método | Regla | Para qué está |
| --- | --- | --- |
| **Umbral CPU>80 %** | Alarma si la CPU del servicio supera 80 % | Baseline que pide el enunciado |
| **Persistencia** | Alarma si ya hay cuello de botella ahora | Referencia mínima: la etiqueta deriva de la latencia, que también es entrada del modelo. En vista B vale 0 por construcción |
| **LSTM** | Alarma si P ≥ umbral elegido en validación | La primera versión de IA |

### 9.6 Archivos que produce `entrenar.py` (en `data/resultados/`)

| Archivo | Contenido |
| --- | --- |
| `metricas_A.csv`, `metricas_B.csv` | Precision, recall, F1 por método, vistas A y B |
| `por_escenario.csv` | Métricas por grupo de la campaña |
| `por_fold.csv` | Métricas por fold (para media ± desviación) |
| `lead_time_LSTM.csv`, `lead_time_threshold.csv` | Detalle por episodio: ¿detectado? ¿con cuánta anticipación? |
| `predicciones.csv` | Probabilidades por servicio y segundo |
| `config.json` | Configuración usada |
| `comparacion.png`, `ejemplo_timeline.png` | Gráficas de comparación y un ejemplo temporal |
| `models/lstm_final.pt` | Modelo final |

---

## 10. Métricas de evaluación

Se cuenta cada **servicio-segundo** como una decisión: "¿alarmo o no?".

| Resultado | Significado |
| --- | --- |
| **TP** (verdadero positivo) | Alarmé y de verdad venía un cuello de botella |
| **FP** (falso positivo) | Alarmé y no venía nada (falsa alarma) |
| **FN** (falso negativo) | No alarmé y sí venía (se escapó) |
| **TN** (verdadero negativo) | No alarmé y no venía nada |

| Métrica | Fórmula | Cómo interpretarla |
| --- | --- | --- |
| **Precision** | TP / (TP + FP) | De las alarmas que doy, qué fracción es real. Baja = muchas falsas alarmas |
| **Recall** | TP / (TP + FN) | De los cuellos de botella que venían, qué fracción detecté |
| **F1** | 2·P·R / (P + R) | Equilibrio entre ambas |
| **fpr** | FP / (FP + TN) | Fracción de instantes normales en que alarmé |
| **Prediction Lead Time** | segundos entre la **primera alarma** y el **inicio del episodio** (máximo 30 s) | Cuánto margen tiene un operador para reaccionar |

**Cómo se calcula el lead time aquí:** para cada episodio se mira la ventana de hasta 30 s **anterior** a su inicio en que el servicio estaba libre. Si hubo una alarma, el lead es el tiempo desde la primera alarma hasta el inicio. Se excluyen episodios con menos de 3 s libres antes.

**Trampa a evitar:** una alarma encendida *siempre* da lead de 30 s y recall alto, pero con precision baja. Por eso **el lead time nunca se reporta solo**: siempre junto a precision y fpr.

---

## 11. Comandos en orden

Desde la raíz del proyecto (con Docker Desktop abierto):

```powershell
# 0. Levantar los servicios (primera vez o tras cambiar código)
docker compose up -d --build

# 1. Ver el plan sin ejecutar
python run_campaign.py --dry-run

# 2. Prueba corta de un grupo
python run_campaign.py --only normal_base

# 3. Campaña completa + consolidar
python run_campaign.py            # o: --reps 2 para la versión ampliada

# 4. (opcional) Ver una corrida
python ml/graficar_corrida.py data/<run_id> --smooth 3

# 5. Etiquetar
python ml/etiquetar.py

# 6. Baseline
python ml/baseline.py

# 7. LSTM (prueba rápida y luego completo)
python ml/entrenar.py --epocas 3 --sin-final
python ml/entrenar.py
```

Para una variante de etiquetado más permisiva, sin pisar los archivos: `python ml/etiquetar.py --piso-ms 50 --sufijo _piso50` y luego `--sufijo _piso50` en `baseline.py` y `entrenar.py`.

---

## 12. Limitaciones que debes conocer

- **Toda inyección satura**, incluso la "leve": faltan degradaciones lentas y suaves (más anticipables).
- **La etiqueta sale de la latencia**, que además es una entrada del modelo; por eso existe la persistencia como referencia.
- **Lo abrupto no se puede anticipar:** sin señal previa en las métricas, ningún modelo puede avisar a tiempo.
- **Conjunto de prueba pequeño** (decenas de episodios por tipo; solo 2 corridas de `progresivo_base`).
- **Todo corre en una sola máquina:** hay ruido por contención y, bajo saturación fuerte, el `/metrics` a veces no responde (`scrape_ok = 0`).
- **Muestreo irregular** en las corridas de la primera campaña; el p95 es aproximado (cubetas de histograma).
- **Tráfico simulado y simple:** todas las peticiones usan el mismo usuario.
- **Resultados actuales (resumen):** el LSTM supera al umbral CPU>80 % en recall y F1 (≈ 0,71 y 0,40 contra 0,02 y 0,04), con precision baja (≈ 0,29) y anticipación real de unos 2–11 s en inyecciones graduales; el umbral gana en `progresivo_base` (solo 2 corridas).
