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
13. [Resultados del modelo final (v9)](#13-resultados-del-modelo-final-v9)
14. [Demo](#14-demo)

---

## 1. La idea completa en un minuto

1. Tenemos **6 microservicios** (una boletería) que se llaman entre sí.
2. Les mandamos **tráfico simulado** (Locust) y, a propósito, **les provocamos cuellos de botella** (inyector) en momentos conocidos.
3. Un **recolector** mide cada servicio cada segundo (CPU, RAM, latencia, peticiones por segundo, cola, errores) y guarda todo en CSV.
4. Un script de **etiquetado** decide, mirando la latencia, en qué instantes cada servicio estaba en cuello de botella **por su propio trabajo** (no por esperar a otro servicio), y marca los **30 segundos anteriores** como "se viene un cuello de botella".
5. Comparamos dos formas de predecirlo:
   - **Baseline:** una regla fija ("alarma si CPU > 80 %").
   - **LSTM:** una red neuronal que mira los últimos 30 s de métricas del servicio **y de los servicios a los que llama**, y estima la probabilidad de que haya un cuello de botella en los próximos 30 s.
6. Medimos con **Precision, Recall, F1, falsas alarmas por hora y Prediction Lead Time** (cuántos segundos antes avisamos, contado de forma honesta).

**Resultado del modelo final (v9, ensamble de 3 LSTM, 116 corridas):** se probó con **36 corridas de calendario aleatorio** (fallos y ráfagas en orden y momentos al azar), donde el modelo no puede "adivinar" por el horario del experimento. Ahí v9 logra F1 0,21 contra 0,10 de la regla de CPU, avisa el 88 % de las degradaciones graduales (mediana 8 s antes) y, a la misma tasa de falsas alarmas que el modelo anterior (v8), es mejor en casi todo: precisión 0,63 contra 0,58, recall 0,11 contra 0,08. En la validación cruzada de las 116 corridas: F1 0,31 contra 0,11, precisión 0,55, 5,2 falsas alarmas por hora de servicio. Detalle en la [sección 13](#13-resultados-del-modelo-final-v9).

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

Sirve para ver **dónde** se propaga un problema. En P1 se usa en el **etiquetado**: con ella se calcula el **tiempo propio** de cada servicio (latencia media menos lo que tardaron sus llamadas a otros), que distingue un cuello de botella real de un servicio que solo está esperando (sección 7). El LSTM no lee `edges.csv` directamente; ve el estado de sus dependencias a través de `metrics.csv` (sección 9.1).

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
| `OK` | Marca de corrida completa y válida (la campaña salta las que ya la tienen) |
| `CONTAMINADA` | Solo si la corrida no pasó el control de calidad (sección 6.4); no recibe `OK` y se repite |

Y tras consolidar, en `data/`: `dataset_metrics.csv`, `dataset_edges.csv`, `resumen_corridas.csv` (una fila por corrida) y `calidad_corridas.csv` (resultado del control de calidad de cada corrida).

---

## 6. Parámetros de los experimentos y las campañas

### 6.1 Los escenarios de tráfico (`loadtest/locustfile.py`)

Locust crea **usuarios virtuales en bucle cerrado**: cada usuario hace una petición, espera la respuesta y repite. Un usuario genera unas 1,25 req/s. Si el sistema se lentifica, los usuarios esperan más y generan menos tráfico (por eso el sistema "se frena solo").

| Escenario | Cómo es la carga | Para qué sirve |
| --- | --- | --- |
| **normal** | Usuarios constantes (`--base-users`, 15 por defecto) | Línea base y base para inyectar cuellos de botella |
| **progresivo** | Crece de 5 usuarios hasta `--peak-users` (60) y vuelve a bajar, en forma de triángulo (≈65 % del tiempo subiendo) | Saturación natural y gradual |
| **burst** | Carga base con **ráfagas** a `--peak-users` (70): la primera a los 60 s, luego cada 120 s, de 25 s de duración. Con `BURST_TIMES` las ráfagas empiezan en los segundos indicados (calendario irregular) | Picos bruscos de tráfico |

Parámetros de entorno de Locust: `SCENARIO`, `DURATION`, `BASE_USERS`, `PEAK_USERS`, `BURST_START=60`, `BURST_EVERY=120`, `BURST_LEN=25`, `BURST_TIMES` (lista de inicios separados por coma; si está, reemplaza a `BURST_START`/`BURST_EVERY`).

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
| `--burst-times` | ninguno | Inicios de las ráfagas en burst, p. ej. `67,179,329` (calendario irregular) |
| `--burst-len` | 25 | Duración de cada ráfaga (s) |

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
| `--repetir-contaminadas` | no | Quita la marca `OK` a las corridas que no pasan el control de calidad, para repetirlas |
| `--aceptar-contaminadas` | no | Marca `OK` una corrida nueva aunque no pase el control de calidad |

**Control de calidad.** Todo corre en una sola máquina, así que si la laptop está ocupada (otro programa, un entrenamiento, modo ahorro de energía) los servicios gastan más CPU por el mismo trabajo y la corrida queda contaminada. Al terminar cada corrida se mide la **CPU total del sistema por petición antes de la primera inyección**; si supera **1,5 ×** la de `normal_base`, la corrida no recibe `OK`, deja un archivo `CONTAMINADA` y se repite la próxima vez. Así se detectaron y repitieron 10 corridas grabadas con la máquina al doble de su costo normal. **No entrenes ni corras nada pesado mientras haya una campaña activa.**

La **verificación previa** comprueba que los 6 servicios responden, que una reserva atraviesa toda la cadena y que los contenedores tienen el `common.py` nuevo (métricas de dependencias y cubetas finas). Así no se pierden horas con código viejo.

Si interrumpes con Ctrl+C, consolida lo terminado; al reanudar, **salta** las corridas con marca `OK` y repite la incompleta.

### 6.5 El plan de la campaña (`--reps 1` = 40 corridas, ≈ 4,7 h; `--reps 2` = 80 corridas)

| Grupo | Corridas | Escenario | Duración | Usuarios | Inyección |
| --- | --- | --- | --- | --- | --- |
| `normal_base` | 2 | normal | 300 s | 15 | Ninguna (negativos limpios; de aquí salen los umbrales) |
| `normal_inj` | 6 (una por servicio) | normal | 360 s | 15 | Gradual `s:60:90:150:40` (rampa de 40 s) y abrupta `s:220:60:150` |
| `burst_inj` | 6 (una por servicio) | burst | 360 s | 15 base, pico 70 | Gradual `s:95:75:150:30` y abrupta `s:215:75:150`, esquivando las ráfagas (que caen en 60–85, 180–205 y 300–325 s) |
| `progresivo_base` | 2 | progresivo | 360 s | 5 → 60 → 5 | Ninguna (saturación natural de `reservaciones`) |
| `var_inj` | 12 (6 leves + 6 fuertes) | normal | 360 s | leve 15 / fuerte 20 | Leve: `cpu_load` 90, rampa 40 s. Fuerte: `cpu_load` 250, rampa 20 s. Cada una con evento gradual (60 s) y abrupto (220 s) |
| `suave_inj` | 12 (6 con carga 25 + 6 con carga 40) | normal | 360 s | 15 | Gradual `s:60:120:carga:90` (rampa de 90 s) y abrupta `s:230:90:carga`. Algunas no llegan a saturar: dan degradaciones lentas y negativos difíciles |

Cada corrida con inyección contiene **dos eventos**: uno gradual (con rampa) y uno abrupto (salto). Por eso en los resultados aparecen las causas `inyeccion_gradual` e `inyeccion_abrupto`.

**Por qué hay un grupo de cada tipo**

- `normal_base`, `progresivo_base`: ejemplos **sin inyección**, para que el modelo aprenda cómo es lo normal y no solo lo anómalo.
- `normal_inj`, `burst_inj`: cuellos de botella provocados con tráfico estable y con ráfagas (más ruido).
- `var_inj`: variedad de severidad, para no aprender un único patrón.
- `suave_inj`: degradaciones lentas, las únicas realmente anticipables, y casos que no llegan a saturar.

Los primeros datos son la campaña con `--reps 2`: **80 corridas**, todas dentro del control de calidad, con 144 inyecciones (72 graduales y 72 abruptas).

### 6.6 Campaña con calendario aleatorio (grupos `rnd_*`)

En el plan fijo, el fallo abrupto llega siempre ~2,5 min después del gradual en el mismo servicio y las ráfagas caen cada 120 s. Un modelo puede aprender ese **horario** en vez de las métricas (se vio con la ventana de 60 s, sección 9.3). Para medirlo y evitarlo se agregaron 40 corridas con el calendario sorteado (`python run_campaign.py --only rnd_normal,rnd_burst,rnd_progresivo`):

| Grupo | Corridas | Escenario | Qué se sortea |
| --- | --- | --- | --- |
| `rnd_normal` | 24 | normal, 300–420 s, 12–20 usuarios | 1 a 3 inyecciones sin solaparse, desde el segundo 45: servicio, orden, tipo (gradual o abrupta, 50/50), intensidad (25, 40, 90, 150 o 250), duración y rampa |
| `rnd_burst` | 10 | burst, 360–420 s, pico 55–85 | 0 a 2 inyecciones como arriba, y además 2 a 4 ráfagas de 15–35 s en momentos sorteados (desde el segundo 60, separadas 40–120 s, vía `--burst-times`) |
| `rnd_progresivo` | 6 | progresivo, 300–420 s, pico 45–90 | 0 o 1 inyección sorteada sobre el crecimiento natural |

Cada corrida usa su propia semilla (`grupo-número`), así que el plan es reproducible. Hay 54 inyecciones (23 graduales) en todos los órdenes posibles. **36 de las 40** pasaron el control de calidad; las otras 4 (`rnd_normal_10`, `rnd_progresivo_02` a `04`) salieron contaminadas (la máquina estaba ~2 × más lenta ese día) y no se usan. En total, el modelo final se entrena con **116 corridas y 275 episodios**.

---

## 7. Etiquetado: cómo decidimos que hay un cuello de botella

Script: `ml/etiquetar.py`. Opera por servicio y por corrida.

### 7.1 Definición

1. **Umbral de latencia por servicio** = `max(3 × mediana de lat_p95 en corridas normal_base, 150 ms)`.
   Resultado real: 150 ms para `eventos`, `notificaciones`, `usuarios`, `ticket` y `pago`; 292,5 ms para `reservaciones` (es más lento por naturaleza, porque orquesta).
2. Un instante está **en cuello de botella** (`bn_ahora = 1`) si `lat_p95 > umbral` **o** el servicio no respondió a `/metrics` (`scrape_ok = 0`).
   **Etiqueta de cuello de botella propio** (`--solo-propio`, la que usa el modelo final): en los servicios que llaman a otros (`reservaciones`, `pago`, `ticket`) se exige además que el **tiempo propio** supere `max(3 × su mediana en normal_base, 50 ms)`. El tiempo propio es la latencia media menos la suma de las llamadas a dependencias (de `edges.csv`). Si un servicio solo está lento porque espera a otro, se marca `afectado = 1` y **no** cuenta como cuello de botella.
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
| `--solo-propio` | no | Usa la etiqueta de cuello de botella propio (se guarda con `--sufijo _propio`) |
| `--piso-propio-ms` | 50 | Tiempo propio mínimo para considerar que el servicio es el cuello de botella |

**Por qué la etiqueta propia.** Con la etiqueta original, en los episodios de "propagación" solo el 1–2 % de la latencia de `reservaciones`, `pago` y `ticket` era trabajo propio (~8 ms, igual que en reposo): estaban esperando al servicio que de verdad se saturaba. 110 de 118 episodios de propagación eran heredados. Eso hacía que `reservaciones` tuviera la etiqueta positiva en un tercio de su tiempo tranquilo sin ninguna diferencia en sus métricas, y el modelo aprendía a alarmar casi siempre. Con la etiqueta propia, las falsas alarmas bajaron de 17,5 a 5 por hora.

### 7.3 Columnas de `dataset_etiquetado.csv` (las más importantes)

| Columna | Significado |
| --- | --- |
| `bn_ahora` | 1 si en este instante ya hay cuello de botella |
| `afectado` | 1 si el servicio está lento solo por esperar a una dependencia (con `--solo-propio`) |
| `lat_propia_ms` | Tiempo propio: latencia media menos las llamadas a dependencias |
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

Con las 80 corridas y la etiqueta propia hay **195 episodios** = 65 graduales + 64 abruptos + 44 naturales + 22 de propagación (con la etiqueta original eran 397, la mayoría de propagación). Los naturales son todos de `reservaciones`: picos de `burst_inj` y saturación en `progresivo_base`.

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
│   ├── entrenar.py                 ← LSTM con validación cruzada
│   ├── postproceso.py              ← suavizado e híbrido de alarmas, sin reentrenar
│   ├── por_escenario.py            ← métricas por escenario de carga (normal / progresivo / burst)
│   ├── barrido.py                  ← barrido de hiperparámetros y balanceo (varias semillas)
│   ├── ensamble.py                 ← ensamble de semillas: evalúa y arma el modelo final (v8, v9)
│   ├── evaluar_externo.py          ← mide un modelo solo en un grupo de corridas (p. ej. las rnd_*)
│   ├── pipeline_v9.sh              ← campaña aleatoria → etiquetado → v8 en rnd_* → v9 → tablas
│   ├── demo.py                     ← demo en vivo y en modo repetición
│   └── verificar_vivo.py           ← corridas nuevas en vivo + evaluación con el mismo protocolo
└── data/                           ← (no se sube a git: se regenera con los scripts)
    ├── <run_id>/                   ← una carpeta por corrida
    ├── dataset_metrics.csv, dataset_edges.csv, resumen_corridas.csv, calidad_corridas.csv
    ├── dataset_etiquetado[_propio].csv, episodios[_propio].csv, umbrales[_propio].json
    ├── resultados_v9/              ← modelo final: s1, s2, s3 + modelo_ensamble.json + predicciones
    ├── resultados_v8/              ← modelo anterior (mismo ensamble, 80 corridas de calendario fijo)
    ├── resultados_v7/              ← un solo LSTM
    ├── externo_<modelo>_rnd.csv    ← resultados de evaluar_externo.py en las corridas aleatorias
    ├── barrido/                    ← resumen del barrido (resumen.csv, detalle.csv) y los registros de sus 40 entrenamientos
    └── vivo/                       ← corridas de la verificación en vivo
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
                                                       data/<salida>/*.csv, *.png ──► postproceso.py
```

---

## 9. Arquitectura 3: el baseline y el modelo LSTM

### 9.1 Qué entra al modelo

Para cada servicio y cada segundo *t*, el modelo ve una **ventana de 30 s** (de *t-29* a *t*). En el modelo final (cada uno de los 3 LSTM de v9, con la misma configuración que v7 y v8: `--contexto --tendencias`) cada segundo es un vector de **27 valores**:

| Parte | Cantidad | Detalle |
| --- | --- | --- |
| Métricas propias | 7 | `cpu, ram_mb, rps, lat_mean_ms, lat_p95_ms, cola, err_rps` (las de cola larga en log1p; estandarizadas con la media y desviación del entrenamiento) |
| Contexto de dependencias | 3 | `cpu_dep, cola_dep, lat_p95_dep`: el máximo entre los servicios a los que llama, directa o indirectamente (0 en los servicios hoja). Así `pago` ve venir la saturación de `ticket` o `notificaciones` |
| Tendencias | 10 | `d5_*` y `d15_*`: cuánto cambiaron en 5 y 15 s `cpu`, `cola`, `rps`, `lat_p95_ms` y `cpu_dep` |
| Dato faltante | 1 | 1 si ese segundo no tenía medición |
| Servicio | 6 | One-hot del servicio (cuál de los 6) |

Sin `--contexto --tendencias` el modelo usa solo 14 valores (las 7 métricas, el dato faltante y el servicio). Las entradas nuevas duplicaron los episodios avisados con 15 s o más de anticipación (de 11 % a 25 %).

Salida: **una probabilidad** (0 a 1) de que haya cuello de botella en los próximos 30 s.

### 9.2 ¿Qué es un LSTM, en simple?

Una red neuronal para **series de tiempo**. Lee la ventana segundo a segundo y mantiene una "memoria" interna que decide qué recordar y qué olvidar. Al final del segundo 30 resume lo visto, y una capa lineal convierte ese resumen en la probabilidad. Así puede captar **tendencias** (la cola sube, la latencia se acelera) que un umbral instantáneo no ve.

```
ventana 30 s × 27 valores ─► LSTM (1 capa, 64 unidades) ─► último paso ─► capa lineal ─► sigmoide ─► P(cuello de botella en 30 s)
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
| `--sufijo` | vacío | Qué etiquetado usar (`_propio` = cuello de botella propio) |
| `--contexto` | no | Añade las métricas de las dependencias (`*_dep`) |
| `--tendencias` | no | Añade los cambios en 5 y 15 s (`d5_*`, `d15_*`) |
| `--criterio` | `fbeta` | Métrica que maximiza el umbral en validación (`fbeta` o `f1`) |
| `--beta` | 0,5 | β de F-beta (< 1 favorece la precisión) |
| `--fpr-max` | ninguno | Solo umbrales con tasa de falsas alarmas ≤ este valor |
| `--objetivo` | `y_30s` | `y_inicio_30s` entrena solo en segundos sin cuello de botella (se probó; no ayudó) |
| `--sin-latencia` | no | Quita las entradas de latencia (prueba de que el modelo no solo lee la etiqueta) |
| `--excluir-contaminadas` | no | Deja fuera las corridas marcadas en `calidad_corridas.csv` |
| `--salida` | `resultados` | Subcarpeta de `data/` donde guardar los resultados |
| `--seed-folds` | = `--seed` | Semilla del reparto en folds. Fijarla (42) permite comparar variantes con las mismas corridas de prueba |
| `--balanceo` | `pos_weight` | `ninguno` o `submuestreo` (todos los positivos + `--ratio` negativos por positivo en cada época) |
| `--parada` | `loss` | Métrica de la parada temprana: `loss` (BCE), `loss_pond` (BCE ponderada) o `ap` (average precision) |
| `--capas`, `--dropout`, `--wd`, `--ventana` | 1, 0, 0, 30 | Capas del LSTM, dropout, weight decay y segundos de historia |

**Qué se aprendió con el barrido** (`ml/barrido.py`, 13 variantes × 3 semillas, mismos folds): ninguna variante de balanceo, parada temprana, tamaño, capas, lr, weight decay o lote mejoró a la configuración de v7 por más que la variación entre semillas (±0,02 de F1). La **ventana de 60 s** subió el F1 a 0,56, pero al compararla sobre los mismos segundos y por causa se vio que la mejora venía de **memorizar el calendario del experimento**: "avisaba" el 58 % de los fallos abruptos con 21 s de anticipación (imposible: no tienen señal previa) porque veía el final del fallo gradual, y anticipaba los picos de burst, que caen siempre cada 120 s. Se descartó. Lo que sí sirvió fue el **ensamble de semillas** (sección 13).

**La ventana de 60 s, otra vez, con el calendario aleatorio** (`data/resultados_v9w60/`): con las 116 corridas su F1 en validación cruzada sube a 0,42 (v9: 0,31), pero el desglose confirma el diagnóstico. En las 80 corridas de calendario fijo sigue "avisando" el **48–52 % de los fallos abruptos**, imposible sin señal previa. En las 36 corridas aleatorias, donde no hay horario que memorizar, a igual tasa de falsas alarmas **no mejora a v9** (F1 0,22 contra 0,24 con ~3,3 falsas alarmas/h; precisión 0,40 contra 0,56) y sus abruptos avisados bajan al nivel del azar (7 %). Se descartó de nuevo.

Pérdida: entropía cruzada binaria con `pos_weight`, porque los instantes "peligrosos" son pocos frente a los normales.

### 9.4 Cómo se evita hacer trampa (protocolo)

- **Validación cruzada agrupada por corrida:** ninguna corrida aparece a la vez en entrenamiento y prueba. Si no, el modelo memorizaría la corrida.
- Dentro de cada fold, ~15 % de las corridas de entrenamiento se usan como **validación** para decidir cuándo parar y **qué umbral de decisión** usar (el que maximiza **F0,5** en vista B, probando de 0,05 a 0,95; F0,5 pesa más la precisión y reduce las falsas alarmas. Elegirlo por F1 dio umbrales inestables).
- La **normalización** se calcula solo con datos de entrenamiento.
- Se reúnen las predicciones de los 4 folds y se evalúa todo junto, además de por fold.
- Al final se entrena un **modelo final** con todas las corridas (`models/lstm_final.pt`), con el umbral = mediana de los umbrales de los folds.

### 9.5 Los métodos que se comparan

| Método | Regla | Para qué está |
| --- | --- | --- |
| **Umbral CPU>80 %** | Alarma si la CPU del servicio supera 80 % | Baseline que pide el enunciado |
| **Persistencia** | Alarma si ya hay cuello de botella ahora | Referencia mínima: la etiqueta deriva de la latencia, que también es entrada del modelo. En vista B vale 0 por construcción |
| **LSTM** | Alarma si P ≥ umbral elegido en validación | La primera versión de IA |

### 9.6 Archivos que produce `entrenar.py` (en `data/<--salida>/`; el modelo final está en `data/resultados_v9/`)

| Archivo | Contenido |
| --- | --- |
| `metricas_A.csv`, `metricas_B.csv` | Precision, recall, F1 por método, vistas A y B |
| `por_escenario.csv` | Métricas por grupo de la campaña |
| `por_fold.csv` | Métricas por fold (para media ± desviación) |
| `lead_time_LSTM.csv`, `lead_time_threshold.csv` | Detalle por episodio: ¿detectado? ¿con cuánta anticipación? |
| `predicciones.csv` | Probabilidades por servicio y segundo |
| `config.json` | Configuración usada |
| `comparacion.png`, `ejemplo_timeline.png` | Gráficas de comparación y un ejemplo temporal |
| `models/lstm_final.pt` | Modelo final (pesos, normalización, umbral y columnas usadas) |
| `postproceso.csv` | Lo genera `postproceso.py`: métricas con suavizado de la alarma y del híbrido LSTM o CPU |

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
| **F0,5** | F-beta con β = 0,5 | Como F1, pero pesa más la precision; con ella se elige el umbral |
| **fpr** | FP / (FP + TN) | Fracción de instantes normales en que alarmé |
| **Falsas alarmas por hora** | encendidos de alarma sin cuello de botella en los 30 s siguientes / horas de servicio | Cuántas veces molestaría a un operador; cuenta encendidos, no segundos |
| **Prediction Lead Time** | segundos entre el **encendido de la alarma** y el **inicio del episodio** (máximo 30 s) | Cuánto margen tiene un operador para reaccionar |

**Lead time honesto (el que se reporta):** para cada episodio se mira la ventana de hasta 30 s **anterior** a su inicio en que el servicio estaba libre. Solo cuenta una alarma que **se enciende** dentro de esa ventana **después de estar ≥ 5 s apagada**; el lead es el tiempo desde ese encendido hasta el inicio. Una alarma que ya estaba encendida no suma anticipación. Se excluyen episodios con menos de 3 s libres antes. Las columnas `*_bruto` guardan el cálculo anterior (cualquier segundo con alarma) solo para comparar.

**Trampa a evitar:** una alarma encendida *siempre* daba un lead de 30 s con el cálculo bruto. Así se vio en la primera versión: en propagación, 66 de 72 detecciones tenían la alarma encendida toda la ventana. Por eso el lead se cuenta de forma honesta y **nunca se reporta solo**: siempre junto a precision y falsas alarmas por hora.

**Nivel del azar:** con *F* falsas alarmas por hora, una ventana de 30 s recibe una alarma casual con probabilidad ≈ *F* × 30 / 3600. Con 2,3 por hora (v8) es ≈ 2 %; con 5,2 por hora (v9) ≈ 4 %; con 9 por hora (v7) era ≈ 8 %. Una tasa de detección cercana a ese valor no es anticipación.

**Falsas alarmas: por servicio y por sistema.** La cifra de las tablas es **por hora de servicio**. La demo muestra los 6 servicios a la vez, así que en pantalla se ven **6 veces más**: con v9, ~31 por hora en todo el sistema (una cada ~2 min) en la validación cruzada y ~19 en las corridas aleatorias; con v8, ~14; con v7 eran ~54 (casi una por minuto).

**Precisión por segundo y por alarma.** La precisión de las tablas cuenta segundos con alarma. Contando cada encendido de la alarma, la de v7 bajaba de 0,42 a 0,29, porque sus alarmas falsas eran parpadeos cortos y las correctas, largas. Con v8 era **0,52 por alarma** (86 de 166 encendidos); con v9 es 0,34 en la validación cruzada y 0,40 en las corridas aleatorias. Es lo que percibe un operador.

**Comparar modelos a igual tasa de falsas alarmas.** Cada modelo elige su umbral en validación, así que uno puede tener más recall solo porque alarma más. Para compararlos de forma justa se mueve el umbral de ambos hasta que tengan las mismas falsas alarmas por hora y se mira el resto (sección 13.1).

**Post-proceso (`ml/postproceso.py`):** sin reentrenar, mide el efecto de **suavizar** la alarma (que suene solo si el modelo dijo "sí" en 3 de los últimos 5 s) y del **híbrido** "LSTM o CPU > 80 %". El suavizado reduce las falsas alarmas a la mitad pero retrasa el aviso; el híbrido no aporta.

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

# 3. Campaña completa + consolidar (80 corridas, ≈ 9,5 h; se puede cortar con Ctrl+C y reanudar)
python run_campaign.py --reps 2
python run_campaign.py --reps 2 --repetir-contaminadas   # si alguna no pasó el control de calidad

# 4. (opcional) Ver una corrida
python ml/graficar_corrida.py data/<run_id> --smooth 3

# 5. Etiquetar (original y de cuello de botella propio)
python ml/etiquetar.py
python ml/etiquetar.py --solo-propio --sufijo _propio

# 6. Baseline
python ml/baseline.py --sufijo _propio

# 3b. Campaña con calendario aleatorio (40 corridas, ≈ 4,9 h)
python run_campaign.py --only rnd_normal,rnd_burst,rnd_progresivo

# 7. Modelo final v9: 3 LSTM con la configuración de v7 y distinta semilla (mismos folds), y su ensamble
python ml/entrenar.py --sufijo _propio --contexto --tendencias --seed-folds 42 --seed 1 --salida resultados_v9/s1
python ml/entrenar.py --sufijo _propio --contexto --tendencias --seed-folds 42 --seed 2 --salida resultados_v9/s2
python ml/entrenar.py --sufijo _propio --contexto --tendencias --seed-folds 42 --seed 3 --salida resultados_v9/s3
python ml/ensamble.py --dirs resultados_v9/s1 resultados_v9/s2 resultados_v9/s3 --salida resultados_v9

# 8. Tablas del modelo final: post-proceso y por escenario de carga
python ml/postproceso.py --sufijo _propio --resultados resultados_v9
python ml/por_escenario.py --resultados resultados_v9

# 8b. Solo las corridas de calendario aleatorio: v8 (que nunca las vio) y v9 (validación cruzada)
python ml/evaluar_externo.py --modelo data/resultados_v8/modelo_ensamble.json --nombre v8
python ml/evaluar_externo.py --resultados resultados_v9 --nombre v9

# 9. (opcional) Barrido de hiperparámetros, ~4 h con la máquina libre
python ml/barrido.py

# 10. (opcional) Verificación en vivo: 6 corridas nuevas (~45 min, Docker arriba) y su evaluación
python ml/verificar_vivo.py correr
python ml/verificar_vivo.py evaluar        # v9; para otro modelo: --modelo ... --resultados ...
```

Los pasos 3b a 8b, más la prueba de la ventana de 60 s, están encadenados en `bash ml/pipeline_v9.sh` (salta las corridas ya hechas; registro en `data/pipeline_v9.log`).

Cada entrenamiento tarda ~5–7 min con la máquina libre. **No entrenes mientras corre una campaña** (contamina las mediciones).

Para una variante de etiquetado más permisiva, sin pisar los archivos: `python ml/etiquetar.py --piso-ms 50 --sufijo _piso50` y luego `--sufijo _piso50` en `baseline.py` y `entrenar.py`.

---

## 12. Limitaciones que debes conocer

- **No se alcanzan los 30 s de anticipación:** las métricas solo empiezan a moverse unos segundos antes del problema. La mediana real es ~9 s en fallos graduales. En el crecimiento progresivo el modelo sí llega a avisar con más de 30 s, pero la métrica estricta no lo cuenta (solo cuenta encendidos dentro de los 30 s previos).
- **Lo abrupto no se puede anticipar:** sin señal previa en las métricas, ningún modelo puede avisar a tiempo. Una vez iniciado, tanto el LSTM como la regla de CPU lo detectan en ~1 s.
- **El calendario del experimento se puede memorizar:** en las 80 corridas de calendario fijo el fallo abrupto llega siempre después del gradual en el mismo servicio y los picos de burst caen cada 120 s. Con las 36 corridas de calendario aleatorio (sección 6.6) se midió cuánto de eso había: v8, en corridas que nunca vio, avisó solo el 7 % de los abruptos y el 4 % de los picos, al nivel del azar. **Las cifras honestas del proyecto son las de esas corridas (sección 13.1)**, más bajas que las de la validación cruzada. En las corridas de calendario fijo, las cifras de cualquier modelo siguen algo infladas por ese patrón.
- **Recall bajo en corridas realistas:** con calendario aleatorio v9 detecta ~13 % de los segundos de riesgo. Lo que sí anticipa de forma fiable son las degradaciones graduales (88 %); los fallos abruptos, la propagación y la mayoría de las ráfagas solo se detectan cuando ya empezaron.
- **Falsas alarmas:** 5,2 por hora **por servicio** con v9 en la validación cruzada (umbral de cada fold) y 3,1 en las corridas aleatorias con el umbral final, que en el sistema completo (6 servicios) son ~19–31 por hora. Siguen siendo la principal debilidad para un uso real. Subiendo el umbral a 0,91 bajan a ~1,5 por hora de servicio, con algo menos de recall.
- **Faltan 4 corridas aleatorias** (`rnd_normal_10`, `rnd_progresivo_02` a `04`) que salieron contaminadas; se pueden completar otro día con `python run_campaign.py --only rnd_normal,rnd_burst,rnd_progresivo --repetir-contaminadas` si el control de calidad vuelve a dar ~1 ×.
- **La etiqueta sale de la latencia**, que además es una entrada del modelo. Sin las entradas de latencia el LSTM conserva ~85 % de su F1, así que no se limita a leer la etiqueta.
- **Pocos episodios en algunos escenarios:** el crecimiento progresivo tiene 7 corridas (13 episodios).
- **Todo corre en una sola máquina:** hay ruido por contención (de ahí el control de calidad) y, bajo saturación fuerte, el `/metrics` a veces no responde (`scrape_ok = 0`).
- **El p95 es aproximado** (cubetas de histograma).
- **Tráfico simulado y simple:** todas las peticiones usan el mismo usuario (no genera contención: ningún servicio bloquea por usuario).
- **Decisiones de diseño sin análisis de sensibilidad completo:** el factor 3 × la mediana y los pisos de 150 ms y 50 ms.

---

## 13. Resultados del modelo final (v9)

**Modelo v9:** un **ensamble de 3 LSTM** con la misma configuración que v7 y v8 (contexto de dependencias, tendencias, etiqueta de cuello de botella propio, ventana de 30 s), entrenado ahora con **116 corridas**, incluidas las 36 de calendario aleatorio, con las semillas 1, 2 y 3. La probabilidad es el **promedio de los tres** y la alarma suena si supera **0,875** (mediana de los umbrales por fold del ensamble, la misma regla que dio 0,91 en v8). Archivo: `data/resultados_v9/modelo_ensamble.json`, que apunta a `s1`, `s2` y `s3/models/lstm_final.pt`.

**Por qué un ensamble.** Con una sola red, la semilla cambia mucho el umbral que elige la validación (v7 eligió 0,55 en un fold y disparó sus falsas alarmas). Promediar 3 modelos reduce esa variación y cancela los parpadeos sueltos de cada uno.

### 13.1 La prueba que importa: 36 corridas de calendario aleatorio

En estas corridas los fallos y las ráfagas llegan en orden y momentos sorteados, así que un modelo no puede acertar por memorizar el horario del experimento. **v8 nunca las vio** (se entrenó antes de que existieran): es una prueba externa. Para **v9** se usan las predicciones de su validación cruzada: cada corrida la predijo un modelo que no la vio. Se reproduce con `ml/evaluar_externo.py`.

| Modelo (umbral) | Precisión | Recall | F1 | Falsas alarmas/h (servicio / sistema) | Precisión por alarma | Graduales avisados (mediana) | Abruptos avisados | Ráfagas y saturación natural avisadas | Propagación avisada |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v8 (0,91) | 0,58 | 0,08 | 0,14 | 1,6 / 9,5 | 0,45 | 94 % (7 s) | 7 % | 4 % | 0 % |
| **v9 (0,91, misma tasa de falsas alarmas que v8)** | **0,63** | **0,11** | **0,19** | 1,5 / 9,2 | **0,54** | 88 % | 15 % | 21 % | 8 % |
| **v9 (0,875, umbral del modelo final)** | 0,57 | 0,13 | 0,21 | 3,1 / 19 | 0,40 | 88 % | 19 % | 29 % | 15 % |
| v9 (umbral de cada fold, 0,80–0,90) | 0,41 | 0,14 | 0,21 | 6,2 / 37 | 0,27 | 88 % (8 s) | 15 % | 33 % | 8 % |
| CPU > 80 % | 0,37 | 0,06 | 0,10 | 2,1 / 12 | 0,42 | 38 % (4,5 s) | 11 % | 4 % | 8 % |

Episodios: 16 graduales, 27 abruptos, 24 de ráfagas o saturación natural y 13 de propagación.

- **Lo que v8 había aprendido del calendario no se sostiene:** en corridas aleatorias avisa el 7 % de los abruptos y el 4 % de los picos, al nivel del azar (~1,3 % por ventana con su tasa de falsas alarmas). Su anticipación de las degradaciones graduales, en cambio, es real: 94 %.
- **A igual tasa de falsas alarmas, v9 es mejor que v8 en casi todo** (solo pierde un episodio gradual de 16): más precisión, más recall, mejor precisión por alarma, y empieza a anticipar algunas ráfagas (21 % contra 4 %), que ahora llegan en momentos imprevisibles, así que es por las métricas.
- **El umbral decide el equilibrio:** con el umbral del modelo final (0,875) v9 avisa más, pero con el doble de falsas alarmas que con 0,91. En la validación cruzada un fold eligió 0,80, y por eso esa fila tiene 6,2 falsas alarmas por hora.

### 13.2 Validación cruzada completa (116 corridas)

| Método | Precisión | Recall | F1 | Falsas alarmas/h (por servicio) | Precisión por alarma | Graduales avisados (mediana) | Episodios con ≥ 15 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **v9 (ensamble, umbral de cada fold)** | **0,55** | **0,21** | **0,31** | 5,2 | 0,34 | **81 % (9 s)** | 13 % |
| v9 con umbral único 0,875 | 0,62 | 0,21 | 0,31 | 3,7 | 0,41 | 81 % | — |
| Cada semilla sola (1, 2, 3) | 0,44–0,52 | 0,23–0,24 | 0,30–0,32 | 5,7–7,7 | — | 83–85 % | 13–14 % |
| CPU > 80 % | 0,44 | 0,06 | 0,11 | 2,6 | 0,42 | 38 % (4 s) | 4 % |

Estas cifras **no se comparan directamente con las de v8** (F1 0,36 y 2,3 falsas alarmas/h con 80 corridas): ahora entran las 36 corridas aleatorias, más difíciles y sin horario que aprovechar. La comparación justa es la de la sección 13.1.

### 13.3 Por escenario de carga (validación cruzada, 116 corridas)

| Escenario | Corridas | Episodios | F1 v9 / CPU | Falsas alarmas/h v9 / CPU | Avisados antes de empezar v9 / CPU | Anticipación mediana v9 / CPU |
| --- | --- | --- | --- | --- | --- | --- |
| Carga normal (con fallos inyectados) | 87 | 162 | **0,23** / 0,08 | 4,7 / 2,0 | **40 %** / 21 % | **9 s** / 5 s |
| Crecimiento progresivo | 7 | 13 | **0,80** / 0,51 | **2,9** / 16,1 | **62 %** / 15 % | **22 s** / 13,5 s |
| Burst de solicitudes | 22 | 100 | **0,35** / 0,01 | 7,8 / 0,4 | **28 %** / 7 % | **9 s** / 2 s |

Se reproduce con `python ml/por_escenario.py --resultados resultados_v9`.

### 13.4 Verificación en vivo (6 corridas nuevas que ningún modelo vio)

`ml/verificar_vivo.py` corrió 6 corridas nuevas (2 de carga normal con fallos en `ticket` y `eventos`, 2 de crecimiento progresivo y 2 de burst con fallos en `pago` y `notificaciones`), con la demo en vivo grabando, y las midió con el mismo protocolo. Se excluyó un tramo de 40 s en que la laptop se suspendió (5 episodios falsos). Estas corridas usan el **calendario fijo**, el que v8 conoce.

| Las 6 corridas juntas | Precisión | Recall | F1 | Falsas alarmas/h (servicio / sistema) | Graduales avisados (mediana) |
| --- | --- | --- | --- | --- | --- |
| **v9** | **0,78** | 0,24 | 0,36 | 4,8 / 29 | 4 de 4 (6 s) |
| v8 | 0,77 | 0,28 | **0,41** | **3,7 / 22** | 4 de 4 (5,5 s) |
| v7 | 0,66 | 0,37 | 0,47 | 8,9 / 54 | 4 de 4 (6,5 s) |
| CPU > 80 % | 0,38 | 0,10 | 0,15 | 4,8 / 29 | 1 de 4 (1 s) |

- Los tres modelos avisan todas las degradaciones graduales antes que la regla de CPU, con precisión ~0,8 en v8 y v9.
- En calendario fijo v8 queda algo por encima de v9, como era de esperar: es el patrón que aprendió. En calendario aleatorio (13.1) gana v9.
- En el crecimiento progresivo los modelos encendieron la alarma **34–38 s antes** de la saturación; la métrica estricta no lo cuenta porque empezó antes de la ventana de 30 s.

### 13.5 Cómo se llegó aquí

| Versión | Qué cambió | F1 | Falsas alarmas/h | Lección |
| --- | --- | --- | --- | --- |
| v1 | Primer modelo (28 corridas) | 0,40 | — | Su lead de 30 s era una ilusión: la alarma estaba casi siempre encendida |
| v2 | Lead honesto, umbral por F0,5, 40 corridas | 0,46 | 20,8 | Anticipación real de ~6,5 s |
| v3 / v4 | Contexto de dependencias + tendencias | 0,44 / 0,49 | 20,7 / 17,5 | Duplicó los avisos con ≥ 15 s (de 11 % a 25 %) |
| v5 | Etiqueta de cuello de botella propio | 0,33* | 5,0 | Muchos "cuellos de botella" eran servicios esperando a otro |
| v6 | 10 corridas contaminadas repetidas en limpio | 0,26* | 8,4 | v5 estaba inflado por la máquina lenta |
| v7 | 80 corridas (2 repeticiones) | 0,32* | 9,0 | Más datos: +0,06 de F1 |
| v8 | Ensamble de 3 semillas de la configuración de v7 | 0,36* | 2,3 | El barrido no mejoró nada; el ensamble redujo las falsas alarmas a la cuarta parte |
| **v9** | **+ 36 corridas de calendario aleatorio (116 en total)** | **0,31\*\*** | **5,2** | En corridas aleatorias, v8 solo anticipaba de verdad lo gradual; v9 es mejor a igual tasa de falsas alarmas (F1 0,19 contra 0,14) |

\* Desde v5 cambia la etiqueta, así que su F1 no se compara directamente con el de v1–v4. La configuración de v4 con 80 corridas y la etiqueta original llega a F1 0,50, pero esa etiqueta cuenta como cuello de botella la lentitud heredada.
\*\* Con otras corridas (incluye las aleatorias, más difíciles): no se compara con v5–v8. Ver 13.1.

---

## 14. Demo

`ml/demo.py` muestra el modelo final funcionando: una tabla que se actualiza cada segundo con la CPU, la p95 y la cola de cada servicio, la probabilidad de cuello de botella en los próximos 30 s, la alarma del modelo, la de la regla CPU > 80 % y el estado real. Cuando un servicio entra en cuello de botella, dice con cuántos segundos de anticipación avisó cada método (lead time honesto). Por defecto usa **v9**; en pantalla aparece su nombre.

### 14.1 Modo repetición (recomendado para presentar)

No necesita Docker y siempre muestra lo mismo. Reproduce una corrida grabada con las predicciones de la **validación cruzada**: cada corrida la predijo un modelo que no la vio al entrenar.

```powershell
python ml/demo.py --replay var_inj_leve_notificaciones_r1 --velocidad 2
```

En esa corrida `notificaciones` se degrada poco a poco: **v9 avisa 17 s antes** (v8: 18 s), la regla de CPU no lo detecta y no hay falsas alarmas en los otros 5 servicios. Más adelante llega una inyección abrupta (t = 220 s): ninguno de los dos la anticipa, lo que ilustra el límite del modelo. Para ver la misma corrida con v8: `--resultados resultados_v8`.

(La corrida que se usaba antes, `suave_inj_40_ticket_r2`, ya no sirve de ejemplo: en la validación cruzada de v9 le tocó otro fold y su modelo no la anticipa. Es un recordatorio de que un solo ejemplo no prueba nada; las cifras de la sección 13 sí.)

### 14.2 Modo en vivo

Mide el sistema real con el modelo final. Las entradas se arman con las mismas funciones del entrenamiento (verificado: diferencia máxima 2 × 10⁻⁷).

```powershell
docker compose restart                     # estado limpio de los servicios
python ml/verificar_vivo.py chequeo        # ~2 min: dice si la máquina está en condiciones de medir
python ml/demo.py                          # terminal 1: la tabla en vivo (v9)
# terminal 2, uno de estos:
python run_experiment.py --scenario normal --duration 360 --peak-users 100 --base-users 15 --run-id vivo_normal_mio --inject ticket:60:90:150:40 --inject ticket:220:60:150
python run_experiment.py --scenario progresivo --duration 360 --peak-users 60 --base-users 15 --run-id vivo_progresivo_mio
python run_experiment.py --scenario burst --duration 360 --peak-users 70 --base-users 15 --run-id vivo_burst_mio --inject pago:95:75:150:30 --inject pago:215:75:150
```

- **Haz el chequeo antes de probar.** Con la máquina ocupada (una llamada de Zoom, un juego, modo ahorro de energía) los servicios gastan el doble de CPU por petición: se saturan antes y en servicios que normalmente aguantan, y el modelo rinde mucho peor. Así salieron las 4 pruebas en vivo del 22-oct (2,1–2,5 × lo normal).
- Usa un `--run-id` nuevo en cada prueba; si repites uno, la corrida anterior se reemplaza.
- Espera 60 s **con tráfico** antes de predecir: el arranque de Locust confunde al modelo en `reservaciones`.
- En vivo el resultado varía de una corrida a otra, igual que en la evaluación. Para una presentación es más seguro el modo repetición.
- Las falsas alarmas que se ven en pantalla son las de los 6 servicios juntos: con v9, ~20–30 por hora (una cada 2–3 min). Con `--umbral 0.91` bajan a la mitad, a cambio de algo menos de recall.
- Para comparar con v8: `python ml/demo.py --modelo data/resultados_v8/modelo_ensamble.json`.

Opciones: `--modelo` (un `.pt` o un ensamble `.json`), `--umbral` (corte de la alarma), `--log archivo.csv` (guarda cada segundo), `--sin-pantalla` (texto en vez de tabla), `--duracion` (segundos).

### 14.3 Medir tus propias corridas en vivo

```powershell
Move-Item data\vivo_normal_mio data\vivo\vivo_normal_mio
New-Item data\vivo\vivo_normal_mio\OK
python ml/verificar_vivo.py evaluar        # v9 por defecto
```

El nombre debe empezar con `vivo_normal`, `vivo_progresivo` o `vivo_burst` (así se sabe el escenario). Se miden juntas todas las corridas `vivo_*` de `data/vivo/`.
