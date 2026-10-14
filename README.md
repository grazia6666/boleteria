# Boletería en la Nube: guía de los microservicios

Este documento explica cómo funciona cada uno de los 6 microservicios del proyecto *Microservice Bottleneck Prediction*, cómo se conectan entre sí y qué papel cumplen en el experimento.

---

## 1. Visión general

La aplicación simula una **boletería**: un usuario reserva boletos para un evento, se cobra el pago, se genera el ticket y se envía una notificación.

Cada paso es un microservicio independiente, con su propio contenedor Docker, y se comunican entre sí por HTTP.

### Cadena de dependencias

```
                         ┌──────────┐
                    ┌───►│ usuarios │   (hoja)
                    │    └──────────┘
┌───────────────┐   │    ┌──────────┐
│ reservaciones │───┼───►│ eventos  │   (hoja)
└───────────────┘   │    └──────────┘
                    │    ┌──────────┐    ┌──────────┐    ┌────────────────┐
                    └───►│   pago   │───►│  ticket  │───►│ notificaciones │
                         └──────────┘    └──────────┘    └────────────────┘
```

`reservaciones` es el **orquestador**: recibe la petición del cliente y llama a los demás. Las llamadas son **síncronas**: cada servicio espera la respuesta del siguiente antes de contestar.

### Puertos

| Servicio | Puerto host | Puerto interno | Llama a |
|---|---|---|---|
| usuarios | 8001 | 8000 | ninguno |
| eventos | 8002 | 8000 | ninguno |
| reservaciones | 8003 | 8000 | usuarios, eventos, pago |
| pago | 8004 | 8000 | ticket |
| ticket | 8005 | 8000 | notificaciones |
| notificaciones | 8006 | 8000 | ninguno |

Desde tu máquina usas el puerto host (`localhost:8003`). Entre contenedores se usa el nombre del servicio y el puerto interno (`http://pago:8000`).

### Estructura de cada servicio

```
servicios/<nombre>/
├── Dockerfile
├── requirements.txt
└── app/
    ├── __init__.py
    ├── common.py     # idéntico en los 6 servicios
    └── main.py       # lógica propia de cada servicio
```

---

## 2. `common.py`: lo que comparten todos

Es el archivo con la parte de **medición y experimentación**. Separa la lógica de negocio (`main.py`) de lo que se necesita para detectar cuellos de botella.

| Elemento | Qué hace |
|---|---|
| `config` | Guarda `base_delay_ms` y `cpu_load`, leídos de variables de entorno al arrancar |
| `simulate_work()` | Espera unos ms y ejecuta un bucle que consume CPU real. Fabrica el cuello de botella |
| `call_service()` | Llama a otro microservicio por HTTP. Si falla devuelve un error 502 claro |
| `create_app()` | Crea la app FastAPI con middleware de métricas y endpoints de utilidad |

### Métricas que se registran (Prometheus)

| Métrica | Tipo | Corresponde a |
|---|---|---|
| `http_requests_total` | Counter | requests/s (al derivarla en el tiempo) |
| `http_request_duration_seconds` | Histogram | latencia |
| `http_requests_in_flight` | Gauge | longitud de cola (requests en curso) |

El middleware mide automáticamente cada request, excepto `/health` y `/metrics`. La etiqueta `path` usa la plantilla de la ruta (`/users/{user_id}`) y no el valor real, para evitar miles de series distintas.

### Endpoints que reciben todos los servicios

| Endpoint | Función |
|---|---|
| `GET /health` | Confirma que el servicio responde |
| `GET /metrics` | Expone las métricas en formato Prometheus |
| `GET /admin/load` | Muestra la carga artificial actual |
| `POST /admin/load` | Cambia `cpu_load` y `base_delay_ms` en caliente, sin reiniciar |

### Cómo se inyecta un cuello de botella

```bash
# Saturar el servicio de pago
curl -X POST http://localhost:8004/admin/load -H "Content-Type: application/json" -d '{"cpu_load":200}'

# Quitar la carga
curl -X POST http://localhost:8004/admin/load -H "Content-Type: application/json" -d '{"cpu_load":0}'
```

El momento exacto en que ejecutas estos comandos es lo que después sirve para **etiquetar** el dataset.

---

## 3. Servicio `usuarios` (puerto 8001)

**Función:** guardar usuarios y confirmar si un usuario existe.

**Datos:** una lista en memoria. Arranca con un usuario demo (`id: 1`).

**Modelo:**
```json
{"name": "Juan", "email": "juan@mail.com", "password": "1234"}
```

| Endpoint | Descripción |
|---|---|
| `GET /` | Mensaje de servicio activo |
| `POST /users` | Crea un usuario |
| `GET /users` | Lista usuarios |
| `GET /users/{id}` | Obtiene uno, o 404 |
| `DELETE /users/{id}` | Elimina uno, o 404 |

**Detalles:**
- Los ids los genera `itertools.count(2)`, seguro con concurrencia.
- La función `public()` quita la contraseña de todas las respuestas.
- Todos los endpoints llaman a `simulate_work()`.

**Papel en la cadena:** `reservaciones` consulta `GET /users/{id}` para validar al usuario. No llama a nadie, es una **hoja** del grafo.

```bash
curl http://localhost:8001/users
curl -X POST http://localhost:8001/users -H "Content-Type: application/json" -d '{"name":"Juan Perez","email":"juan@mail.com","password":"1234"}'
curl http://localhost:8001/users/1
```

---

## 4. Servicio `eventos` (puerto 8002)

**Función:** mantener el catálogo de eventos y **controlar los boletos disponibles**.

**Datos:** un diccionario `{id: evento}` en memoria, con 3 eventos precargados. Cada uno tiene `capacity` y `available`, ambos en **10 millones** para que no se agoten durante las pruebas de carga y no aparezcan errores 409 que ensucien los datos.

**Modelos:**
```json
{"name": "Stand Up", "venue": "Teatro Mayor", "price": 25, "capacity": 100}
{"quantity": 3}
```

| Endpoint | Descripción |
|---|---|
| `GET /events` | Lista eventos |
| `POST /events` | Crea un evento |
| `GET /events/{id}` | Obtiene uno, o 404 |
| `POST /events/{id}/reserve` | **Descuenta boletos**, devuelve cantidad, disponibles y precio |

### El endpoint `reserve`

```python
simulate_work()
with lock:
    # validar existencia (404) y disponibilidad (409)
    ev["available"] -= body.quantity
```

- Usa un **lock** porque FastAPI ejecuta los endpoints síncronos en varios hilos a la vez. Sin él, dos requests podrían leer "queda 1 boleto" simultáneamente y venderlo dos veces (condición de carrera).
- `simulate_work()` está **fuera** del lock a propósito. Si estuviera dentro, los requests harían fila por el lock y el servicio se degradaría por eso y no por la carga que tú controlas.
- Devuelve `price` para que `reservaciones` calcule el monto sin una segunda llamada.

**Papel en la cadena:** `reservaciones` llama a `POST /events/{id}/reserve` justo después de validar al usuario. Es otra **hoja**.

```bash
curl http://localhost:8002/events
curl -X POST http://localhost:8002/events/1/reserve -H "Content-Type: application/json" -d '{"quantity":3}'
```

---

## 5. Servicio `reservaciones` (puerto 8003)

**Función:** es el **orquestador**. Recibe la solicitud del cliente y coordina a los demás servicios para completarla.

**Variables de entorno:** `USUARIOS_URL`, `EVENTOS_URL`, `PAGO_URL` (definidas en `docker-compose.yml`).

**Modelo:**
```json
{"user_id": 1, "event_id": 1, "quantity": 2}
```

| Endpoint | Descripción |
|---|---|
| `POST /reservations` | Crea una reservación (recorre toda la cadena) |
| `GET /reservations` | Últimas 100 reservaciones |
| `GET /reservations/{id}` | Obtiene una, o 404 |

### Flujo de `POST /reservations`

```
1. simulate_work()
2. GET  usuarios/users/{user_id}        → valida que el usuario existe
3. POST eventos/events/{id}/reserve     → descuenta boletos, obtiene el precio
4. POST pago/payments                   → cobra (precio × cantidad)
        └─► pago llama a ticket
              └─► ticket llama a notificaciones
5. Guarda la reservación con estado "confirmed" y la devuelve
```

La respuesta incluye el pago, y dentro del pago el ticket. Eso confirma que la cadena completa se ejecutó.

**Comportamiento ante fallos:**
- Si `usuarios` responde 404, la reservación falla ahí y no se toca eventos.
- Si `eventos` responde 409 (sin disponibilidad), no se llega a pago.
- Si un servicio no responde, `call_service()` devuelve 502.
- No hay *rollback*: si `pago` falla después de descontar boletos, esos boletos quedan descontados. Es una simplificación aceptable para el experimento.

**Papel en el experimento:** como espera la respuesta de todos los demás, **su latencia es la suma de la de toda la cadena**. Es el servicio donde mejor se ve la propagación de un cuello de botella hacia atrás.

```bash
curl -X POST http://localhost:8003/reservations -H "Content-Type: application/json" -d '{"user_id":1,"event_id":1,"quantity":2}'
```

---

## 6. Servicio `pago` (puerto 8004)

**Función:** registrar el cobro de una reservación.

**Variable de entorno:** `TICKET_URL`.

**Modelo:**
```json
{"reservation_id": 1, "user_id": 1, "event_id": 1, "amount": 100.0}
```

| Endpoint | Descripción |
|---|---|
| `POST /payments` | Crea un pago y dispara la generación del ticket |
| `GET /payments` | Últimos 100 pagos |

### Flujo de `POST /payments`

```
1. simulate_work()
2. Crea el pago con estado "approved" y lo guarda
3. POST ticket/tickets
4. Devuelve el pago con el ticket anidado
```

**Detalles:**
- Todos los pagos se aprueban. No hay pasarela real, el objetivo es la carga, no la lógica financiera.
- El pago se guarda **antes** de llamar a ticket.

**Papel en la cadena:** está en el **medio**. Recibe de `reservaciones` y delega en `ticket`. Un cuello de botella aquí se propaga hacia atrás (a `reservaciones`) y hacia adelante la cadena queda sin tráfico.

```bash
curl -X POST http://localhost:8004/payments -H "Content-Type: application/json" -d '{"reservation_id":1,"user_id":1,"event_id":1,"amount":50}'
```

---

## 7. Servicio `ticket` (puerto 8005)

**Función:** generar el boleto con un código único.

**Variable de entorno:** `NOTIFICACIONES_URL`.

**Modelo:**
```json
{"payment_id": 1, "user_id": 1, "event_id": 1}
```

| Endpoint | Descripción |
|---|---|
| `POST /tickets` | Crea un ticket y dispara la notificación |
| `GET /tickets` | Últimos 100 tickets |

### Flujo de `POST /tickets`

```
1. simulate_work()
2. Genera un código aleatorio (10 caracteres en mayúsculas, vía uuid)
3. Guarda el ticket
4. POST notificaciones/notifications  ("Tu ticket XXXX fue generado")
5. Devuelve el ticket
```

**Papel en la cadena:** está en el **medio**, entre `pago` y `notificaciones`.

```bash
curl -X POST http://localhost:8005/tickets -H "Content-Type: application/json" -d '{"payment_id":1,"user_id":1,"event_id":1}'
```

---

## 8. Servicio `notificaciones` (puerto 8006)

**Función:** simular el envío de mensajes al usuario.

**Modelo:**
```json
{"user_id": 1, "message": "Tu ticket fue generado"}
```

| Endpoint | Descripción |
|---|---|
| `POST /notifications` | Registra una notificación como enviada |
| `GET /notifications` | Últimas 100 notificaciones |

**Detalles:** no envía nada real, solo guarda el mensaje en una lista con `sent: true`.

**Papel en la cadena:** es el **final**. No llama a nadie. Como está al fondo, su degradación se propaga hacia atrás por toda la cadena: `ticket` → `pago` → `reservaciones`.

```bash
curl http://localhost:8006/notifications
```

---

## 9. Cómo se propaga un cuello de botella

Como las llamadas son síncronas, si un servicio se vuelve lento, **todos los que lo llaman esperan** y también se degradan.

| Servicio saturado | Servicios que se ven afectados |
|---|---|
| `notificaciones` | ticket, pago, reservaciones |
| `ticket` | pago, reservaciones |
| `pago` | reservaciones |
| `usuarios` o `eventos` | reservaciones (y la cadena de pago hacia adelante queda sin tráfico) |
| `reservaciones` | solo el cliente |

Esto es lo que hace interesante el problema: el síntoma (latencia alta en `reservaciones`) aparece lejos de la causa raíz (por ejemplo, `notificaciones`). Una regla simple como "CPU > 80%" mira un servicio a la vez y no entiende esa relación, y es la motivación para pasar a una Temporal GNN en el segundo parcial.

---

## 10. Limitaciones conocidas

- **Todo el estado vive en memoria.** Al reiniciar un contenedor se pierde, lo cual conviene: cada corrida experimental parte de cero.
- **Un solo worker de uvicorn por servicio.** Con varios, cada worker tendría su propia lista.
- **No hay rollback** entre servicios si algo falla a mitad de la cadena.
- **`common.py` está copiado 6 veces.** Si lo modificas, debes copiar el cambio a las otras 5 carpetas.
- **Las contraseñas se guardan en texto plano** (solo en memoria, y no se devuelven en las respuestas). Aceptable para este proyecto.

---

## 11. Probar todo en orden

```bash
# 1. Levantar
docker compose up --build

# 2. Ver que estén vivos
docker compose ps

# 3. Flujo completo
curl -X POST http://localhost:8003/reservations -H "Content-Type: application/json" -d '{"user_id":1,"event_id":1,"quantity":2}'

# 4. Confirmar que llegó la notificación al final de la cadena
curl http://localhost:8006/notifications

# 5. Ver métricas
curl http://localhost:8003/metrics

# 6. Apagar
docker compose down
```
