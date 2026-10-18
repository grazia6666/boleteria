"""Utilidades compartidas: métricas Prometheus, carga simulada y llamadas entre servicios."""
import os
import time
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel

SERVICE_NAME = os.getenv("SERVICE_NAME", "service")

# Configuración mutable en caliente (sirve para inyectar cuellos de botella)
config = {
    "base_delay_ms": int(os.getenv("BASE_DELAY_MS", "5")),
    "cpu_load": int(os.getenv("CPU_LOAD", "0")),
}

BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 0.8, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 30)

# --- métricas del propio servicio ---
REQUESTS = Counter("http_requests_total", "Total de requests", ["service", "method", "path", "status"])
LATENCY = Histogram("http_request_duration_seconds", "Latencia por request", ["service", "path"], buckets=BUCKETS)
IN_FLIGHT = Gauge("http_requests_in_flight", "Requests en curso (proxy de longitud de cola)", ["service"])

# --- métricas de las dependencias (aristas origen -> destino) ---
DOWN_LAT = Histogram("downstream_request_duration_seconds", "Latencia de llamadas a otros servicios",
                     ["service", "target"], buckets=BUCKETS)
DOWN_REQ = Counter("downstream_requests_total", "Llamadas a otros servicios",
                   ["service", "target", "status"])


def simulate_work():
    """Latencia base + carga de CPU real (genera saturación)."""
    if config["base_delay_ms"]:
        time.sleep(config["base_delay_ms"] / 1000)
    n = config["cpu_load"]
    if n:
        x = 0
        for i in range(n * 10000):
            x += i * i


def call_service(method: str, base_url: str, path: str, json=None):
    """Llama a otro microservicio, mide la arista y propaga errores como 502."""
    target = urlparse(base_url).hostname or base_url
    start = time.perf_counter()
    try:
        r = httpx.request(method, f"{base_url}{path}", json=json, timeout=10.0)
    except httpx.HTTPError as e:
        DOWN_LAT.labels(SERVICE_NAME, target).observe(time.perf_counter() - start)
        DOWN_REQ.labels(SERVICE_NAME, target, "error").inc()
        raise HTTPException(status_code=502, detail=f"Error llamando {base_url}{path}: {e}")
    DOWN_LAT.labels(SERVICE_NAME, target).observe(time.perf_counter() - start)
    DOWN_REQ.labels(SERVICE_NAME, target, str(r.status_code)).inc()
    if r.status_code >= 400:
        raise HTTPException(status_code=r.status_code, detail=r.text)
    return r.json()


class LoadConfig(BaseModel):
    base_delay_ms: int | None = None
    cpu_load: int | None = None


def create_app(title: str) -> FastAPI:
    app = FastAPI(title=title)

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        if request.url.path in ("/metrics", "/health"):
            return await call_next(request)
        IN_FLIGHT.labels(SERVICE_NAME).inc()
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            elapsed = time.perf_counter() - start
            route = request.scope.get("route")
            path = route.path if route else request.url.path
            LATENCY.labels(SERVICE_NAME, path).observe(elapsed)
            REQUESTS.labels(SERVICE_NAME, request.method, path, str(status)).inc()
            IN_FLIGHT.labels(SERVICE_NAME).dec()

    @app.get("/health")
    def health():
        return {"service": SERVICE_NAME, "status": "ok"}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/admin/load")
    def get_load():
        return config

    @app.post("/admin/load")
    def set_load(cfg: LoadConfig):
        """Cambia la carga en caliente: POST /admin/load {"cpu_load": 50}"""
        if cfg.base_delay_ms is not None:
            config["base_delay_ms"] = cfg.base_delay_ms
        if cfg.cpu_load is not None:
            config["cpu_load"] = cfg.cpu_load
        return config

    return app