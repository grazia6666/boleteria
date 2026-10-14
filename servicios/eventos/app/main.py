import itertools
import threading

from fastapi import HTTPException
from pydantic import BaseModel

from app.common import create_app, simulate_work

app = create_app("Event Service")
lock = threading.Lock()


class Event(BaseModel):
    name: str
    venue: str
    price: float
    capacity: int


class Reserve(BaseModel):
    quantity: int = 1


_ids = itertools.count(4)
# Capacidad grande para que no se agoten durante las pruebas de carga
events = {
    1: {"id": 1, "name": "BTS ARIRANG TOUR", "venue": "Estadio San Marcos", "price": 50.0, "capacity": 10_000_000, "available": 10_000_000},
    2: {"id": 2, "name": "HABLANDO HUEVADAS", "venue": "Teatro Central", "price": 30.0, "capacity": 10_000_000, "available": 10_000_000},
    3: {"id": 3, "name": "Festival Jazz", "venue": "Parque Sur", "price": 40.0, "capacity": 10_000_000, "available": 10_000_000},
}


@app.get("/")
def root():
    return {"message": "Event Service funcionando"}


@app.get("/events")
def list_events():
    simulate_work()
    return list(events.values())


@app.post("/events")
def create_event(ev: Event):
    simulate_work()
    eid = next(_ids)
    events[eid] = {"id": eid, **ev.model_dump(), "available": ev.capacity}
    return events[eid]


@app.get("/events/{event_id}")
def get_event(event_id: int):
    simulate_work()
    if event_id not in events:
        raise HTTPException(status_code=404, detail="Evento no encontrado")
    return events[event_id]


@app.post("/events/{event_id}/reserve")
def reserve(event_id: int, body: Reserve):
    simulate_work()
    with lock:
        ev = events.get(event_id)
        if not ev:
            raise HTTPException(status_code=404, detail="Evento no encontrado")
        if ev["available"] < body.quantity:
            raise HTTPException(status_code=409, detail="Sin disponibilidad")
        ev["available"] -= body.quantity
        return {"event_id": event_id, "reserved": body.quantity, "available": ev["available"], "price": ev["price"]}
