import itertools
import os

from fastapi import HTTPException
from pydantic import BaseModel

from app.common import call_service, create_app, simulate_work

app = create_app("Reservation Service")

USUARIOS_URL = os.getenv("USUARIOS_URL", "http://usuarios:8000")
EVENTOS_URL = os.getenv("EVENTOS_URL", "http://eventos:8000")
PAGO_URL = os.getenv("PAGO_URL", "http://pago:8000")


class Reservation(BaseModel):
    user_id: int
    event_id: int
    quantity: int = 1


_ids = itertools.count(1)
reservations = []


@app.get("/")
def root():
    return {"message": "Reservation Service funcionando"}


@app.post("/reservations")
def create_reservation(body: Reservation):
    """Flujo: usuarios - eventos - pago - (ticket - notificaciones)"""
    simulate_work()
    user = call_service("GET", USUARIOS_URL, f"/users/{body.user_id}")
    stock = call_service("POST", EVENTOS_URL, f"/events/{body.event_id}/reserve", {"quantity": body.quantity})

    rid = next(_ids)
    payment = call_service("POST", PAGO_URL, "/payments", {
        "reservation_id": rid,
        "user_id": body.user_id,
        "event_id": body.event_id,
        "amount": stock["price"] * body.quantity,
    })

    reservation = {"id": rid, "user": user["name"], "event_id": body.event_id,
                   "quantity": body.quantity, "payment": payment, "status": "confirmed"}
    reservations.append(reservation)
    return reservation


@app.get("/reservations")
def list_reservations():
    simulate_work()
    return reservations[-100:]


@app.get("/reservations/{rid}")
def get_reservation(rid: int):
    simulate_work()
    for r in reservations:
        if r["id"] == rid:
            return r
    raise HTTPException(status_code=404, detail="Reservación no encontrada")
