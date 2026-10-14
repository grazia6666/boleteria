import itertools
import os
import uuid

from pydantic import BaseModel

from app.common import call_service, create_app, simulate_work

app = create_app("Ticket Service")

NOTIFICACIONES_URL = os.getenv("NOTIFICACIONES_URL", "http://notificaciones:8000")


class TicketRequest(BaseModel):
    payment_id: int
    user_id: int
    event_id: int


_ids = itertools.count(1)
tickets = []


@app.get("/")
def root():
    return {"message": "Ticket Service funcionando"}


@app.post("/tickets")
def create_ticket(t: TicketRequest):
    simulate_work()
    ticket = {"id": next(_ids), "code": uuid.uuid4().hex[:10].upper(), **t.model_dump()}
    tickets.append(ticket)
    call_service("POST", NOTIFICACIONES_URL, "/notifications", {
        "user_id": t.user_id, "message": f"Tu ticket {ticket['code']} fue generado",
    })
    return ticket


@app.get("/tickets")
def list_tickets():
    simulate_work()
    return tickets[-100:]
