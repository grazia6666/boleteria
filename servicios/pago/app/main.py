import itertools
import os

from pydantic import BaseModel

from app.common import call_service, create_app, simulate_work

app = create_app("Payment Service")

TICKET_URL = os.getenv("TICKET_URL", "http://ticket:8000")


class Payment(BaseModel):
    reservation_id: int
    user_id: int
    event_id: int
    amount: float


_ids = itertools.count(1)
payments = []


@app.get("/")
def root():
    return {"message": "Payment Service funcionando"}


@app.post("/payments")
def create_payment(p: Payment):
    simulate_work()
    payment = {"id": next(_ids), **p.model_dump(), "status": "approved"}
    payments.append(payment)
    ticket = call_service("POST", TICKET_URL, "/tickets", {
        "payment_id": payment["id"], "user_id": p.user_id, "event_id": p.event_id,
    })
    return {**payment, "ticket": ticket}


@app.get("/payments")
def list_payments():
    simulate_work()
    return payments[-100:]
