import itertools

from pydantic import BaseModel

from app.common import create_app, simulate_work

app = create_app("Notification Service")


class Notification(BaseModel):
    user_id: int
    message: str


_ids = itertools.count(1)
notifications = []


@app.get("/")
def root():
    return {"message": "Notification Service funcionando"}


@app.post("/notifications")
def send_notification(n: Notification):
    simulate_work()
    item = {"id": next(_ids), **n.model_dump(), "sent": True}
    notifications.append(item)
    return item


@app.get("/notifications")
def list_notifications():
    simulate_work()
    return notifications[-100:]
