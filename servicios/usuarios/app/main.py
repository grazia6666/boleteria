import itertools

from fastapi import HTTPException
from pydantic import BaseModel

from app.common import create_app, simulate_work

app = create_app("User Service")


class User(BaseModel):
    name: str
    email: str
    password: str


_ids = itertools.count(2)
users = [{"id": 1, "name": "Usuario Demo", "email": "demo@mail.com", "password": "demo"}]


def public(u):
    return {k: v for k, v in u.items() if k != "password"}


@app.get("/")
def root():
    return {"message": "User Service funcionando"}


@app.post("/users")
def create_user(user: User):
    simulate_work()
    new_user = {"id": next(_ids), **user.model_dump()}
    users.append(new_user)
    return public(new_user)


@app.get("/users")
def get_users():
    simulate_work()
    return [public(u) for u in users]


@app.get("/users/{user_id}")
def get_user(user_id: int):
    simulate_work()
    for u in users:
        if u["id"] == user_id:
            return public(u)
    raise HTTPException(status_code=404, detail="Usuario no encontrado")


@app.delete("/users/{user_id}")
def delete_user(user_id: int):
    simulate_work()
    for u in users:
        if u["id"] == user_id:
            users.remove(u)
            return {"message": "Usuario eliminado"}
    raise HTTPException(status_code=404, detail="Usuario no encontrado")
