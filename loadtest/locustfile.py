"""
Generador de carga. El escenario se elige con la variable SCENARIO:
  normal      -> usuarios constantes
  progresivo  -> sube de 5 a PEAK_USERS (65% del tiempo) y vuelve a bajar
  burst       -> carga base con picos subitos de PEAK_USERS
"""
import os
import random

from locust import HttpUser, LoadTestShape, between, task

SCENARIO = os.getenv("SCENARIO", "normal")
DURATION = int(os.getenv("DURATION", "600"))
BASE_USERS = int(os.getenv("BASE_USERS", "15"))
PEAK_USERS = int(os.getenv("PEAK_USERS", "100"))
BURST_START = int(os.getenv("BURST_START", "60"))
BURST_EVERY = int(os.getenv("BURST_EVERY", "120"))
BURST_LEN = int(os.getenv("BURST_LEN", "25"))


class ReservaUser(HttpUser):
    wait_time = between(0.3, 1.0)

    @task
    def reservar(self):
        self.client.post(
            "/reservations",
            json={"user_id": 1, "event_id": random.randint(1, 3), "quantity": 1},
            name="/reservations",
        )


class Shape(LoadTestShape):
    def tick(self):
        t = self.get_run_time()
        if t >= DURATION:
            return None
        if SCENARIO == "normal":
            return (BASE_USERS, 10)
        if SCENARIO == "progresivo":
            # sube de 5 a PEAK_USERS en el 65% inicial y baja de vuelta (da ejemplos de recuperacion)
            up = DURATION * 0.65
            if t <= up:
                users = 5 + (PEAK_USERS - 5) * t / up
            else:
                users = PEAK_USERS - (PEAK_USERS - 5) * (t - up) / (DURATION - up)
            return (max(int(users), 1), 5)
        if SCENARIO == "burst":
            in_burst = t >= BURST_START and ((t - BURST_START) % BURST_EVERY) < BURST_LEN
            return (PEAK_USERS if in_burst else BASE_USERS, 50)
        raise ValueError(f"SCENARIO desconocido: {SCENARIO}")