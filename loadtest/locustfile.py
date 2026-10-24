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
# Momentos de inicio de cada rafaga, en segundos, separados por coma (ej. "70,155,290"). Si se da, reemplaza
# al patron periodico BURST_START/BURST_EVERY: asi los picos no caen siempre en el mismo ritmo y el modelo
# no puede aprender el calendario del experimento.
BURST_TIMES = [float(x) for x in os.getenv("BURST_TIMES", "").split(",") if x.strip()]
# Segundos que tarda cada rafaga en subir de BASE_USERS a PEAK_USERS (0 = de golpe). Con rampa, la rafaga
# dura BURST_RAMP + BURST_LEN: sube en linea recta, se mantiene en el pico BURST_LEN s y vuelve a la base.
BURST_RAMP = float(os.getenv("BURST_RAMP", "0"))


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
            if BURST_TIMES:
                for s in BURST_TIMES:
                    if s <= t < s + BURST_RAMP + BURST_LEN:
                        frac = min(1.0, (t - s) / BURST_RAMP) if BURST_RAMP > 0 else 1.0
                        return (int(round(BASE_USERS + (PEAK_USERS - BASE_USERS) * frac)), 50)
                return (BASE_USERS, 50)
            in_burst = t >= BURST_START and ((t - BURST_START) % BURST_EVERY) < BURST_LEN
            return (PEAK_USERS if in_burst else BASE_USERS, 50)
        raise ValueError(f"SCENARIO desconocido: {SCENARIO}")