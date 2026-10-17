Primero compruebo con los CSV de lead time una sospecha que veo en los datos: muchos leads valen exactamente 30 s.

## Qué dicen los resultados (28 corridas, CV agrupada por corrida)

**Resumen por fold, vista B (media sobre 4 folds):**

| Método | Precision | Recall | F1 |
|---|---|---|---|
| LSTM | ≈0.29 | ≈0.71 | ≈0.40 (rango 0.30–0.46) |
| Threshold CPU>80% | ≈0.43 | ≈0.02 | ≈0.04 |

**1. El LSTM le gana al umbral en F1, pero la ganancia viene de recall y no de precisión.**
- El umbral casi nunca dispara: el recall es 0.6–4 % y solo detectó 11 de 148 episodios.
- Cuando dispara acierta bastante (precision 0.4–0.7), pero llega tarde: en los gradual detecta 7 de 25 con mediana de 1 s.
- El LSTM tiene un 73 % de falsas alarmas. Eso es mucho para un sistema real.
- El umbral de decisión que eligió la validación es bajo (0.2–0.35). Eso explica el recall alto y la precisión baja.

**2. El umbral solo gana en `progresivo_base`.** F1 0.63 contra 0.07 del LSTM.
- Ahí el cuello de botella es saturación real de CPU en `reservaciones`, justo el caso para el que sirve la regla CPU>80%.
- El LSTM tiene recall 1.0 y precisión 0.037, es decir, alarma casi siempre.
- Solo hay 2 corridas y 2 episodios, así que no es concluyente. Pero hay que reportarlo: no conviene presentar "el LSTM gana en todo".
- En `normal_base` el LSTM también genera falsas alarmas (precisión 0.0, sin positivos reales).

**3. El Lead Time de 30 s es engañoso. No lo pongas tal cual en el informe.**
- En propagación, 66 de 72 detecciones tienen la alarma encendida durante toda la ventana. En natural son 22 de 22, y en abrupto 11 de 11.
- Eso es lo que sale de una alarma casi permanente en `pago`, `reservaciones` y `ticket`, no de anticipación real. Con precisión 0.27 tiene sentido.
- Lo que sí parece anticipación real es la inyección gradual en los servicios "limpios". Ahí la mediana es 2 s en `eventos`, 3 s en `usuarios`, 8.5 s en `notificaciones` y 11 s en `ticket`.
- Es decir: aprende a reaccionar cuando la CPU y la latencia ya están subiendo, con pocos segundos de ventaja. Es mucho menos que los 30 s que pedía el PDF.

**4. Lo abrupto no se anticipa, como esperábamos.**
- Detecta 0/4 en `eventos`, `notificaciones` y `usuarios`.
- Solo "detecta" en `pago`, `reservaciones` y `ticket`, y ahí la causa es la alarma ya encendida, no la anticipación.

**5. La persistencia queda en 0 en la vista B por construcción.** Esa vista solo evalúa filas sin cuello de botella ahora, y la persistencia solo alarma si ya hay uno. No es un fallo ni una comparación útil. Úsala solo en la vista A como referencia de que el modelo no solo copia la etiqueta actual.

**Sobre el timeline** (`burst_inj_eventos_r1`, `eventos`): la probabilidad se mantiene en ~0.12 y salta a 0.96 un segundo antes del inicio (línea roja). Se ve que reacciona a la subida de CPU (95 s en adelante) y no la predice con 30 s de adelanto. Es un buen ejemplo para el informe, porque muestra lo que realmente hace el modelo.

## Cómo plantearlo en la discusión (honesto y defendible)
- El LSTM supera al umbral CPU>80% en recall y F1 en 4 de 5 escenarios, sobre todo en propagación entre servicios, donde la CPU del servicio que se degrada no sube.
- Su ventaja de anticipación real es de pocos segundos (≈2–11 s), no de 30.
- Tiene muchas falsas alarmas y pierde en el escenario de saturación natural.
- Los datos tienen limitaciones: todas las inyecciones saturan, el conjunto de test es pequeño, todo corre en una sola máquina y la etiqueta sale de la latencia, que también es una entrada.

## Lo que haría antes de relanzar la campaña
Cambios para la corrida ampliada y el reentrenamiento:
1. **Reducir falsas alarmas.** Elegir el umbral maximizando F1 con una restricción de fpr, o usar F-beta con β≈0.5. Reportar también la tasa de falsas alarmas por hora, y no solo el lead time.
2. **Lead time honesto.** Contar solo detecciones cuya alarma *empieza* dentro de la ventana, tras un periodo sin alarma. Así una alarma permanente no suma lead.
3. **Inyecciones suaves** (`cpu_load` ≈25–40) para tener episodios que crecen despacio y que de verdad se puedan anticipar.
4. **Más repeticiones** (`--reps 2`) y collector V2, para que el test no dependa de 4 episodios por tipo.
5. **Quitar `lat_*` de las entradas** como experimento, para ver cuánto del resultado viene de leer la latencia y cuánto de las señales anticipatorias (`cpu`, `cola`, `rps`).

Si quieres, te preparo ahora los cambios 1, 2 y 5 en `datos.py` y `entrenar.py`, y el grupo de inyecciones suaves en `run_campaign.py`, para dejar todo listo antes de relanzar la campaña.
