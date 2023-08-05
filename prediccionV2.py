import os
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

def is_finite(value):
    return np.isfinite(value) and not pd.isna(value) and not pd.isnull(value)

# Función para calcular la ganancia potencial
def calcular_ganancia_potencial(jugador):
    prediccion_7_dias = jugador.get("prediccion_7_dias")
    if prediccion_7_dias is None:
        return 0.0
    return prediccion_7_dias - jugador["precio_actual"]

ruta_players = "players"

# Crear una lista para almacenar los datos de los jugadores
lista_jugadores = []

# Recorrer los archivos de los jugadores
for equipo in os.listdir(ruta_players):
    ruta_equipo = os.path.join(ruta_players, equipo)
    if os.path.isdir(ruta_equipo):
        for jugador_json in os.listdir(ruta_equipo):
            ruta_jugador = os.path.join(ruta_equipo, jugador_json)
            with open(ruta_jugador, "r") as archivo_json:
                datos_jugador = json.load(archivo_json)
                id_jugador = datos_jugador["id"]
                nombre_jugador = datos_jugador["name"]
                market_values = datos_jugador["marketValue"]

                fechas = sorted([datetime.strptime(fecha, "%d/%m/%Y") for fecha in market_values.keys()])
                precios = np.array([market_values[fecha.strftime("%d/%m/%Y")] for fecha in fechas], dtype=float)

                # Obtener todos los precios pasados de hace 15 días
                precios_pasados_15_dias = precios[-15:]

                # Obtener el precio actual
                precio_actual = precios[-1]

                # Añadir los datos del jugador a la lista
                jugador_info = {
                    "id": id_jugador,
                    "nombre": nombre_jugador,
                    "nickname": datos_jugador["nickname"] if "nickname" in datos_jugador else "",
                    "precio_actual": precio_actual,
                    "precios_pasados_15_dias": precios_pasados_15_dias.tolist(),  # Convertir a lista para JSON
                    "marketValue": json.dumps(market_values)
                }

                lista_jugadores.append(jugador_info)

# Convertir la lista de jugadores en un DataFrame
data = pd.DataFrame(lista_jugadores)

# Pad the sequences to make them have the same length
max_sequence_length = 15
data["precios_pasados_15_dias"] = list(pad_sequences(data["precios_pasados_15_dias"], maxlen=max_sequence_length, padding='post'))

# Convertir los precios pasados en una matriz de características
X = np.array(data["precios_pasados_15_dias"].tolist())

# Obtener el precio actual como vector de etiquetas
y = data["precio_actual"].values

# División de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalización de datos
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Verificar las dimensiones de los conjuntos de entrenamiento y prueba
print("Dimensiones de X_train_scaled:", X_train_scaled.shape)
print("Dimensiones de X_test_scaled:", X_test_scaled.shape)

# Verificar el número de características en los datos de entrenamiento y prueba
print("Número de características en X_train_scaled:", X_train_scaled.shape[1])
print("Número de características en X_test_scaled:", X_test_scaled.shape[1])

# Definición del modelo
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])

# Compilación del modelo
model.compile(optimizer="adam", loss="mean_squared_error")

# Entrenamiento del modelo
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32)

# Evaluación del modelo
loss = model.evaluate(X_test_scaled, y_test)
print("Mean Squared Error:", loss)

# Actualizar la lista de jugadores con las predicciones
# Actualizar la lista de jugadores con las predicciones
for i, jugador in enumerate(lista_jugadores):
    # Validamos que tenga al menos 15 precios pasados
    if len(jugador["precios_pasados_15_dias"]) < 15:
        lista_jugadores[i]["prediccion_7_dias"] = None
        continue
    precios_pasados_15_dias = np.array(jugador["precios_pasados_15_dias"])

    # Use the last 15 prices to create a sequence of length 15 for prediction
    nuevos_datos = precios_pasados_15_dias.reshape(1, -1)
    nuevos_datos_scaled = scaler.transform(nuevos_datos)
    prediccion = model.predict(nuevos_datos_scaled)[0][0]
    lista_jugadores[i]["prediccion_7_dias"] = float(prediccion)  # Convertir a float antes de guardar

    # Calcular los campos adicionales
    jugador_info = lista_jugadores[i]
    precio_hace_15_dias = jugador_info["precios_pasados_15_dias"][0]
    precio_hace_7_dias = jugador_info["precios_pasados_15_dias"][8]
    precio_hace_5_dias = jugador_info["precios_pasados_15_dias"][10]
    porcentaje_aumento_15_dias = ((jugador_info["precio_actual"] - precio_hace_15_dias) / precio_hace_15_dias) * 100
    porcentaje_aumento_7_dias = ((jugador_info["precio_actual"] - precio_hace_7_dias) / precio_hace_7_dias) * 100
    porcentaje_aumento_5_dias = ((jugador_info["precio_actual"] - precio_hace_5_dias) / precio_hace_5_dias) * 100
    ganancia_potencial = jugador_info["prediccion_7_dias"] - jugador_info["precio_actual"]
    porcentaje_ganancia_potencial = (ganancia_potencial / jugador_info["precio_actual"]) * 100

    # Añadir los campos adicionales al diccionario del jugador
    jugador_info["precio_hace_15_dias"] = float(precio_hace_15_dias)
    jugador_info["precio_hace_7_dias"] = float(precio_hace_7_dias)
    jugador_info["precio_hace_5_dias"] = float(precio_hace_5_dias)
    jugador_info["porcentaje_aumento_15_dias"] = float(porcentaje_aumento_15_dias) if is_finite(porcentaje_aumento_15_dias) else None
    jugador_info["porcentaje_aumento_7_dias"] = float(porcentaje_aumento_7_dias) if is_finite(porcentaje_aumento_7_dias) else None
    jugador_info["porcentaje_aumento_5_dias"] = float(porcentaje_aumento_5_dias) if is_finite(porcentaje_aumento_5_dias) else None
    jugador_info["ganancia_potencial"] = float(ganancia_potencial) if is_finite(ganancia_potencial) else None
    jugador_info["porcentaje_ganancia_potencial"] = float(porcentaje_ganancia_potencial) if is_finite(porcentaje_ganancia_potencial) else None

# Ordenar la lista de jugadores por la ganancia potencial (de mayor a menor)
jugadores_aumento = sorted(lista_jugadores, key=calcular_ganancia_potencial, reverse=True)


# Guardar el resultado en un archivo JSON
with open("jugadores_aumento.json", "w", encoding="utf-8") as archivo_json:
    json.dump(jugadores_aumento, archivo_json)
