import os
import json
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX
from contextlib import redirect_stdout
import itertools
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        print_end    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()

def calcular_porcentaje_aumento(market_values, num_dias):
    fecha_mas_reciente = max(market_values.keys(), key=lambda x: datetime.strptime(x, "%d/%m/%Y"))
    fecha_inicio = datetime.strptime(fecha_mas_reciente, "%d/%m/%Y") - timedelta(days=num_dias)
    fechas = [fecha_inicio + timedelta(days=d) for d in range(num_dias + 1)]

    valores_fechas = [market_values.get(fecha.strftime("%d/%m/%Y"), None) for fecha in fechas]

    if all(valor is not None for valor in valores_fechas):
        valor_inicial = valores_fechas[0]
        valor_final = valores_fechas[-1]

        porcentaje_aumento = ((valor_final - valor_inicial) / valor_inicial) * 100
        return porcentaje_aumento
    else:
        return None
    
def obtener_mejores_parametros(serie_temporal):
    # Definir los rangos de búsqueda para los parámetros
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 7) for x in pdq]  # Estacionalidad semanal (7 días)

    # Inicializar variables para almacenar los resultados óptimos
    best_aic = float("inf")
    best_order = None
    best_seasonal_order = None

    # Bucle anidado para buscar los mejores parámetros
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = SARIMAX(serie_temporal,
                              order=param,
                              seasonal_order=param_seasonal,
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                results = mod.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = param
                    best_seasonal_order = param_seasonal
            except Exception as e:
                continue

    return best_order, best_seasonal_order

def obtener_prediccion_7_dias(market_values):
    fechas = sorted([datetime.strptime(fecha, "%d/%m/%Y") for fecha in market_values.keys()])
    valores = np.array([market_values[fecha.strftime("%d/%m/%Y")] for fecha in fechas], dtype=float)

    if len(valores) > 1:
        # Crear una serie temporal utilizando pandas
        valores_enteros = valores.astype(int)
        fechas = pd.date_range(start=fechas[0], periods=len(valores), freq='D')
        serie_temporal = pd.Series(valores_enteros, index=fechas)
        
        try:
            # Ajustar el modelo SARIMA a la serie temporal completa
            modelo_sarima = SARIMAX(serie_temporal, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
            modelo_sarima_fit = modelo_sarima.fit(disp=False)

            # Realizar la predicción de los próximos 7 días
            prediccion = modelo_sarima_fit.get_forecast(steps=30)
            prediccion_7_dias = prediccion.predicted_mean.iloc[-1]  # Última predicción para 7 días

        except Exception as e:
            print("Error al ajustar el modelo SARIMA: " + str(e))
            # Si hay algún error en el ajuste del modelo SARIMA, utilizar Random Forest Regressor
            x = np.arange(len(valores)).reshape(-1, 1)
            y = valores.ravel()  # Utilizar ravel() para convertir a una matriz 1D
            reg = RandomForestRegressor()
            reg.fit(x, y)

            prediccion_7_dias = reg.predict([[len(valores) + 7]])[0]  # Quitar [0][0] para obtener el valor único

        return prediccion_7_dias
    else:
        return None
    
import numpy as np

def procesar_jugadores(ruta_players):
    lista_jugadores = []

    jugador_procesado = 0

    total_jugadores = sum(1 for equipo in os.listdir(ruta_players) for jugador_json in os.listdir(os.path.join(ruta_players, equipo)))

    for equipo in os.listdir(ruta_players):
        ruta_equipo = os.path.join(ruta_players, equipo)
        if os.path.isdir(ruta_equipo):
            for jugador_json in os.listdir(ruta_equipo):
                ruta_jugador = os.path.join(ruta_equipo, jugador_json)
                jugador_procesado += 1

                #if(jugador_procesado > 15 ):
                #    break
                #else:
                with open(ruta_jugador, "r") as archivo_json:
                    datos_jugador = json.load(archivo_json)
                    id_jugador = datos_jugador["id"]
                    int_id_jugador = int(id_jugador)
                    nombre_jugador = datos_jugador["name"]
                    market_values = datos_jugador["marketValue"]

                    fechas = sorted([datetime.strptime(fecha, "%d/%m/%Y") for fecha in market_values.keys()])
                    valores = np.array([market_values[fecha.strftime("%d/%m/%Y")] for fecha in fechas], dtype=float)

                    porcentaje_aumento_15_dias = calcular_porcentaje_aumento(market_values, 15)

                    # Calcular el porcentaje de aumento a 7 y 5 días
                    porcentaje_aumento_7_dias = calcular_porcentaje_aumento(market_values, 7)
                    porcentaje_aumento_5_dias = calcular_porcentaje_aumento(market_values, 5)

                    # Calcular la ganancia potencial para cada jugador según el modelo sin printear nada en consola
                    prediccion_7_dias = obtener_prediccion_7_dias(market_values)


                    ganancia_potencial = prediccion_7_dias - valores[-1] if prediccion_7_dias is not None else -15

                    #Calcular % de ganancia potencial respecto al precio actual
                    porcentaje_ganancia_potencial = (ganancia_potencial / valores[-1]) * 100 if ganancia_potencial is not None else -15

                    # Guardar los market values en formato JSON
                    market_values_json = json.dumps(market_values)

                    # Añadir la información del jugador a la lista
                    jugador_info = {
                        "nombre": nombre_jugador,
                        "nickname": datos_jugador["nickname"] if "nickname" in datos_jugador else "",
                        "id": id_jugador,
                        "precio_actual": valores[-1],
                        "precio_hace_15_dias": valores[0],
                        "precio_hace_7_dias": valores[-7] if len(valores) >= 8 else None,
                        "precio_hace_5_dias": valores[-5] if len(valores) >= 6 else None,
                        "porcentaje_aumento_15_dias": porcentaje_aumento_15_dias,
                        "porcentaje_aumento_7_dias": porcentaje_aumento_7_dias,
                        "porcentaje_aumento_5_dias": porcentaje_aumento_5_dias,
                        "ganancia_potencial": ganancia_potencial,
                        "prediccion_7_dias": prediccion_7_dias,
                        "porcentaje_ganancia_potencial": porcentaje_ganancia_potencial,
                        "accion_potencial": obtener_accion_potencial(porcentaje_ganancia_potencial),
                        "marketValue": market_values_json
                    }

                    lista_jugadores.append(jugador_info)

                 # Calcular el progreso actual y mostrar la barra de progreso
                print_progress_bar(jugador_procesado, total_jugadores, prefix='Progreso:', suffix='Completado', length=70)
    
    # Ordenar la lista de jugadores por la ganancia potencial (de mayor a menor)
    jugadores_aumento = sorted(lista_jugadores, key=lambda x: x["ganancia_potencial"], reverse=True)

    with open("jugadores_aumento.json", "w", encoding="utf-8") as archivo_json:
        json.dump(jugadores_aumento, archivo_json)


def obtener_accion_potencial(porcentaje_ganancia_potencial):
    # Verificar si porcentaje_ganancia_potencial es None y manejarlo
    if porcentaje_ganancia_potencial is None:
        return "No se puede clasificar"

    # Definir las categorías y sus rangos de porcentaje
    categorias = {
        "muy recomendable": (10.0, np.inf),
        "recomendable": (5.0, 10.0),
        "algo recomendable": (2.5, 5.0),
        "mantener": (-2.5, 2.5),
        "poco recomendable": (-5.0, -2.5),
        "muy poco recomendable": (-10.0, -5.0),
        "nada recomendable": (-np.inf, -10.0)
    }

    # Determinar la acción en base al porcentaje de ganancia potencial
    for categoria, rango in categorias.items():
        if rango[0] <= porcentaje_ganancia_potencial <= rango[1]:
            return categoria

    return "No se puede clasificar"

ruta_players = "players"

procesar_jugadores(ruta_players)