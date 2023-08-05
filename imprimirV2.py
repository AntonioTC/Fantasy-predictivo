import json
import numpy as np
import locale
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from PIL import Image
import io
import os

locale.setlocale(locale.LC_ALL, 'es_ES')

def obtener_accion_potencial(porcentaje_ganancia_potencial):
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

def graficar_historial_precios(jugador, img_folder):
    market_values = json.loads(jugador["marketValue"])
    fechas = sorted([datetime.strptime(fecha, "%d/%m/%Y") for fecha in market_values.keys()])
    valores = np.array([market_values[fecha.strftime("%d/%m/%Y")] for fecha in fechas], dtype=float) / 1000000  # Dividir por un millón
    fig, ax = plt.subplots()  # Crear una nueva figura para cada jugador
    ax.plot(fechas, valores, label=jugador["nombre"])
    ax.set_xlabel("Fechas")
    ax.set_ylabel("Precios (M)")
    ax.set_title(f"Historial de precios de {jugador['nombre']}")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))  # Formato de fechas en el eje x
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Convertir la figura a imagen
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)

    # Cerrar la figura para liberar memoria
    plt.close(fig)

    # Guardar la imagen en la carpeta de imágenes
    img_path = os.path.join(img_folder, f"{jugador['nombre']}_historial.png")
    img.save(img_path)

    return img_path

# Crear la carpeta para las imágenes
img_folder = "img"
os.makedirs(img_folder, exist_ok=True)

# Leer los datos del archivo JSON
with open("jugadores_aumento.json", "r", encoding="utf-8") as archivo_json:
    jugadores_aumento = json.load(archivo_json)

# Crear el documento HTML
html_content = "<html><head><meta charset='UTF-8'></head><body>"

for jugador in jugadores_aumento:
    html_content += f"<h2>Nombre: {jugador['nombre']}</h2>"
    html_content += f"<p>Nick: {jugador['nickname']} (ID: {jugador['id']})</p>"
    html_content += f"<p>Precio actual: {locale.format_string('%.2f', jugador['precio_actual'], grouping=True)}€</p>"

    if jugador.get('porcentaje_aumento_5_dias') is not None and isinstance(jugador['porcentaje_aumento_5_dias'], (int, float)):
        html_content += f"<p>Porcentaje de aumento en 5 días: {jugador['porcentaje_aumento_5_dias']:.2f}%</p>"
    if jugador.get('porcentaje_aumento_7_dias') is not None and isinstance(jugador['porcentaje_aumento_7_dias'], (int, float)):
        html_content += f"<p>Porcentaje de aumento en 7 días: {jugador['porcentaje_aumento_7_dias']:.2f}%</p>"
    if jugador.get('porcentaje_aumento_15_dias') is not None and isinstance(jugador['porcentaje_aumento_15_dias'], (int, float)):
        html_content += f"<p>Porcentaje de aumento en 15 días: {jugador['porcentaje_aumento_15_dias']:.2f}%</p>"

    if jugador.get('prediccion_7_dias') is not None and isinstance(jugador['prediccion_7_dias'], (int, float)):
        html_content += f"<p>Predicción de precio: [{locale.format_string('%.2f', jugador['prediccion_7_dias'] / 1.05, grouping=True)}€ / {locale.format_string('%.2f', jugador['prediccion_7_dias'], grouping=True)}€ / {locale.format_string('%.2f', jugador['prediccion_7_dias'] * 1.05, grouping=True)}€] (-0,5% / = / +0,5%)</p>"
        html_content += f"<p>Ganancia potencial: {locale.format_string('%.2f', jugador['ganancia_potencial'], grouping=True)}€ ({jugador['porcentaje_ganancia_potencial']:.2f}%)</p>"
        html_content += f"<p>Acción a tomar: {obtener_accion_potencial(jugador['porcentaje_ganancia_potencial'])}</p>"
    else:
        html_content += "<p>No se puede calcular la predicción de precio a 7 días debido a datos faltantes.</p>"

    # Guardar la imagen como archivo temporal
    imagen_path = graficar_historial_precios(jugador, img_folder)

    # Agregar la imagen al documento HTML
    html_content += f"<h2>Historial de precios de {jugador['nombre']}</h2>"
    html_content += f"<img src='{imagen_path}' alt='Historial de precios de {jugador['nombre']}'><br><br>"

html_content += "</body></html>"

# Guardar el contenido en un archivo HTML
with open("index.html", "w", encoding="utf-8") as archivo_html:
    archivo_html.write(html_content)
