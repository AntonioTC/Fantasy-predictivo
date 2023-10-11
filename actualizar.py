import subprocess
import sys
import threading
import time

def run_python_scripts():
    # Ejecutar los tres archivos de Python
    python_files = ['fantasy_scraper.py', 'comprar_vender.py', 'imprimir_resultados.py']
    for file in python_files:
        subprocess.run(['python', file])

# Función para realizar el commit y push con puntos
def git_commit_push():
    for _ in range(3):
        print("Realizando commit y push al repositorio de GitHub", end='')
        sys.stdout.flush()
        for _ in range(3):
            time.sleep(0.5)
            print(".", end='')
            sys.stdout.flush()
        time.sleep(1)

    # Agregar todos los cambios al área de preparación
    subprocess.run(['git', 'add', '.'])

    # Realizar el commit
    subprocess.run(['git', 'commit', '-m', 'Actualización de archivos Python'])

    # Realizar el push al repositorio
    subprocess.run(['git', 'push', 'origin', 'main'])  # Cambia 'main' si tu rama principal es diferente
    print("Realizando commit y push al repositorio de GitHub... [COMPLETADO]")

# Función para imprimir "Actualización de archivos Python completada"
def print_completion_message():
    print("Actualización de archivos Python completada")

if __name__ == "__main__":
    # Ejecutar los archivos de Python
    run_python_scripts()

    # Crear un hilo para el proceso de commit y push
    commit_push_thread = threading.Thread(target=git_commit_push)

    # Iniciar el hilo para el commit y push
    commit_push_thread.start()

    # Esperar a que el hilo termine antes de imprimir el mensaje de finalización
    commit_push_thread.join()

    # Imprimir el mensaje de finalización
    print_completion_message()
