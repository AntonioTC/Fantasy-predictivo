import os
import subprocess

def run_python_scripts():
    # Ejecutar los tres archivos de Python
    python_files = ['fantasy_scraper.py', 'comprar_vender.py', 'imprimir_resultados.py']
    for file in python_files:
        subprocess.run(['python', file])

def git_commit_push():
    # Agregar todos los cambios al área de preparación
    subprocess.run(['git', 'add', '.'])

    # Realizar el commit
    subprocess.run(['git', 'commit', '-m', 'Actualización de archivos Python'])

    # Realizar el push al repositorio
    subprocess.run(['git', 'push', 'origin', 'main'])  # Cambia 'main' si tu rama principal es diferente

if __name__ == "__main__":
    # Ejecutar los archivos de Python
    run_python_scripts()
    print('Archivos Python ejecutados')

    # Hacer un commit y push al repositorio de GitHub
    git_commit_push()

    print('Actualización de archivos Python completada')
