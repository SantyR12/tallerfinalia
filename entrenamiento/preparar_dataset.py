"""
Prepara el dataset de la CNN: de carpetas de imágenes a un CSV.

    python entrenamiento/preparar_dataset.py --origen RUTA/AL/DATASET/train

`--origen` es una carpeta con una subcarpeta por clase, que es como vienen los
datasets de clasificación (por ejemplo MIO-TCD-Classification):

    train/
        car/            00000001.jpg ...
        motorcycle/
        pedestrian/
        background/
        ...

El script toma hasta `--por-clase` imágenes de cada clase del proyecto, las
reparte en entrenamiento, validación y prueba, y escribe datos/dataset_objetos.csv
con tres columnas: ruta, clase, conjunto.
"""

import argparse
import csv
import os
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

# Nombre de la carpeta en el dataset -> clase del proyecto.
# Sirve para MIO-TCD y para datasets con nombres en inglés o español.
# Para otro dataset basta con agregar aquí sus nombres de carpeta.
EQUIVALENCIAS = {
    "car": "carro", "pickup_truck": "carro", "work_van": "carro", "carro": "carro",
    "motorcycle": "motocicleta", "motorbike": "motocicleta", "motocicleta": "motocicleta",
    "pedestrian": "persona", "person": "persona", "persona": "persona",
    "bus": "bus",
    "articulated_truck": "camion", "single_unit_truck": "camion", "truck": "camion", "camion": "camion",
    "bicycle": "bicicleta", "bicicleta": "bicicleta",
    "background": "fondo", "fondo": "fondo",
}
EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp"}


def nombre_normal(carpeta):
    return carpeta.lower().strip().replace(" ", "_").replace("-", "_")


def main():
    parser = argparse.ArgumentParser(description="Crea el CSV del dataset de objetos")
    parser.add_argument("--origen", required=True, help="carpeta con una subcarpeta por clase")
    parser.add_argument("--por-clase", type=int, default=4000, help="máximo de imágenes por clase")
    parser.add_argument("--salida", default="datos/dataset_objetos.csv")
    parser.add_argument("--semilla", type=int, default=7)
    args = parser.parse_args()
    random.seed(args.semilla)

    origen = Path(args.origen).expanduser()
    if not origen.is_dir():
        sys.exit(
            f"No existe la carpeta '{origen}'.\n"
            "Hay que descargar primero el dataset de imágenes y descomprimirlo; después se\n"
            "pasa en --origen la carpeta que contiene una subcarpeta por clase (car, bus,\n"
            "pedestrian, background...). En el Finder puedes arrastrar esa carpeta a la\n"
            "Terminal para que escriba la ruta exacta."
        )

    por_clase = {}
    ignoradas = []
    for carpeta in sorted(origen.iterdir()):
        if not carpeta.is_dir():
            continue
        clase = EQUIVALENCIAS.get(nombre_normal(carpeta.name))
        if clase is None:
            ignoradas.append(carpeta.name)
            continue
        imagenes = [str(p) for p in carpeta.rglob("*") if p.suffix.lower() in EXTENSIONES]
        por_clase.setdefault(clase, []).extend(imagenes)

    faltantes = [c for c in config.CLASES if c not in por_clase]
    if faltantes:
        print(f"AVISO: el dataset no tiene imágenes para: {', '.join(faltantes)}. "
              "La red no aprenderá esas clases.")
    if ignoradas:
        print(f"Carpetas ignoradas (no corresponden a ninguna clase): {', '.join(ignoradas)}")
    if not por_clase:
        subcarpetas = ", ".join(sorted(c.name for c in origen.iterdir() if c.is_dir())) or "ninguna"
        sys.exit(
            f"En '{origen}' no hay ninguna carpeta de clase conocida.\n"
            f"Subcarpetas encontradas: {subcarpetas}.\n"
            "Si las clases están un nivel más adentro, pasa esa carpeta en --origen. Si tienen "
            "otros nombres, agrégalos al diccionario EQUIVALENCIAS de este archivo."
        )

    filas = []
    for clase, imagenes in sorted(por_clase.items()):
        random.shuffle(imagenes)
        imagenes = imagenes[:args.por_clase]
        n1, n2 = int(0.70 * len(imagenes)), int(0.85 * len(imagenes))
        for i, ruta in enumerate(imagenes):
            conjunto = "entrenamiento" if i < n1 else "validacion" if i < n2 else "prueba"
            filas.append((ruta, clase, conjunto))
    random.shuffle(filas)

    os.makedirs(os.path.dirname(args.salida) or ".", exist_ok=True)
    with open(args.salida, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["ruta", "clase", "conjunto"])
        escritor.writerows(filas)

    conteo = Counter((clase, conjunto) for _, clase, conjunto in filas)
    print(f"\n{len(filas)} imágenes -> {args.salida}")
    print(f"{'clase':14s}{'entrenamiento':>14s}{'validacion':>12s}{'prueba':>8s}")
    for clase in sorted(por_clase):
        print(f"{clase:14s}{conteo[(clase, 'entrenamiento')]:>14d}"
              f"{conteo[(clase, 'validacion')]:>12d}{conteo[(clase, 'prueba')]:>8d}")


if __name__ == "__main__":
    main()
