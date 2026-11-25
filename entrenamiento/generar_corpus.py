"""
Crea el corpus del modelo de lenguaje: un CSV con miles de ejemplos de
(datos del video, pregunta, respuesta).

    python entrenamiento/generar_corpus.py

Para cada ejemplo se inventa un video (cuántos eventos y objetos hubo), se
elige una pregunta del dataset de preguntas y una de las formas correctas de
responderla para ese video. El resultado queda en datos/corpus_asistente.csv
con tres columnas: datos, pregunta, respuesta.
"""

import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from entrenamiento.generar_preguntas import generar as generar_preguntas  # noqa: E402
from src.lenguaje import cuenta_al_azar, fichas_de_datos, respuestas_validas  # noqa: E402


def main(ejemplos=24000, semilla=11):
    """Escribe el CSV con `ejemplos` filas (datos, pregunta, respuesta)."""
    rng = random.Random(semilla)        # generador de azar con semilla: siempre da el mismo corpus
    # Se agrupan las preguntas por intención: {"contar_colisiones": [...], ...}
    por_intencion = {}
    for pregunta, intencion in generar_preguntas():
        por_intencion.setdefault(intencion, []).append(pregunta)
    intenciones = sorted(por_intencion)

    filas = []
    for _ in range(ejemplos):
        intencion = rng.choice(intenciones)             # todas las intenciones por igual
        cuenta = cuenta_al_azar(rng)                    # un video inventado
        filas.append((
            " ".join(fichas_de_datos(cuenta)),          # "colisiones_uno riesgos_cero ..."
            rng.choice(por_intencion[intencion]),       # una forma de hacer la pregunta
            rng.choice(respuestas_validas(intencion, cuenta)),     # una respuesta correcta
        ))

    os.makedirs(os.path.dirname(config.RUTA_CORPUS), exist_ok=True)
    with open(config.RUTA_CORPUS, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["datos", "pregunta", "respuesta"])
        escritor.writerows(filas)
    print(f"{len(filas)} ejemplos -> {config.RUTA_CORPUS}")
    for datos, pregunta, respuesta in filas[:3]:
        print(f"\n  datos:     {datos[:70]}...\n  pregunta:  {pregunta}\n  respuesta: {respuesta}")


if __name__ == "__main__":
    main()
