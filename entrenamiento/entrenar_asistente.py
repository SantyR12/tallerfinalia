"""
Entrena desde cero el Transformer del asistente.

    python entrenamiento/generar_preguntas.py      (crea el CSV, una sola vez)
    python entrenamiento/entrenar_asistente.py

Lee datos/preguntas_asistente.csv, arma el vocabulario, entrena el modelo de
src/transformer.py y guarda:
    modelos/asistente_transformer.keras
    modelos/asistente_vocabulario.json   (vocabulario + lista de intenciones)
Tarda alrededor de un minuto en un computador sin tarjeta gráfica.
"""

import csv
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import keras  # noqa: E402

import config  # noqa: E402
from src.texto import codificar, normalizar  # noqa: E402
from src.transformer import construir_transformer  # noqa: E402


def main(epocas=25, semilla=7):
    """Lee las preguntas, entrena el clasificador de intenciones y lo guarda."""
    keras.utils.set_random_seed(semilla)        # misma semilla = mismos resultados al repetir
    with open(config.RUTA_PREGUNTAS, encoding="utf-8") as archivo:
        filas = [(f["pregunta"], f["intencion"]) for f in csv.DictReader(archivo)]
    random.Random(semilla).shuffle(filas)       # barajar antes de repartir en conjuntos

    # Vocabulario: 0 = relleno, 1 = palabra desconocida, 2... = palabras del dataset
    palabras = sorted({p for pregunta, _ in filas for p in normalizar(pregunta).split()})
    vocabulario = {palabra: indice for indice, palabra in enumerate(palabras, start=2)}
    intenciones = sorted({intencion for _, intencion in filas})

    # x: cada pregunta como 16 números. y: el número de su intención.
    x = np.array([codificar(p, vocabulario, config.LARGO_PREGUNTA) for p, _ in filas], dtype="int32")
    y = np.array([intenciones.index(i) for _, i in filas], dtype="int32")

    # 70 % entrenamiento, 15 % validación, 15 % prueba
    n1, n2 = int(0.70 * len(x)), int(0.85 * len(x))
    modelo = construir_transformer(config.LARGO_PREGUNTA, len(vocabulario) + 2, len(intenciones))
    # Adam ajusta los pesos; la entropía cruzada es la pérdida para elegir una
    # entre varias clases (aquí, 17 intenciones); accuracy = fracción de aciertos
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    modelo.summary()
    # EarlyStopping detiene el entrenamiento si la validación deja de mejorar
    modelo.fit(
        x[:n1], y[:n1], validation_data=(x[n1:n2], y[n1:n2]),
        epochs=epocas, batch_size=64, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True)],
    )
    perdida, exactitud = modelo.evaluate(x[n2:], y[n2:], verbose=0)
    print(f"\nExactitud en prueba: {exactitud:.3f} ({len(x) - n2} preguntas)")
    print("Ojo: las preguntas de prueba salen de las mismas plantillas que las de "
          "entrenamiento, así que esta cifra es optimista frente a preguntas reales.")

    os.makedirs(os.path.dirname(config.RUTA_ASISTENTE), exist_ok=True)
    modelo.save(config.RUTA_ASISTENTE)
    with open(config.RUTA_VOCABULARIO, "w", encoding="utf-8") as archivo:
        json.dump({"vocabulario": vocabulario, "intenciones": intenciones,
                   "largo": config.LARGO_PREGUNTA}, archivo, ensure_ascii=False, indent=1)
    print(f"Modelo guardado en {config.RUTA_ASISTENTE}")


if __name__ == "__main__":
    main()
