"""
Entrena desde cero el modelo de lenguaje propio (Transformer decodificador).

    python entrenamiento/generar_corpus.py      (crea el CSV, una sola vez)
    python entrenamiento/entrenar_llm.py

Cada fila del corpus se convierte en una secuencia de fichas:

    <datos> ... <pregunta> ... <respuesta> ... <fin>

y el modelo aprende a predecir cada ficha a partir de las anteriores. Guarda:
    modelos/modelo_lenguaje.keras
    modelos/modelo_lenguaje_vocabulario.json
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
from src.lenguaje import DATOS, FIN, PREGUNTA, RESPUESTA, fichas_de_pregunta  # noqa: E402
from src.modelo_lenguaje import construir_modelo_lenguaje  # noqa: E402


def secuencia_de_fila(fila):
    """Devuelve (fichas, posición donde empieza la pregunta)."""
    datos = fila["datos"].split()
    pregunta = fichas_de_pregunta(fila["pregunta"], config.LARGO_PREGUNTA)
    fichas = [DATOS] + datos + [PREGUNTA] + pregunta + [RESPUESTA] + fila["respuesta"].split() + [FIN]
    return fichas, len(datos) + 2


def main(epocas=10, semilla=11):
    """Lee el corpus, entrena el modelo de lenguaje, lo guarda y lo evalúa."""
    keras.utils.set_random_seed(semilla)        # misma semilla = mismos resultados al repetir
    with open(config.RUTA_CORPUS, encoding="utf-8") as archivo:
        filas = list(csv.DictReader(archivo))
    random.Random(semilla).shuffle(filas)
    secuencias = [secuencia_de_fila(f) for f in filas]
    mas_larga = max(len(fichas) for fichas, _ in secuencias)
    assert mas_larga <= config.LARGO_SECUENCIA, f"Sube LARGO_SECUENCIA a {mas_larga} o más"

    # Vocabulario: 0 = relleno, 1 = desconocida, 2... = fichas del corpus
    fichas_unicas = sorted({f for fichas, _ in secuencias for f in fichas})
    vocabulario = {ficha: n for n, ficha in enumerate(fichas_unicas, start=2)}
    tam_vocabulario = len(vocabulario) + 2

    # x = la secuencia; y = la misma secuencia corrida una posición (la "siguiente ficha")
    largo = config.LARGO_SECUENCIA
    x = np.zeros((len(secuencias), largo), dtype="int32")
    y = np.zeros((len(secuencias), largo), dtype="int32")
    peso = np.zeros((len(secuencias), largo), dtype="float32")
    for i, (fichas, inicio_pregunta) in enumerate(secuencias):
        indices = [vocabulario[f] for f in fichas]
        x[i, :len(indices)] = indices
        y[i, :len(indices) - 1] = indices[1:]
        # Solo cuenta el error al predecir la pregunta y la respuesta. Los datos
        # del video son al azar: no hay nada que aprender en predecirlos.
        peso[i, inicio_pregunta - 1:len(indices) - 1] = 1.0

    # 85 % entrenamiento, 10 % validación; el 5 % restante queda sin usar,
    # porque la evaluación final se hace con casos nuevos (ver evaluar())
    n1, n2 = int(0.85 * len(x)), int(0.95 * len(x))
    modelo = construir_modelo_lenguaje(tam_vocabulario)
    # La pérdida es la misma de un clasificador (entropía cruzada), aplicada en
    # cada posición: en cada una hay que "clasificar" cuál es la ficha siguiente
    modelo.compile(optimizer=keras.optimizers.Adam(2e-3), loss="sparse_categorical_crossentropy",
                   weighted_metrics=["accuracy"])
    modelo.summary()
    modelo.fit(
        x[:n1], y[:n1], sample_weight=peso[:n1],
        validation_data=(x[n1:n2], y[n1:n2], peso[n1:n2]),
        epochs=epocas, batch_size=64, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
    )

    os.makedirs(os.path.dirname(config.RUTA_LLM), exist_ok=True)
    modelo.save(config.RUTA_LLM)
    with open(config.RUTA_VOCABULARIO_LLM, "w", encoding="utf-8") as archivo:
        json.dump({"vocabulario": vocabulario, "largo": largo}, archivo, ensure_ascii=False, indent=1)
    print(f"\nVocabulario: {tam_vocabulario} fichas. Secuencia más larga: {mas_larga} fichas.")
    print(f"Modelo guardado en {config.RUTA_LLM}")

    from src.modelo_lenguaje import ModeloLenguaje
    evaluar(ModeloLenguaje())


def evaluar(generador, casos=300, semilla=999):
    """
    Lo que importa no es acertar ficha por ficha sino que la respuesta completa
    que escribe el modelo sea correcta. Se inventan videos y preguntas nuevos,
    el modelo responde, y se comprueba si lo que escribió es una de las formas
    válidas de responder esa pregunta para ese video.
    """
    from entrenamiento.generar_preguntas import generar as generar_preguntas
    from src.lenguaje import armar_entrada, cuenta_al_azar, respuestas_validas

    rng = random.Random(semilla)
    por_intencion = {}
    for pregunta, intencion in generar_preguntas():
        por_intencion.setdefault(intencion, []).append(pregunta)
    correctas = 0
    for _ in range(casos):
        intencion = rng.choice(sorted(por_intencion))
        cuenta = cuenta_al_azar(rng)
        entrada = armar_entrada(cuenta, rng.choice(por_intencion[intencion]), config.LARGO_PREGUNTA)
        fichas, termino = generador.generar(entrada)
        correctas += termino and " ".join(fichas) in respuestas_validas(intencion, cuenta)
    print(f"Respuestas correctas en {casos} casos nuevos: {correctas} ({100 * correctas / casos:.1f} %)")
    print("Ojo: las preguntas salen de las mismas plantillas del entrenamiento; con preguntas "
          "escritas libremente el porcentaje es menor.")
    return correctas / casos


if __name__ == "__main__":
    main()
