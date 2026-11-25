"""
Modelo de lenguaje propio: un Transformer de solo decodificador (la
arquitectura de GPT) en tamaño pequeño, entrenado desde cero.

Tarea de entrenamiento: predecir la SIGUIENTE ficha. Dada la secuencia

    <datos> ... <pregunta> cuantas colisiones hubo <respuesta> se detectó

el modelo aprende que la ficha siguiente es "<n_colisiones>". Para generar una
respuesta se repite esa predicción: se añade la ficha elegida al final y se
vuelve a preguntar, hasta que el modelo escribe <fin>. Eso es generación
AUTORREGRESIVA, la misma forma en que escribe cualquier LLM.

    fichas
      │
    EmbeddingConPosicion             vector de la ficha + vector de su posición
      │
    BloqueTransformer(causal) × N    cada ficha solo mira a las anteriores
      │
    Dense(vocabulario) + softmax     probabilidad de cada ficha del vocabulario
                                     como siguiente
"""

import json

import keras
import numpy as np
from keras import layers

import config
from src.lenguaje import FIN
from src.transformer import BloqueTransformer, EmbeddingConPosicion


def construir_modelo_lenguaje(vocabulario, largo=config.LARGO_SECUENCIA, dim=config.DIM_LLM,
                              cabezas=config.CABEZAS_LLM, bloques=config.BLOQUES_LLM):
    """
    Arma el modelo de lenguaje. Entrada: `largo` fichas como números.
    Salida: para CADA posición, la probabilidad de cada ficha del vocabulario
    de ser la siguiente.
    """
    entrada = keras.Input(shape=(largo,), dtype="int32", name="fichas")
    x = EmbeddingConPosicion(largo, vocabulario, dim)(entrada)      # números -> vectores
    for _ in range(bloques):
        # causal=True: cada ficha solo puede mirar a las anteriores. Sin eso el
        # modelo "vería la respuesta" al entrenar y no aprendería a escribir.
        x = BloqueTransformer(dim, cabezas, dim * 2, causal=True)(x, entrada)
    salida = layers.Dense(vocabulario, activation="softmax", name="siguiente_ficha")(x)
    return keras.Model(entrada, salida, name="modelo_lenguaje")


class ModeloLenguaje:
    """El modelo entrenado más su vocabulario, listo para generar."""

    def __init__(self, ruta_modelo=config.RUTA_LLM, ruta_vocabulario=config.RUTA_VOCABULARIO_LLM):
        with open(ruta_vocabulario, encoding="utf-8") as archivo:
            datos = json.load(archivo)
        self.indice = datos["vocabulario"]                      # ficha -> número
        self.ficha = {n: f for f, n in self.indice.items()}     # número -> ficha
        self.largo = datos["largo"]
        self.modelo = keras.models.load_model(ruta_modelo)

    def generar(self, fichas_entrada, maximo=config.MAX_FICHAS_RESPUESTA, temperatura=0.0):
        """
        Continúa la secuencia ficha por ficha hasta escribir <fin>.
        Devuelve (fichas generadas, terminó bien).
        temperatura = 0: siempre la ficha más probable. Mayor que 0: sortea
        entre las probables, lo que da respuestas más variadas.
        """
        secuencia = [self.indice.get(f, 1) for f in fichas_entrada]     # 1 = desconocida
        generadas = []
        for _ in range(maximo):                 # una vuelta del ciclo = una ficha escrita
            if len(secuencia) >= self.largo:
                break                           # no cabe más texto en la entrada del modelo
            # El modelo siempre recibe `largo` números: lo que falta se rellena con ceros
            relleno = secuencia + [0] * (self.largo - len(secuencia))
            probabilidades = np.asarray(
                self.modelo(np.array([relleno], dtype="int32"), training=False)
            )[0, len(secuencia) - 1]            # predicción para la posición siguiente
            if temperatura > 0:
                # Sorteo: la temperatura aplana (alta) o afila (baja) las probabilidades
                logaritmos = np.log(probabilidades + 1e-9) / temperatura
                p = np.exp(logaritmos - logaritmos.max())
                siguiente = int(np.random.choice(len(p), p=p / p.sum()))
            else:
                siguiente = int(probabilidades.argmax())
            ficha = self.ficha.get(siguiente, "")
            if ficha == FIN:
                return generadas, True          # el modelo decidió terminar la frase
            generadas.append(ficha)
            secuencia.append(siguiente)         # la ficha escrita pasa a ser parte de la entrada
        return generadas, False                 # se acabó el espacio sin llegar a <fin>
