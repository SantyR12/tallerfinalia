"""
Paso 2 de la detección: decir qué es cada objeto.

Una red neuronal convolucional (CNN) PROPIA, entrenada desde cero con
entrenamiento/entrenar_cnn.py, recibe el recorte de cada zona en movimiento y
calcula la probabilidad de que sea carro, bus, camión, motocicleta, bicicleta,
persona o fondo. Las probabilidades de carro, bus y camión se suman en un solo
grupo, "vehiculo" (ver config.GRUPOS). No se usa ningún modelo preentrenado.
"""

import json
from pathlib import Path

import cv2
import numpy as np

import config


def encuadrar(imagen, tam=config.TAM_IMAGEN):
    """
    Lleva una imagen BGR a un cuadro RGB de tam x tam SIN deformarla: se escala
    hasta que el lado más largo mida `tam` y lo que falta se rellena con gris.
    Así una persona sigue viéndose alta y angosta y un carro, ancho; si la
    imagen se estirara hasta llenar el cuadro, esa pista se perdería.
    La usan por igual el entrenamiento y la detección.
    """
    alto, ancho = imagen.shape[:2]
    escala = tam / max(alto, ancho)
    nuevo_ancho, nuevo_alto = max(1, round(ancho * escala)), max(1, round(alto * escala))
    metodo = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
    reducida = cv2.resize(imagen, (nuevo_ancho, nuevo_alto), interpolation=metodo)
    cuadro = np.full((tam, tam, 3), 128, dtype=np.uint8)
    y, x = (tam - nuevo_alto) // 2, (tam - nuevo_ancho) // 2
    cuadro[y:y + nuevo_alto, x:x + nuevo_ancho] = reducida
    return cv2.cvtColor(cuadro, cv2.COLOR_BGR2RGB)          # OpenCV lee en BGR


def preparar_recorte(frame, caja, tam=config.TAM_IMAGEN):
    """Recorta la caja y la deja como la espera la red: tam x tam, RGB, entre 0 y 1."""
    alto, ancho = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in caja)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, ancho), min(y2, alto)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return np.full((tam, tam, 3), 0.5, dtype=np.float32)
    return encuadrar(frame[y1:y2, x1:x2], tam).astype(np.float32) / 255.0


class ClasificadorCNN:
    def __init__(self, ruta_modelo=config.RUTA_CNN, ruta_clases=config.RUTA_CLASES):
        if not Path(ruta_modelo).exists():
            raise FileNotFoundError(
                f"No se encontró la red entrenada en '{ruta_modelo}'. "
                "Entrénala primero con: python entrenamiento/entrenar_cnn.py"
            )
        import keras    # se importa aquí para que las reglas se puedan probar sin Keras

        self.modelo = keras.models.load_model(ruta_modelo)
        with open(ruta_clases, encoding="utf-8") as archivo:
            self.clases = json.load(archivo)                    # las que aprendió la red
        # Matriz que suma las probabilidades de las clases de un mismo grupo
        de_grupo = [config.GRUPOS.get(clase, clase) for clase in self.clases]
        self.grupos = sorted(set(de_grupo))                     # las que reporta el sistema
        self.sumar = np.zeros((len(self.clases), len(self.grupos)), dtype=np.float32)
        for i, grupo in enumerate(de_grupo):
            self.sumar[i, self.grupos.index(grupo)] = 1.0

    def clasificar(self, frame, cajas):
        """Devuelve, para cada caja, la pareja (clase, confianza)."""
        if not cajas:
            return []
        lote = np.stack([preparar_recorte(frame, caja) for caja in cajas])
        probabilidades = np.asarray(self.modelo(lote, training=False))     # salida softmax
        probabilidades = probabilidades @ self.sumar                       # por grupo
        indices = probabilidades.argmax(axis=1)
        return [(self.grupos[i], float(probabilidades[n, i])) for n, i in enumerate(indices)]
