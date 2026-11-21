"""
Texto -> números. Funciones compartidas por el clasificador de intenciones y
por el modelo de lenguaje (no dependen de Keras).
"""

import re
import unicodedata


def normalizar(texto):
    """Minúsculas, sin tildes ni signos: '¿Cuántas Colisiones?' -> 'cuantas colisiones'."""
    texto = unicodedata.normalize("NFD", texto.lower())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9ñ ]+", " ", texto).strip()


def codificar(texto, vocabulario, largo):
    """
    Convierte la pregunta en una lista de `largo` índices.
    0 = relleno, 1 = palabra desconocida.
    """
    indices = [vocabulario.get(palabra, 1) for palabra in normalizar(texto).split()][:largo]
    return indices + [0] * (largo - len(indices))
