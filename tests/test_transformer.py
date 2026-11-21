"""
Pruebas del clasificador de intenciones (Transformer codificador propio).

Entrenan un modelo pequeño durante unos segundos con una parte del dataset de
preguntas y comprueban que aprende, que se puede guardar y volver a cargar, y
que el asistente lo usa para responder.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

keras = pytest.importorskip("keras")
np = pytest.importorskip("numpy")

from entrenamiento.generar_preguntas import generar  # noqa: E402
from src.asistente import Asistente  # noqa: E402
from src.transformer import codificar, construir_transformer, normalizar  # noqa: E402
from tests.test_asistente import EVENTOS, RESUMEN  # noqa: E402

LARGO = 16


def test_normalizar_y_codificar():
    assert normalizar("¿Cuántas COLISIONES hubo?") == "cuantas colisiones hubo"
    vocabulario = {"cuantas": 2, "colisiones": 3}
    indices = codificar("¿Cuántas colisiones hubo?", vocabulario, 6)
    assert indices == [2, 3, 1, 0, 0, 0]        # 1 = desconocida, 0 = relleno


def test_el_dataset_de_preguntas_cubre_todas_las_intenciones():
    filas = generar()
    intenciones = {intencion for _, intencion in filas}
    assert len(intenciones) == 17
    assert {"contar_colisiones", "momento_caidas", "resumen", "contar_motos",
            "contar_vehiculos", "tipo_vehiculo"} <= intenciones


@pytest.fixture(scope="module")
def modelo_entrenado(tmp_path_factory):
    """Entrena unos segundos con tres intenciones fáciles de distinguir."""
    keras.utils.set_random_seed(1)
    usar = ["contar_colisiones", "contar_motos", "duracion"]
    filas = [(p, i) for p, i in generar() if i in usar][::3]
    palabras = sorted({w for p, _ in filas for w in normalizar(p).split()})
    vocabulario = {w: n for n, w in enumerate(palabras, start=2)}
    x = np.array([codificar(p, vocabulario, LARGO) for p, _ in filas], dtype="int32")
    y = np.array([usar.index(i) for _, i in filas], dtype="int32")

    modelo = construir_transformer(LARGO, len(vocabulario) + 2, len(usar))
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    modelo.fit(x, y, epochs=6, batch_size=32, verbose=0)
    exactitud = modelo.evaluate(x, y, verbose=0)[1]

    carpeta = tmp_path_factory.mktemp("modelo")
    ruta_modelo, ruta_vocabulario = str(carpeta / "t.keras"), str(carpeta / "v.json")
    modelo.save(ruta_modelo)
    with open(ruta_vocabulario, "w", encoding="utf-8") as archivo:
        json.dump({"vocabulario": vocabulario, "intenciones": usar, "largo": LARGO}, archivo)
    return ruta_modelo, ruta_vocabulario, exactitud, modelo, x


def test_el_transformer_aprende(modelo_entrenado):
    assert modelo_entrenado[2] > 0.95


def test_guardar_y_cargar_da_las_mismas_predicciones(modelo_entrenado):
    ruta_modelo, _, _, modelo, x = modelo_entrenado
    cargado = keras.models.load_model(ruta_modelo)
    antes = np.asarray(modelo(x[:20], training=False))
    despues = np.asarray(cargado(x[:20], training=False))
    assert np.allclose(antes, despues, atol=1e-5)


def test_el_relleno_no_cambia_la_prediccion(modelo_entrenado):
    """La máscara de atención debe ignorar las posiciones de relleno."""
    _, ruta_vocabulario, _, modelo, _ = modelo_entrenado
    with open(ruta_vocabulario, encoding="utf-8") as archivo:
        vocabulario = json.load(archivo)["vocabulario"]
    corta = np.array([codificar("cuantas motos hubo", vocabulario, LARGO)], dtype="int32")
    salida = np.asarray(modelo(corta, training=False))[0]
    assert salida.argmax() == 1                 # contar_motos
    assert abs(salida.sum() - 1.0) < 1e-5       # softmax: las probabilidades suman 1


def test_el_asistente_responde_con_el_transformer(modelo_entrenado):
    ruta_modelo, ruta_vocabulario, *_ = modelo_entrenado
    asistente = Asistente(RESUMEN, EVENTOS, ruta_modelo=ruta_modelo,
                          ruta_vocabulario=ruta_vocabulario, usar_llm=False)
    r = asistente.responder("¿Cuántas colisiones hubo?")
    assert r["fuente"] == "transformer" and r["intencion"] == "contar_colisiones"
    assert "1 colisión" in r["respuesta"]

    r = asistente.responder("zzz qqq")          # ninguna palabra conocida
    assert r["fuente"] == "reglas"
