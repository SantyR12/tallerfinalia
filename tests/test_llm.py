"""
Pruebas del modelo de lenguaje propio (Transformer decodificador).

Comprueban el "idioma" del modelo (datos, respuestas válidas, ranuras) y
entrenan durante unos segundos un modelo pequeño con dos intenciones para
verificar que aprende a escribir respuestas, que respeta la máscara causal,
que se guarda y se carga, y que el asistente lo usa con su control.
"""

import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lenguaje import (CANTIDADES, FIN, RESPUESTA, armar_entrada, contar, cuenta_al_azar,  # noqa: E402
                          fichas_de_datos, rellenar, respuestas_validas)
from tests.test_asistente import EVENTOS, RESUMEN, SIN_NADA  # noqa: E402

INTENCIONES = ["contar_colisiones", "duracion"]


# -------------------------------------------------- el idioma (sin redes)
def test_contar_y_fichas_de_datos():
    cuenta = contar(RESUMEN, EVENTOS)
    assert cuenta["colisiones"] == 1 and cuenta["caidas"] == 1 and cuenta["riesgos"] == 1
    assert cuenta["vehiculos"] == 4 and cuenta["peatones"] == 3 and cuenta["motos"] == 2
    assert cuenta["objetos"] == 9 and cuenta["eventos"] == 3 and cuenta["graves"] == 2
    fichas = fichas_de_datos(cuenta)
    assert len(fichas) == len(CANTIDADES)
    assert "colisiones_uno" in fichas and "vehiculos_varios" in fichas and "bicicletas_cero" in fichas


def test_respuestas_validas_dependen_de_los_datos():
    cuenta = contar(RESUMEN, EVENTOS)
    assert all("<n_colisiones>" in r for r in respuestas_validas("contar_colisiones", cuenta))
    sin_nada = contar(SIN_NADA, [])
    assert all("<" not in r for r in respuestas_validas("contar_colisiones", sin_nada))
    assert "no hubo colisiones" in respuestas_validas("contar_colisiones", sin_nada)
    # por buses o camiones: el sistema dice que no separa los tipos y da el total
    assert all("no" in r and "<n_vehiculos>" in r for r in respuestas_validas("tipo_vehiculo", cuenta))
    # todas las intenciones tienen respuesta para cualquier video
    from src.asistente import INTENCIONES as TODAS
    rng = random.Random(0)
    for _ in range(50):
        azar = cuenta_al_azar(rng)
        for intencion in TODAS:
            assert respuestas_validas(intencion, azar)


def test_rellenar_pone_los_datos_reales():
    texto = rellenar("se detectó <n_colisiones> colisión : <lista_colisiones>", RESUMEN, EVENTOS)
    assert texto == ("Se detectó 1 colisión: Posible colisión entre vehículo 4 y motocicleta 7 "
                     "en el segundo 3.5.")
    texto = rellenar("el video dura <duracion> segundos . se detectaron <n_objetos> objetos : "
                     "<lista_objetos> .", RESUMEN, EVENTOS)
    assert texto == ("El video dura 10.0 segundos. Se detectaron 9 objetos: 4 vehículos, "
                     "3 personas y 2 motocicletas.")
    assert "<" not in rellenar("las caídas ocurrieron en : <t_caidas>", RESUMEN, EVENTOS)


# ------------------------------------------------------------- la red
@pytest.fixture(scope="module")
def modelo_entrenado(tmp_path_factory):
    keras = pytest.importorskip("keras")
    np = pytest.importorskip("numpy")
    from entrenamiento.generar_preguntas import generar
    from src.lenguaje import DATOS, PREGUNTA, fichas_de_pregunta
    from src.modelo_lenguaje import ModeloLenguaje, construir_modelo_lenguaje

    keras.utils.set_random_seed(5)
    rng = random.Random(5)
    todas = generar()
    preguntas = {i: rng.sample([p for p, intencion in todas if intencion == i], 300)
                 for i in INTENCIONES}
    largo = 48
    secuencias = []
    for _ in range(4000):
        intencion = rng.choice(INTENCIONES)
        cuenta = cuenta_al_azar(rng)
        pregunta = fichas_de_pregunta(rng.choice(preguntas[intencion]), 16)
        respuesta = rng.choice(respuestas_validas(intencion, cuenta)).split()
        secuencias.append([DATOS] + fichas_de_datos(cuenta) + [PREGUNTA] + pregunta
                          + [RESPUESTA] + respuesta + [FIN])
    vocabulario = {f: n for n, f in enumerate(sorted({f for s in secuencias for f in s}), start=2)}
    x = np.zeros((len(secuencias), largo), dtype="int32")
    y = np.zeros_like(x)
    peso = np.zeros((len(secuencias), largo), dtype="float32")
    for i, fichas in enumerate(secuencias):
        indices = [vocabulario[f] for f in fichas]
        x[i, :len(indices)] = indices
        y[i, :len(indices) - 1] = indices[1:]
        peso[i, len(CANTIDADES) + 1:len(indices) - 1] = 1.0

    modelo = construir_modelo_lenguaje(len(vocabulario) + 2, largo=largo, dim=48, cabezas=4, bloques=2)
    modelo.compile(optimizer=keras.optimizers.Adam(3e-3), loss="sparse_categorical_crossentropy")
    modelo.fit(x, y, sample_weight=peso, epochs=14, batch_size=64, verbose=0)

    carpeta = tmp_path_factory.mktemp("llm")
    ruta_modelo, ruta_vocabulario = str(carpeta / "llm.keras"), str(carpeta / "vocab.json")
    modelo.save(ruta_modelo)
    with open(ruta_vocabulario, "w", encoding="utf-8") as archivo:
        json.dump({"vocabulario": vocabulario, "largo": largo}, archivo)
    return ModeloLenguaje(ruta_modelo, ruta_vocabulario), ruta_modelo, ruta_vocabulario, modelo, x, np


def test_el_modelo_escribe_respuestas_correctas(modelo_entrenado):
    generador = modelo_entrenado[0]
    rng = random.Random(77)
    correctas = 0
    for _ in range(40):
        intencion = rng.choice(INTENCIONES)
        cuenta = cuenta_al_azar(rng)
        pregunta = "cuantas colisiones hubo" if intencion == "contar_colisiones" else "cuanto dura el video"
        fichas, termino = generador.generar(armar_entrada(cuenta, pregunta, 16))
        correctas += termino and " ".join(fichas) in respuestas_validas(intencion, cuenta)
    assert correctas >= 36              # al menos el 90 %


def test_la_mascara_causal_impide_ver_el_futuro(modelo_entrenado):
    """Cambiar fichas posteriores no debe alterar la predicción de una posición anterior."""
    *_, modelo, x, np = modelo_entrenado
    original = x[:1].copy()
    alterada = original.copy()
    alterada[0, 20:] = 5                # se cambia todo lo que viene después de la posición 19
    antes = np.asarray(modelo(original, training=False))[0, :20]
    despues = np.asarray(modelo(alterada, training=False))[0, :20]
    assert np.allclose(antes, despues, atol=1e-5)


def test_la_salida_es_una_distribucion_de_probabilidad(modelo_entrenado):
    *_, modelo, x, np = modelo_entrenado
    salida = np.asarray(modelo(x[:2], training=False))
    assert salida.shape[:2] == x[:2].shape
    assert np.allclose(salida.sum(axis=-1), 1.0, atol=1e-4)


def test_el_asistente_usa_el_modelo_de_lenguaje_con_control(modelo_entrenado, tmp_path):
    from src.asistente import Asistente

    _, ruta_modelo, ruta_vocabulario, *_ = modelo_entrenado
    falta = str(tmp_path / "no_existe")
    asistente = Asistente(RESUMEN, EVENTOS, ruta_modelo=falta, ruta_vocabulario=falta,
                          ruta_llm=ruta_modelo, ruta_vocabulario_llm=ruta_vocabulario)
    r = asistente.responder("¿Cuántas colisiones hubo?")
    assert r["fuente"] == "llm"
    assert "1 colisión" in r["respuesta"] and "segundo 3.5" in r["respuesta"]
    assert "<" not in r["respuesta"]

    # Este modelo de prueba no conoce las motos: lo que escriba no pasa el
    # control y la respuesta sale del respaldo, con el dato correcto.
    r = asistente.responder("¿Cuántas motos pasaron?")
    assert r["fuente"] == "reglas" and "2 motocicletas" in r["respuesta"]
