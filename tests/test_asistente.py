"""
Pruebas del asistente sin redes: el informe en texto y las respuestas armadas
con la intención deducida por palabras clave. El clasificador se prueba en
test_transformer.py y el modelo de lenguaje en test_llm.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.asistente import Asistente, construir_informe  # noqa: E402

RESUMEN = {
    "frames": 300, "duracion_s": 10.0, "fps": 30.0,
    "objetos": {"vehiculo": 4, "motocicleta": 2, "persona": 3},
    "eventos": {"colision": 1, "riesgo": 1, "caida": 1},
}
EVENTOS = [
    {"tipo": "riesgo", "nivel": "MEDIO", "frame": 90, "tiempo_s": 3.0, "ids": [4, 7],
     "clases": ["vehiculo", "motocicleta"],
     "descripcion": "Riesgo de colisión entre vehículo 4 y motocicleta 7 en el segundo 3.0"},
    {"tipo": "colision", "nivel": "ALTO", "frame": 105, "tiempo_s": 3.5, "ids": [4, 7],
     "clases": ["vehiculo", "motocicleta"],
     "descripcion": "Posible colisión entre vehículo 4 y motocicleta 7 en el segundo 3.5"},
    {"tipo": "caida", "nivel": "ALTO", "frame": 120, "tiempo_s": 4.0, "ids": [8, 7],
     "clases": ["persona", "motocicleta"],
     "descripcion": "Posible caída de motociclista (persona 8) en el segundo 4.0"},
]
SIN_NADA = {"frames": 60, "duracion_s": 2.0, "fps": 30.0, "objetos": {"vehiculo": 2},
            "eventos": {"colision": 0, "riesgo": 0, "caida": 0}}


def test_informe_contiene_los_datos_clave():
    informe = construir_informe(RESUMEN, EVENTOS)
    assert "dura 10.0 segundos" in informe
    assert "9 objetos" in informe
    assert "4 vehículos" in informe and "2 motocicletas" in informe and "3 personas" in informe
    assert "3 eventos" in informe
    assert "1 colisión" in informe and "1 caída" in informe
    assert "La primera colisión ocurrió en el segundo 3.5" in informe
    assert "El último evento ocurrió en el segundo 4.0" in informe


def test_informe_sin_eventos():
    informe = construir_informe(SIN_NADA, [])
    assert "No se registró ningún evento" in informe
    assert "2 vehículos" in informe


def test_reglas_responden_conteos_y_momentos():
    asistente = Asistente(RESUMEN, EVENTOS, usar_transformer=False, usar_llm=False)

    r = asistente.responder("¿Cuántas colisiones hubo?")
    assert r["fuente"] == "reglas" and "1 colisión" in r["respuesta"]

    r = asistente.responder("¿En qué segundo fue el choque?")
    assert "segundo 3.5" in r["respuesta"]

    r = asistente.responder("¿Alguien se cayó?")
    assert "1 caída" in r["respuesta"] and "motociclista" in r["respuesta"]

    r = asistente.responder("¿Cuántas motos pasaron?")
    assert "2 motocicletas" in r["respuesta"]

    r = asistente.responder("¿Cuántos peatones había?")
    assert "3 peatones" in r["respuesta"]

    r = asistente.responder("¿Cuántos carros pasaron?")
    assert "4 vehículos" in r["respuesta"]

    # El sistema no separa carros, buses y camiones, y lo dice
    r = asistente.responder("¿Cuántos buses pasaron?")
    assert r["intencion"] == "tipo_vehiculo"
    assert "no distingue el tipo de vehículo" in r["respuesta"] and "4 vehículos" in r["respuesta"]

    r = asistente.responder("¿Cuáles son los eventos más graves?")
    assert "2 eventos de nivel alto" in r["respuesta"]


def test_reglas_cuando_no_hubo_eventos():
    asistente = Asistente(SIN_NADA, [], usar_transformer=False, usar_llm=False)
    assert "No se detectó ninguna colisión" in asistente.responder("¿Hubo algún accidente?")["respuesta"]
    assert "No se registró ningún evento" in asistente.responder("¿Cuántas alertas hubo?")["respuesta"]


def test_si_el_modelo_no_esta_entrenado_responde_por_reglas(tmp_path):
    falta = str(tmp_path / "no_existe")
    asistente = Asistente(RESUMEN, EVENTOS, ruta_modelo=falta, ruta_vocabulario=falta,
                          ruta_llm=falta, ruta_vocabulario_llm=falta)
    r = asistente.responder("¿Cuántas colisiones hubo?")
    assert r["fuente"] == "reglas" and "1 colisión" in r["respuesta"]
    assert "entrenar_asistente.py" in asistente.error_modelo
    assert "entrenar_llm.py" in asistente.error_llm
