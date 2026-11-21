"""
Pruebas de punta a punta de la detección sin modelos preentrenados:
sustracción de fondo + seguidor propio + reglas, sobre escenas sintéticas.
La CNN se reemplaza por un clasificador fijo (la red se prueba aparte).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import Detector  # noqa: E402
from src.eventos import AnalizadorEventos  # noqa: E402
from src.seguimiento import SeguidorObjetos, iou  # noqa: E402
from src.movimiento import DetectorMovimiento  # noqa: E402
from tests.escenas import FPS, ClasificadorFijo, fondo, generar  # noqa: E402

AZUL, ROJO, VERDE = (160, 60, 30), (40, 40, 190), (60, 150, 60)


def analizar(escena, frames):
    detector = Detector(fps=FPS, clasificador=ClasificadorFijo())
    analizador = AnalizadorEventos(fps=FPS)
    vistos = []
    for indice, frame in enumerate(generar(escena, frames)):
        detecciones = detector.detectar(frame)
        analizador.actualizar(indice, detecciones)
        vistos.append(detecciones)
    return analizador, vistos


def tipos(analizador):
    return [e.tipo for e in analizador.eventos]


# ------------------------------------------------------------------ seguidor
def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_seguidor_conserva_el_id_y_detecta_la_fusion():
    seguidor = SeguidorObjetos(fps=FPS)
    for f in range(20):                         # dos objetos que se acercan
        salida = seguidor.actualizar([(100 + 5 * f, 100, 150 + 5 * f, 140),
                                      (400 - 5 * f, 100, 450 - 5 * f, 140)])
    assert sorted(s.id for s in salida) == [1, 2]

    salida = seguidor.actualizar([(200, 100, 350, 140)])    # una sola mancha grande
    ocultos = [s for s in salida if s.oculta_en is not None]
    assert len(salida) == 2 and len(ocultos) == 1
    assert ocultos[0].estimada


def test_un_objeto_que_cruza_la_escena_conserva_su_id():
    def escena(f):
        return [(60 + 6 * f, 250, 90, 50, AZUL)]

    analizador, vistos = analizar(escena, 80)
    ids = {d.id for detecciones in vistos for d in detecciones}
    assert len(ids) == 1
    assert analizador.resumen()["objetos"] == {"vehiculo": 1}
    assert tipos(analizador) == []


# ----------------------------------------------------------------- colisión
def test_choque_frontal_los_dos_quedan_detenidos():
    """Dos carros se acercan de frente, chocan y quedan quietos."""
    def escena(f):
        x_a = min(80 + 5 * f, 285)
        x_b = max(560 - 5 * f, 365)             # quedan tocándose en x = 325
        return [(x_a, 250, 90, 50, AZUL), (x_b, 250, 90, 50, ROJO)]

    analizador, _ = analizar(escena, 110)
    assert tipos(analizador).count("colision") == 1
    assert "riesgo" in tipos(analizador)
    assert tipos(analizador).index("riesgo") < tipos(analizador).index("colision")


def test_alcance_por_detras():
    """Un carro rápido alcanza a uno lento; tras el golpe quedan quietos."""
    def escena(f):
        contacto = 60
        x_lento = 200 + 2 * min(f, contacto)
        x_rapido = min(-40 + 5 * f, x_lento - 78)
        return [(x_rapido, 250, 90, 50, AZUL), (x_lento, 250, 90, 50, ROJO)]

    analizador, _ = analizar(escena, 110)
    assert "colision" in tipos(analizador)


# ------------------------------------------------ casos que NO son colisión
def test_cruce_en_carriles_distintos_no_es_colision():
    """Sentidos opuestos, carriles contiguos: las manchas se funden al cruzarse."""
    def escena(f):
        return [(60 + 5 * f, 250, 90, 50, AZUL),       # carril cercano
                (580 - 5 * f, 215, 80, 44, ROJO)]      # carril lejano (base más arriba)

    analizador, _ = analizar(escena, 110)
    assert "colision" not in tipos(analizador)


def test_dos_carros_en_fila_no_generan_alertas():
    def escena(f):
        return [(60 + 5 * f, 250, 90, 50, AZUL), (220 + 5 * f, 250, 90, 50, ROJO)]

    analizador, _ = analizar(escena, 70)
    assert tipos(analizador) == []


def test_carro_que_frena_antes_del_peaton_es_riesgo_sin_colision():
    def escena(f):
        # avanza a 6 px por fotograma y frena a fondo durante 0,9 s: queda en x = 313
        t = min(max(f - 72, 0), 27)
        x_carro = -200 + 6 * min(f, 72) + 6 * t - 6 * t * t / 54
        x_peaton = min(330 + 1.5 * f, 420)              # camina y se queda en la vía
        return [(x_carro, 250, 90, 50, AZUL), (x_peaton, 250, 22, 60, VERDE)]

    analizador, _ = analizar(escena, 120)
    assert "riesgo" in tipos(analizador)
    assert "colision" not in tipos(analizador)


# ----------------------------------------------------- cambios de toda la imagen
def test_un_cambio_de_exposicion_no_se_toma_por_movimiento():
    """La cámara aclara toda la imagen un 15 % mientras pasa un objeto."""
    detector = DetectorMovimiento(fps=FPS)

    def escena(f):
        return [(60 + 5 * f, 250, 90, 50, AZUL)]

    por_frame = []
    for f, frame in enumerate(generar(escena, 90)):
        if f >= 75:                                     # 35 de fondo + 40 con el objeto
            frame = np.clip(frame.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
        por_frame.append(detector.detectar(frame))
    # El objeto se sigue viendo como una sola mancha antes y después del cambio
    assert all(len(cajas) == 1 for cajas in por_frame[40:])


def test_si_la_camara_se_mueve_el_fondo_se_aprende_de_nuevo():
    """La escena cambia por completo; a los pocos segundos vuelve a detectar bien."""
    detector = DetectorMovimiento(fps=FPS)
    rng = np.random.default_rng(3)
    escena_1 = fondo()
    escena_2 = rng.integers(0, 255, size=(9, 16, 3)).astype(np.uint8)     # manchas grandes
    escena_2 = np.kron(escena_2, np.ones((40, 40, 1), dtype=np.uint8))
    for _ in range(45):
        detector.detectar(escena_1)
    por_frame = []
    for f in range(120):
        frame = escena_2.copy()
        if f >= 60:                                     # pasa un objeto por la escena nueva
            x = 100 + 6 * (f - 60)
            frame[200:250, x:x + 90] = AZUL
        por_frame.append(detector.detectar(frame))
    assert all(len(c) == 0 for c in por_frame[20:60])   # la escena nueva ya es el fondo
    assert all(len(c) == 1 for c in por_frame[70:])     # y el objeto se detecta solo
