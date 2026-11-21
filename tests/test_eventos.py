"""
Pruebas de las reglas de eventos con trayectorias inventadas.

No necesitan video ni red neuronal: se construyen cajas a mano, fotograma a
fotograma, y se comprueba que el analizador dé (o no dé) la alerta esperada.

    python -m pytest tests -q
"""

import os
import sys
from collections import namedtuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eventos import AnalizadorEventos, solape  # noqa: E402

Det = namedtuple("Det", "id clase caja")
FPS = 30


def caja(x_centro, base, ancho, alto):
    """Caja a partir del centro horizontal y de la base (donde toca el suelo)."""
    return (x_centro - ancho / 2, base - alto, x_centro + ancho / 2, base)


def simular(escena, frames):
    """escena(frame) -> lista de Det. Devuelve el analizador ya alimentado."""
    analizador = AnalizadorEventos(fps=FPS)
    for f in range(frames):
        analizador.actualizar(f, escena(f))
    return analizador


def tipos(analizador):
    return [e.tipo for e in analizador.eventos]


# ------------------------------------------------------------------ colisión
def test_colision_carro_contra_moto_detenida():
    """Un carro avanza, alcanza una moto quieta y se detiene de golpe."""
    def escena(f):
        x_carro = min(100 + 6 * f, 345)         # se frena en seco al llegar a x=345
        return [
            Det(1, "vehiculo", caja(x_carro, 300, 100, 60)),
            Det(2, "motocicleta", caja(400, 300, 50, 60)),
        ]

    analizador = simular(escena, 90)
    assert tipos(analizador).count("colision") == 1
    colision = next(e for e in analizador.eventos if e.tipo == "colision")
    assert set(colision.ids) == {1, 2}
    assert colision.nivel == "ALTO"
    # antes del choque debió avisar del riesgo
    assert tipos(analizador).index("riesgo") < tipos(analizador).index("colision")


def test_colision_el_objeto_golpeado_sale_impulsado():
    """El carro no frena, pero la moto quieta sale despedida tras el contacto."""
    def escena(f):
        x_carro = 100 + 6 * f
        contacto = 40                            # fotograma aproximado del contacto
        x_moto = 400 if f < contacto else 400 + 9 * (f - contacto)
        return [
            Det(1, "vehiculo", caja(x_carro, 300, 100, 60)),
            Det(2, "motocicleta", caja(x_moto, 300, 50, 60)),
        ]

    analizador = simular(escena, 60)
    assert "colision" in tipos(analizador)


# ------------------------------------------------- casos que NO son colisión
def test_cruce_a_distinta_profundidad_no_es_colision():
    """Un carro pasa por delante de otro que está parqueado más lejos."""
    def escena(f):
        return [
            Det(1, "vehiculo", caja(100 + 6 * f, 300, 100, 60)),   # cerca de la cámara
            Det(2, "vehiculo", caja(400, 270, 80, 48)),            # más lejos (base más arriba)
        ]

    analizador = simular(escena, 110)
    assert solape(caja(400, 300, 100, 60), caja(400, 270, 80, 48)) > 0.5   # sí se tapan
    assert tipos(analizador) == []


def test_adelantamiento_sin_frenar_no_es_colision():
    """Dos carros se solapan en la imagen, pero ninguno cambia su movimiento."""
    def escena(f):
        return [
            Det(1, "vehiculo", caja(100 + 6 * f, 300, 100, 60)),
            Det(2, "vehiculo", caja(300 + 2 * f, 305, 100, 60)),
        ]

    analizador = simular(escena, 120)
    assert "colision" not in tipos(analizador)


def test_motociclista_sobre_su_moto_no_es_colision():
    """La persona y la moto siempre se solapan, pero se mueven juntas."""
    def escena(f):
        x = 100 + 5 * f
        return [
            Det(1, "motocicleta", caja(x, 300, 60, 50)),
            Det(2, "persona", caja(x, 290, 30, 70)),
        ]

    analizador = simular(escena, 90)
    assert tipos(analizador) == []


def test_objeto_que_aparece_ya_en_contacto_no_es_colision():
    """Si el seguimiento le cambia el ID a un objeto, el nuevo no tiene historial."""
    def escena(f):
        objetos = [Det(1, "vehiculo", caja(100 + 6 * f, 300, 100, 60))]
        if f >= 30:                              # aparece de repente, pegado al carro
            objetos.append(Det(9, "motocicleta", caja(160 + 6 * f, 300, 50, 60)))
        return objetos

    analizador = simular(escena, 90)
    assert "colision" not in tipos(analizador)


def test_trafico_en_el_mismo_sentido_no_genera_alertas():
    def escena(f):
        return [
            Det(1, "vehiculo", caja(100 + 5 * f, 300, 100, 60)),
            Det(2, "vehiculo", caja(300 + 5 * f, 300, 100, 60)),
            Det(3, "vehiculo", caja(600 + 5 * f, 250, 160, 90)),
        ]

    analizador = simular(escena, 90)
    assert tipos(analizador) == []


# -------------------------------------------------------------------- riesgo
def frenando(x0, v, f_freno, frames_frenado):
    """Posición de un carro que avanza a v px/fotograma y luego frena de forma pareja."""
    def x(f):
        if f <= f_freno:
            return x0 + v * f
        t = min(f - f_freno, frames_frenado)
        return x0 + v * f_freno + v * t - v * t * t / (2 * frames_frenado)
    return x


def test_riesgo_cuando_un_carro_va_hacia_un_peaton_y_frena_antes():
    x_de = frenando(100, 6, 28, 30)             # frena durante 1 s y queda en x = 358

    def escena(f):
        return [                                # se detiene antes de tocar al peatón
            Det(1, "vehiculo", caja(x_de(f), 300, 100, 60)),
            Det(2, "persona", caja(430, 300, 30, 80)),
        ]

    analizador = simular(escena, 90)
    assert "riesgo" in tipos(analizador)
    assert "colision" not in tipos(analizador)
    riesgo = next(e for e in analizador.eventos if e.tipo == "riesgo")
    assert riesgo.nivel == "MEDIO"


def test_no_hay_eventos_con_un_objeto_cortado_por_el_borde():
    """
    Un carro sigue a otro que va saliendo de la imagen. La caja del que sale se
    encoge contra el borde y su centro parece frenar; no es un acercamiento real
    (sin la regla del borde, esta escena daba un riesgo y una colisión falsos).
    """
    DetBorde = namedtuple("DetBorde", "id clase caja en_borde")

    def escena(f):
        base_1 = min(200 + 4 * f, 360)                  # llega al borde inferior (360)
        techo_1 = 140 + 4 * f                           # y su caja se va encogiendo
        base_2 = min(130 + 4 * f, 360)                  # el de atrás va a la misma velocidad
        return [
            DetBorde(1, "vehiculo", (270, min(techo_1, 350), 370, base_1), base_1 >= 358),
            DetBorde(2, "vehiculo", (270, base_2 - 60, 370, base_2), base_2 >= 358),
        ]

    analizador = simular(escena, 70)
    assert tipos(analizador) == []


def test_dos_carros_que_se_alejan_por_la_misma_via_no_son_riesgo():
    """
    Vista en perspectiva: dos carros se alejan de la cámara con la misma
    separación real. En la imagen el de atrás "alcanza" al de adelante porque
    todo se encoge hacia el horizonte, pero no se están acercando.
    """
    def carro(z):                                   # z = distancia a la cámara
        ancho, alto, base = 900 / z, 600 / z, 50 + 3000 / z
        return (320 - ancho / 2, base - alto, 320 + ancho / 2, base)

    def escena(f):
        z = 10 + 0.25 * f
        return [Det(1, "vehiculo", carro(z)), Det(2, "vehiculo", carro(z + 6))]

    analizador = simular(escena, 90)
    assert tipos(analizador) == []


def test_frenada_imposible_de_un_solo_vehiculo_es_colision():
    """Choque contra un poste: venía rápido y queda quieto en un instante."""
    def escena(f):
        return [Det(1, "vehiculo", caja(min(100 + 6 * f, 400), 300, 100, 60))]

    analizador = simular(escena, 110)
    assert tipos(analizador) == ["colision"]
    assert "se detuvo de golpe" in analizador.eventos[0].descripcion


def test_una_frenada_normal_no_es_colision():
    """El mismo carro, pero frenando durante un segundo."""
    x_de = frenando(100, 6, 50, 30)

    def escena(f):
        return [Det(1, "vehiculo", caja(x_de(f), 300, 100, 60))]

    assert tipos(simular(escena, 130)) == []


def test_un_salto_de_la_caja_no_es_una_frenada_imposible():
    """Un carro lento cuya caja da un salto (ruido del seguimiento) y luego se detiene."""
    def escena(f):
        x = 100 + 1.0 * min(f, 60) + (40 if f >= 55 else 0)
        return [Det(1, "vehiculo", caja(x, 300, 100, 60))]

    assert tipos(simular(escena, 120)) == []


def test_carro_detenido_que_sale_empujado_es_colision():
    """Alcance por detrás: las manchas se funden y el que estaba quieto avanza."""
    DetFundida = namedtuple("DetFundida", "id clase caja estimada oculta_en")

    def escena(f):
        if f < 40:                                  # el 2 se acerca al 1, que está quieto
            return [DetFundida(1, "vehiculo", caja(400, 300, 100, 60), False, None),
                    DetFundida(2, "vehiculo", caja(60 + 6 * f, 300, 100, 60), False, None)]
        x = 350 + 3 * (f - 40)                      # una sola mancha, que avanza
        return [DetFundida(1, "vehiculo", caja(x, 300, 200, 60), False, None),
                DetFundida(2, "vehiculo", caja(x - 50, 300, 100, 60), True, 1)]

    analizador = simular(escena, 90)
    assert "colision" in tipos(analizador)
    colision = next(e for e in analizador.eventos if e.tipo == "colision")
    assert set(colision.ids) == {1, 2} and "salió impulsado" in colision.descripcion


def test_moto_que_alcanza_a_un_carro_lento_no_es_colision():
    """La moto se funde con el carro al pasarlo, pero el carro sigue a su ritmo."""
    DetFundida = namedtuple("DetFundida", "id clase caja estimada oculta_en")

    def escena(f):
        x_carro = 300 + 2 * f
        if f < 40:
            return [DetFundida(1, "vehiculo", caja(x_carro, 300, 100, 60), False, None),
                    DetFundida(2, "motocicleta", caja(60 + 7 * f, 300, 40, 50), False, None)]
        return [DetFundida(1, "vehiculo", caja(x_carro, 300, 110, 60), False, None),
                DetFundida(2, "motocicleta", caja(x_carro - 30, 300, 40, 50), True, 1)]

    assert "colision" not in tipos(simular(escena, 100))


# --------------------------------------------------------------------- caída
def test_caida_de_peaton():
    """La caja pasa de 40x100 (de pie) a 100x40 (en el suelo) en 10 fotogramas."""
    def escena(f):
        avance = min(max((f - 30) / 10, 0), 1)   # 0 = de pie, 1 = en el suelo
        ancho = 40 + 60 * avance
        alto = 100 - 60 * avance
        return [Det(7, "persona", caja(300, 400, ancho, alto))]

    analizador = simular(escena, 90)
    assert tipos(analizador) == ["caida"]        # una sola alerta, no una por fotograma
    assert "peatón" in analizador.eventos[0].descripcion


def test_caida_de_motociclista():
    def escena(f):
        avance = min(max((f - 30) / 10, 0), 1)
        return [
            Det(1, "motocicleta", caja(300, 400, 70, 55)),
            Det(2, "persona", caja(300, 395, 40 + 60 * avance, 90 - 55 * avance)),
        ]

    analizador = simular(escena, 90)
    assert "caida" in tipos(analizador)
    caida = next(e for e in analizador.eventos if e.tipo == "caida")
    assert "motociclista" in caida.descripcion


def test_persona_a_la_que_se_le_pega_otra_mancha_no_es_caida():
    """La caja se ensancha pero no se acorta: otra mancha se fundió con ella."""
    def escena(f):
        ancho = 40 if f < 30 else 140
        return [Det(7, "persona", caja(300, 400, ancho, 100))]

    assert tipos(simular(escena, 90)) == []


def test_persona_de_pie_no_es_caida():
    def escena(f):
        return [Det(7, "persona", caja(300 + f, 400, 40, 100))]

    assert tipos(simular(escena, 90)) == []


# ------------------------------------------------------------------- resumen
def test_resumen_cuenta_objetos_por_clase():
    def escena(f):
        objetos = [
            Det(1, "vehiculo", caja(100 + 5 * f, 300, 100, 60)),
            Det(2, "persona", caja(700, 300, 30, 80)),
        ]
        if f < 2:                                # aparece 2 fotogramas: es ruido
            objetos.append(Det(3, "vehiculo", caja(900, 300, 150, 90)))
        return objetos

    resumen = simular(escena, 60).resumen()
    assert resumen["objetos"] == {"vehiculo": 1, "persona": 1}
    assert resumen["frames"] == 60
    assert resumen["duracion_s"] == 2.0
    assert resumen["eventos"] == {"colision": 0, "riesgo": 0, "caida": 0}
