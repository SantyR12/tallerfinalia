"""
Detección completa = movimiento + clasificación + seguimiento.

    fotograma
        │
        ├─ DetectorMovimiento   →  cajas de lo que se mueve        (OpenCV clásico)
        ├─ SeguidorObjetos      →  cada caja conserva su ID        (código propio)
        └─ ClasificadorCNN      →  qué es cada objeto              (CNN propia)

La clase de un objeto no se decide con un solo fotograma: cada fotograma en
que la CNN está segura aporta un voto y gana la mayoría. Los objetos cuya
clase mayoritaria es "fondo" se descartan.
"""

from collections import Counter
from dataclasses import dataclass

import config
from src.movimiento import DetectorMovimiento
from src.seguimiento import SeguidorObjetos


@dataclass
class Deteccion:
    """Un objeto detectado en un fotograma."""

    id: int                     # ID de seguimiento
    clase: str                  # "persona", "carro", "motocicleta", ...
    caja: tuple                 # (x1, y1, x2, y2) en píxeles
    confianza: float = 1.0      # fracción de votos que obtuvo la clase
    estimada: bool = False      # la caja es una predicción (objeto oculto)
    oculta_en: int = None       # ID del objeto con el que se fundió
    en_borde: bool = False      # la caja toca el borde: el objeto se ve a medias


def toca_borde(caja, ancho, alto, margen=config.MARGEN_BORDE):
    """True si la caja llega al borde de la imagen (objeto entrando o saliendo)."""
    x1, y1, x2, y2 = caja
    return x1 <= margen or y1 <= margen or x2 >= ancho - margen or y2 >= alto - margen


class Detector:
    def __init__(self, fps=30.0, clasificador=None):
        self.movimiento = DetectorMovimiento(fps=fps)
        self.seguidor = SeguidorObjetos(fps=fps)
        if clasificador is None:
            from src.clasificador import ClasificadorCNN
            clasificador = ClasificadorCNN()
        self.clasificador = clasificador
        self.votos = {}             # id -> Counter de clases
        self.fps = fps
        self.frame = 0
        self.protegidas = []        # cajas donde el fondo no se actualiza
        self.ultimo_movimiento = {} # id -> (posición, fotograma) de la última vez que se movió

    def reiniciar(self, fps=30.0):
        """Se llama al empezar un video nuevo."""
        self.movimiento = DetectorMovimiento(fps=fps)
        self.seguidor = SeguidorObjetos(fps=fps)
        self.votos = {}
        self.fps = fps
        self.frame = 0
        self.protegidas = []
        self.ultimo_movimiento = {}

    def detectar(self, frame):
        cajas = self.movimiento.detectar(frame, proteger=self.protegidas)
        seguidos = self.seguidor.actualizar(cajas)

        # Se clasifican solo los objetos que se ven solos y completos: una
        # mancha que agrupa a dos objetos fundidos, o un objeto cortado por el
        # borde de la imagen, confundiría a la red.
        alto, ancho = frame.shape[:2]
        en_borde = {s.id for s in seguidos if not s.estimada and toca_borde(s.caja, ancho, alto)}
        con_ocultos = {s.oculta_en for s in seguidos if s.oculta_en is not None}

        def sin_clase(seguido):                 # todavía no reunió los votos mínimos
            return sum(self.votos.get(seguido.id, {}).values()) < config.VOTOS_PARA_MOSTRAR

        # (Una mancha con objetos fundidos sí se clasifica si aún no tiene clase:
        # de lo contrario nunca se mostraría y el choque quedaría sin testigo.)
        a_clasificar = [s for s in seguidos
                        if not s.estimada and s.id not in en_borde
                        and (s.id not in con_ocultos or sin_clase(s))]
        resultados = self.clasificador.clasificar(frame, [s.caja for s in a_clasificar])
        for seguido, (clase, confianza) in zip(a_clasificar, resultados):
            if confianza >= config.CONFIANZA_MINIMA:
                self.votos.setdefault(seguido.id, Counter())[clase] += 1

        detecciones = []
        for s in seguidos:
            votos = self.votos.get(s.id)
            if not votos or sum(votos.values()) < config.VOTOS_PARA_MOSTRAR:
                continue                        # todavía no se sabe qué es
            clase, cuenta = votos.most_common(1)[0]
            if clase == "fondo":
                continue                        # sombra, reflejo, vegetación...
            detecciones.append(Deteccion(
                id=s.id, clase=clase, caja=s.caja,
                confianza=cuenta / sum(votos.values()),
                estimada=s.estimada, oculta_en=s.oculta_en, en_borde=s.id in en_borde,
            ))
        self.protegidas = self._cajas_protegidas(detecciones)
        return detecciones

    def _cajas_protegidas(self, detecciones):
        """
        Opción apagada por defecto (PROTECCION_S = 0). Si se enciende, los
        vehículos reconocidos y visibles quedan protegidos: el fondo no se
        actualiza dentro de sus cajas en el fotograma siguiente, de modo que
        el carro que espera en un cruce no se vuelva fondo antes de recibir
        el golpe. La protección dura PROTECCION_S desde la última vez que el
        vehículo se movió, para que un error del detector no se quede en
        pantalla para siempre.

        Sirve en cruces con poco tráfico. En un trancón hace daño: toda la fila
        detenida sigue a la vista y cada carro que frena detrás del último se
        confunde con un choque por alcance. Por eso viene apagada.
        """
        self.frame += 1
        if config.PROTECCION_S <= 0:
            return []
        cajas = []
        for d in detecciones:
            if d.estimada or d.clase not in config.VEHICULOS_MOTOR:
                continue
            x1, y1, x2, y2 = d.caja
            centro = ((x1 + x2) / 2, (y1 + y2) / 2)
            tam = max(x2 - x1, y2 - y1)
            referencia, desde = self.ultimo_movimiento.get(d.id, (centro, self.frame))
            if abs(centro[0] - referencia[0]) + abs(centro[1] - referencia[1]) > 0.25 * tam:
                referencia, desde = centro, self.frame          # se movió
            self.ultimo_movimiento[d.id] = (referencia, desde)
            if self.frame - desde <= config.PROTECCION_S * self.fps:
                cajas.append(d.caja)
        return cajas
