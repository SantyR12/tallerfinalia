"""
Paso 3 de la detección: seguimiento, escrito desde cero.

El seguidor recibe en cada fotograma las cajas en movimiento y decide cuál
corresponde a cada objeto ya conocido, para que conserve su número (ID).

Cómo funciona:
1. Predice dónde debería estar cada objeto: su caja anterior desplazada con su
   velocidad.
2. Empareja cada predicción con la caja nueva que más se le solapa (IoU).
3. Las cajas sin pareja son objetos nuevos.
4. Un objeto sin pareja se mantiene un tiempo con su posición estimada. Si esa
   posición queda dentro de la caja de otro objeto, es que las dos manchas se
   fundieron en una: se marca como "oculto en" ese objeto. Esa fusión es la
   señal que usan las reglas de colisión.
"""

import math
from dataclasses import dataclass

import config


@dataclass
class Seguido:
    """Un objeto en seguimiento, tal como lo entrega el seguidor en un fotograma."""

    id: int
    caja: tuple                 # (x1, y1, x2, y2)
    estimada: bool = False      # True si la caja es una predicción (no se vio)
    oculta_en: int = None       # ID del objeto con el que se fundió, si aplica


def iou(a, b):
    """Intersección sobre unión de dos cajas: 0 = no se tocan, 1 = idénticas."""
    ancho = min(a[2], b[2]) - max(a[0], b[0])
    alto = min(a[3], b[3]) - max(a[1], b[1])
    if ancho <= 0 or alto <= 0:
        return 0.0
    interseccion = ancho * alto
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - interseccion
    return interseccion / union


def cubierta(interior, exterior):
    """Fracción de la caja `interior` que queda dentro de `exterior`."""
    ancho = min(interior[2], exterior[2]) - max(interior[0], exterior[0])
    alto = min(interior[3], exterior[3]) - max(interior[1], exterior[1])
    if ancho <= 0 or alto <= 0:
        return 0.0
    area = (interior[2] - interior[0]) * (interior[3] - interior[1])
    return (ancho * alto) / max(area, 1.0)


def encajar(caja, contenedor):
    """Desplaza `caja` lo necesario para que quede dentro de `contenedor`."""
    x1, y1, x2, y2 = caja
    dx = max(contenedor[0] - x1, 0) + min(contenedor[2] - x2, 0)
    dy = max(contenedor[1] - y1, 0) + min(contenedor[3] - y2, 0)
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


class _Objeto:
    """Estado interno de un objeto seguido."""

    def __init__(self, id_objeto, caja):
        self.id = id_objeto
        self.caja = caja
        self.velocidad = (0.0, 0.0)     # píxeles por fotograma
        self.perdidos = 0               # fotogramas seguidos sin verlo
        self.oculta_en = None
        self.origen = ((caja[0] + caja[2]) / 2, (caja[1] + caja[3]) / 2)
        self.confirmado = False         # pasa a True cuando de verdad se desplaza

    def predecir(self):
        """Dónde debería estar en el fotograma siguiente: su caja movida con su velocidad."""
        vx, vy = self.velocidad
        x1, y1, x2, y2 = self.caja
        return (x1 + vx, y1 + vy, x2 + vx, y2 + vy)


class SeguidorObjetos:
    """Le pone un número (ID) a cada objeto y se lo conserva de un fotograma a otro."""

    def __init__(self, fps=30.0):
        self.objetos = {}               # id -> _Objeto (todos los que se están siguiendo)
        self.siguiente_id = 1           # número que recibirá el próximo objeto nuevo
        # Fotogramas que se guarda un objeto sin verlo antes de olvidarlo
        self.perdida_maxima = max(2, int(config.PERDIDA_MAXIMA_S * fps))

    def reiniciar(self):
        """Olvida todos los objetos (se usa al empezar otro video)."""
        self.objetos = {}
        self.siguiente_id = 1

    def actualizar(self, cajas):
        """Recibe las cajas del fotograma y devuelve la lista de `Seguido`."""
        # 1. Predicción: dónde se espera ver ahora a cada objeto conocido
        predicciones = {i: o.predecir() for i, o in self.objetos.items()}

        # 2. Emparejamiento: se arman todas las parejas posibles (objeto, caja)
        #    y se ordenan de mayor a menor solape (IoU)
        parejas = sorted(
            (
                (iou(prediccion, caja), id_objeto, indice)
                for id_objeto, prediccion in predicciones.items()
                for indice, caja in enumerate(cajas)
            ),
            reverse=True,
        )
        asignado = {}               # id del objeto -> índice de la caja
        cajas_usadas = set()
        # Se recorren de la mejor pareja a la peor; cada objeto y cada caja se
        # usan una sola vez (emparejamiento "voraz": primero lo más seguro)
        for solape, id_objeto, indice in parejas:
            if solape < config.IOU_EMPAREJAR:
                break                   # de aquí en adelante casi no se solapan: no son el mismo
            if id_objeto in asignado or indice in cajas_usadas:
                continue                # ese objeto o esa caja ya tienen pareja
            asignado[id_objeto] = indice
            cajas_usadas.add(indice)

        # Objetos vistos: se actualiza su caja y su velocidad
        for id_objeto, indice in asignado.items():
            objeto = self.objetos[id_objeto]
            nueva = cajas[indice]
            # Cuánto se movió el centro de la caja desde el fotograma anterior
            dx = ((nueva[0] + nueva[2]) - (objeto.caja[0] + objeto.caja[2])) / 2
            dy = ((nueva[1] + nueva[3]) - (objeto.caja[1] + objeto.caja[3])) / 2
            area_antes = (objeto.caja[2] - objeto.caja[0]) * (objeto.caja[3] - objeto.caja[1])
            area_ahora = (nueva[2] - nueva[0]) * (nueva[3] - nueva[1])
            razon = area_ahora / max(area_antes, 1.0)
            if 1 / config.SALTO_DE_AREA <= razon <= config.SALTO_DE_AREA:
                # Promedio móvil: suaviza el temblor de las cajas
                vx, vy = objeto.velocidad
                objeto.velocidad = (0.6 * vx + 0.4 * dx, 0.6 * vy + 0.4 * dy)
            # Si el área saltó (fusión o separación) el desplazamiento del
            # centro no es movimiento real: se conserva la velocidad anterior.
            objeto.caja = nueva
            objeto.perdidos = 0
            objeto.oculta_en = None
            if not objeto.confirmado:
                # Una mancha que parpadea siempre en el mismo sitio (un reflejo,
                # el "fantasma" de un carro que estaba parqueado) no es un
                # objeto: solo se confirma lo que se desplaza.
                recorrido = math.hypot((nueva[0] + nueva[2]) / 2 - objeto.origen[0],
                                       (nueva[1] + nueva[3]) / 2 - objeto.origen[1])
                tam = math.sqrt(max(area_ahora, 1.0))
                objeto.confirmado = recorrido >= config.RECORRIDO_MINIMO * tam

        # Objetos no vistos: se mantienen con su posición estimada
        for id_objeto, objeto in list(self.objetos.items()):
            if id_objeto in asignado:
                continue
            objeto.perdidos += 1
            objeto.caja = predicciones[id_objeto]
            vx, vy = objeto.velocidad
            objeto.velocidad = (0.9 * vx, 0.9 * vy)     # sin verlo, se confía cada vez menos
            # Si ya estaba oculto en otro objeto, sigue ahí mientras ese otro se
            # vea y él no reaparezca. Si no, se mira si quedó dentro de alguna caja:
            # eso significa que las dos manchas se fundieron en una.
            contenedor = objeto.oculta_en if objeto.oculta_en in asignado else None
            if contenedor is None:
                for otro_id, indice in asignado.items():
                    if cubierta(objeto.caja, cajas[indice]) >= 0.5:
                        contenedor = otro_id
                        break
            objeto.oculta_en = contenedor
            if objeto.perdidos > self.perdida_maxima:
                del self.objetos[id_objeto]             # lleva demasiado sin verse: se olvida

        # Cajas sin pareja: objetos nuevos
        for indice, caja in enumerate(cajas):
            if indice not in cajas_usadas:
                self.objetos[self.siguiente_id] = _Objeto(self.siguiente_id, caja)
                self.siguiente_id += 1

        # Se entregan los objetos vistos y los que están ocultos dentro de otro
        salida = []
        for o in self.objetos.values():
            if not o.confirmado:
                continue
            if o.perdidos == 0:
                salida.append(Seguido(o.id, o.caja))
            elif o.oculta_en in self.objetos and self.objetos[o.oculta_en].confirmado:
                caja = encajar(o.caja, self.objetos[o.oculta_en].caja)
                salida.append(Seguido(o.id, caja, estimada=True, oculta_en=o.oculta_en))
        return salida
