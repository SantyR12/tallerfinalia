"""
Escenas sintéticas para las pruebas: rectángulos con textura que se mueven
sobre un fondo fijo. Sirven para probar de punta a punta la detección de
movimiento, el seguimiento y las reglas, sin depender de un video real.
"""

import numpy as np

ANCHO, ALTO, FPS = 640, 360, 30


def fondo():
    """Imagen de la vía vacía: gris con un poco de grano, como el asfalto."""
    rng = np.random.default_rng(0)
    base = np.full((ALTO, ANCHO, 3), 95, dtype=np.int16)
    base += rng.integers(-6, 7, size=base.shape, dtype=np.int16)     # asfalto con grano
    return np.clip(base, 0, 255).astype(np.uint8)


def dibujar_objeto(frame, x_centro, base, ancho, alto, color):
    """Pinta un rectángulo de color (el 'vehículo' o la 'persona') apoyado en `base`."""
    x1, x2 = int(x_centro - ancho / 2), int(x_centro + ancho / 2)
    y1, y2 = int(base - alto), int(base)
    x1, x2 = max(x1, 0), min(x2, ANCHO)
    if x2 <= x1:
        return
    frame[y1:y2, x1:x2] = color
    frame[y1 + 4:y1 + 10, x1 + 4:max(x2 - 4, x1 + 5)] = (230, 230, 230)   # "ventana"


def generar(escena, frames, calentamiento=35):
    """
    escena(f) -> lista de (x_centro, base, ancho, alto, color).
    Los primeros fotogramas son solo fondo, para que el sustractor lo aprenda
    (el detector dedica a eso el primer segundo y no entrega detecciones).
    """
    base = fondo()
    for _ in range(calentamiento):
        yield base.copy()
    for f in range(frames):
        frame = base.copy()
        for objeto in escena(f):
            dibujar_objeto(frame, *objeto)
        yield frame


class ClasificadorFijo:
    """Sustituye a la CNN en las pruebas: clasifica por el tamaño de la caja."""

    def clasificar(self, frame, cajas):
        """Sin red neuronal: caja alta = persona, caja pequeña = motocicleta, el resto = vehículo."""
        resultado = []
        for x1, y1, x2, y2 in cajas:
            ancho, alto = x2 - x1, y2 - y1
            if alto > 1.5 * ancho:
                resultado.append(("persona", 1.0))
            elif ancho * alto < 2600:
                resultado.append(("motocicleta", 1.0))
            else:
                resultado.append(("vehiculo", 1.0))
        return resultado
