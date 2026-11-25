"""
Recorrido del video: une detección, seguimiento, análisis de eventos y alertas.

procesar() es un generador: entrega un fotograma anotado a la vez. Así lo usan
igual el programa de consola (main.py) y la aplicación web (app.py).
"""

import time
from pathlib import Path

import cv2

import config
from src.alertas import dibujar, guardar_resultados
from src.detector import Detector
from src.eventos import AnalizadorEventos


def procesar(fuente, carpeta_salida=config.CARPETA_SALIDAS, guardar_video=True,
             max_frames=None, detector=None):
    """
    Procesa un video o una cámara.

    fuente: ruta de un archivo de video, o un número de cámara (0 = la principal).
    Entrega en cada paso: (fotograma anotado, eventos nuevos, analizador).
    Al terminar guarda los resultados en `carpeta_salida`.
    """
    # ------------------------------------------------------------ preparación
    es_camara = isinstance(fuente, int)
    captura = cv2.VideoCapture(fuente)
    if not captura.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de video: {fuente}")

    fps = captura.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 240:        # algunas cámaras no informan el fps
        fps = 30.0

    detector = detector or Detector(fps=fps)
    detector.reiniciar(fps=fps)             # fondo y seguimiento empiezan de cero
    analizador = AnalizadorEventos(fps=fps)

    carpeta = Path(carpeta_salida)
    (carpeta / "capturas").mkdir(parents=True, exist_ok=True)
    for vieja in (carpeta / "capturas").glob("*.jpg"):      # capturas del análisis anterior
        vieja.unlink()
    escritor = None
    activos = []                                # eventos que siguen en pantalla
    inicio = time.time()
    indice = -1
    leidos = 0

    # ------------------------------------------------- ciclo: un fotograma por vuelta
    try:
        while True:
            ok, frame = captura.read()
            if not ok:
                break                       # se acabó el video (o la cámara se desconectó)
            leidos += 1

            # Con un archivo, el tiempo es número de fotograma / fps.
            # Con una cámara el procesamiento es más lento que el fps nominal,
            # así que el índice se calcula con el reloj para que las
            # velocidades sigan estando en tiempo real.
            if es_camara:
                indice = max(indice + 1, int((time.time() - inicio) * fps))
            else:
                indice += 1

            detecciones = detector.detectar(frame)              # movimiento + CNN + seguimiento
            nuevos = analizador.actualizar(indice, detecciones) # reglas de eventos

            # Las alertas se quedan en pantalla DURACION_ALERTA_S segundos
            tiempo = indice / fps
            activos = [e for e in activos + nuevos
                       if tiempo - e.tiempo_s <= config.DURACION_ALERTA_S]
            anotado = dibujar(frame, detecciones, activos, analizador.resumen())

            for evento in nuevos:                               # evidencia de cada evento
                numero = len(analizador.eventos) - len(nuevos) + nuevos.index(evento) + 1
                nombre = f"evento_{numero:02d}_{evento.tipo}.jpg"
                cv2.imwrite(str(carpeta / "capturas" / nombre), anotado)

            if guardar_video:
                if escritor is None:
                    alto, ancho = anotado.shape[:2]
                    codigo = cv2.VideoWriter_fourcc(*"mp4v")
                    escritor = cv2.VideoWriter(str(carpeta / "video_anotado.mp4"),
                                               codigo, fps, (ancho, alto))
                escritor.write(anotado)

            # yield entrega el resultado de este fotograma y "pausa" la función
            # hasta que quien la usa (main.py o app.py) pida el siguiente
            yield anotado, nuevos, analizador

            if max_frames and leidos >= max_frames:
                break
    finally:
        # Este bloque se ejecuta siempre, también si el análisis se interrumpe:
        # así los resultados quedan guardados
        captura.release()
        if escritor is not None:
            escritor.release()
        guardar_resultados(analizador.resumen(), analizador.eventos, carpeta)
