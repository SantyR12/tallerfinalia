"""
Evaluación con videos reales.

Pasa cada video de una carpeta por el sistema completo y compara las alertas
de colisión con lo que realmente ocurre, anotado a mano en un archivo CSV:

    video,segundo_choque
    test (12).mp4,7.3        <- hay un choque alrededor del segundo 7.3
    trafico_normal.mp4,      <- sin accidente (segundo vacío)

Un choque cuenta como detectado si hay una alerta de colisión entre un segundo
antes y cuatro segundos después del momento anotado. Cualquier otra alerta de
colisión es una falsa alarma.

La comparación es solo por el momento: el script no sabe qué vehículos chocaron.
Hay que mirar la captura de cada alerta (salidas/capturas) para confirmar que
corresponde al choque y no a otros vehículos que pasaban en ese instante.

Uso:
    python evaluar.py --carpeta ruta/a/los/videos --verdad datos/choques_reales.csv
"""

import argparse
import csv
import os

import cv2

from src.detector import Detector
from src.eventos import AnalizadorEventos

# Ventana en la que una alerta cuenta como acierto: desde 1 s antes hasta 4 s
# después del momento anotado
ANTES_S, DESPUES_S = 1.0, 4.0


def analizar(ruta, detector):
    """Devuelve la lista de eventos que el sistema reporta en un video."""
    captura = cv2.VideoCapture(ruta)
    fps = captura.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 240:
        fps = 30.0
    detector.reiniciar(fps=fps)                 # fondo y seguimiento empiezan de cero
    analizador = AnalizadorEventos(fps=fps)
    indice = 0
    while True:
        ok, frame = captura.read()
        if not ok:
            break
        analizador.actualizar(indice, detector.detectar(frame))
        indice += 1
    captura.release()
    return analizador.eventos


def main():
    """Analiza cada video anotado e imprime el resultado y las métricas."""
    parser = argparse.ArgumentParser(description="Evalúa el sistema con videos reales anotados")
    parser.add_argument("--carpeta", required=True, help="carpeta con los videos")
    parser.add_argument("--verdad", default="datos/choques_reales.csv")
    args = parser.parse_args()

    with open(args.verdad, encoding="utf-8") as archivo:
        anotados = list(csv.DictReader(archivo))

    detector = Detector(fps=30.0)               # carga la CNN una sola vez
    # Contadores: choques detectados, videos con choque, alertas de colisión
    # falsas y alertas de otro tipo (riesgo y caída)
    detectados = con_choque = falsas = otras = 0
    for fila in anotados:
        ruta = os.path.join(args.carpeta, fila["video"])
        if not os.path.exists(ruta):
            print(f"{fila['video']:22s} no está en la carpeta; se omite")
            continue
        eventos = analizar(ruta, detector)
        colisiones = [e.tiempo_s for e in eventos if e.tipo == "colision"]
        otras += len(eventos) - len(colisiones)

        if fila["segundo_choque"].strip():
            con_choque += 1
            t = float(fila["segundo_choque"])
            aciertos = [c for c in colisiones if t - ANTES_S <= c <= t + DESPUES_S]
            falsas += len(colisiones) - len(aciertos)
            detectados += bool(aciertos)
            resultado = f"choque en {t:5.1f} s: " + ("DETECTADO" if aciertos else "no detectado")
        else:
            falsas += len(colisiones)
            resultado = "sin accidente"
        riesgos = sum(e.tipo == "riesgo" for e in eventos)
        caidas = sum(e.tipo == "caida" for e in eventos)
        print(f"{fila['video']:22s} {resultado:32s} colisiones en {colisiones}  "
              f"riesgos: {riesgos}  caídas: {caidas}")

    print(f"\nChoques detectados: {detectados} de {con_choque}")
    print(f"Alertas de colisión falsas: {falsas}")
    # Precisión: de las alertas de colisión, cuántas coinciden con un choque.
    # Recall: de los choques anotados, cuántos se detectaron.
    if detectados + falsas:
        print(f"Precisión: {detectados / (detectados + falsas):.2f}   "
              f"Recall: {detectados / max(con_choque, 1):.2f}")
    print(f"Alertas de riesgo y de caída (no se evalúan aquí): {otras}")
    print("Nota: solo se compara el momento de la alerta. Revise la captura de cada colisión "
          "para confirmar que corresponde al choque.")


if __name__ == "__main__":
    main()
