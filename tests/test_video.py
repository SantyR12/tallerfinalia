"""
Prueba del recorrido completo (video -> eventos -> alertas -> archivos) con un
detector de mentira que devuelve cajas programadas. Comprueba que las alertas
se dibujan y que los resultados se guardan.
"""

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import Deteccion  # noqa: E402
from src.video import procesar  # noqa: E402


class DetectorSimulado:
    """Un carro avanza hacia una moto detenida y frena de golpe al tocarla."""

    def __init__(self):
        self.frame = -1

    def reiniciar(self, fps=30.0):
        self.frame = -1

    def detectar(self, imagen):
        self.frame += 1
        x = min(100 + 6 * self.frame, 345)
        return [
            Deteccion(1, "vehiculo", (x - 50, 240, x + 50, 300), 0.9),
            Deteccion(2, "motocicleta", (375, 240, 425, 300), 0.8),
        ]


def crear_video_gris(ruta, frames=90, fps=30, tam=(640, 360)):
    escritor = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), fps, tam)
    for _ in range(frames):
        escritor.write(np.full((tam[1], tam[0], 3), 90, dtype=np.uint8))
    escritor.release()


def test_recorrido_completo_guarda_alertas_y_resultados(tmp_path):
    video = tmp_path / "entrada.mp4"
    salida = tmp_path / "salidas"
    crear_video_gris(video)

    alertas = []
    franja_roja = False
    for anotado, nuevos, analizador in procesar(str(video), carpeta_salida=salida,
                                                detector=DetectorSimulado()):
        alertas += nuevos
        azul, verde, rojo = anotado[5, 320]             # píxel de la franja superior
        if rojo > 200 and azul < 50 and verde < 50:
            franja_roja = True

    assert [e.tipo for e in alertas] == ["riesgo", "colision"]
    assert franja_roja, "la alerta de nivel ALTO debe pintar la franja roja"

    # Archivos de resultados
    eventos = json.loads((salida / "eventos.json").read_text(encoding="utf-8"))
    resumen = json.loads((salida / "resumen.json").read_text(encoding="utf-8"))
    assert [e["tipo"] for e in eventos] == ["riesgo", "colision"]
    assert resumen["eventos"] == {"colision": 1, "riesgo": 1, "caida": 0}
    assert resumen["objetos"] == {"vehiculo": 1, "motocicleta": 1}
    assert (salida / "eventos.csv").read_text(encoding="utf-8").count("\n") == 3
    assert "1 colisión" in (salida / "informe.txt").read_text(encoding="utf-8")
    assert (salida / "video_anotado.mp4").stat().st_size > 0
    capturas = sorted(p.name for p in (salida / "capturas").glob("*.jpg"))
    assert capturas == ["evento_01_riesgo.jpg", "evento_02_colision.jpg"]
