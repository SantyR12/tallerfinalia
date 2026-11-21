"""
Alertas visuales y registro de resultados.

- dibujar(): pinta sobre el fotograma las cajas, los IDs, el contador de
  objetos y, si hay un evento reciente, un aviso en la parte superior.
- guardar_resultados(): escribe los eventos en CSV y JSON, el resumen y el
  informe en texto que después lee el asistente.
"""

import csv
import json
import unicodedata
from pathlib import Path

import cv2

import config

# Colores en formato BGR (el que usa OpenCV)
COLOR_CLASE = {
    "persona": (255, 180, 0),
    "bicicleta": (200, 200, 0),
    "vehiculo": (0, 200, 0),
    "motocicleta": (0, 200, 200),
}
COLOR_NIVEL = {"ALTO": (0, 0, 255), "MEDIO": (0, 165, 255)}   # rojo y naranja
ETIQUETA_TIPO = {"colision": "COLISION", "riesgo": "RIESGO DE COLISION", "caida": "CAIDA"}
FUENTE = cv2.FONT_HERSHEY_SIMPLEX


def sin_tildes(texto):
    """cv2.putText no dibuja tildes ni eñes; se reemplazan por la letra base."""
    normal = unicodedata.normalize("NFD", texto)
    return "".join(c for c in normal if unicodedata.category(c) != "Mn")


def dibujar(frame, detecciones, eventos_activos, resumen):
    """Devuelve el fotograma con las detecciones y las alertas dibujadas."""
    en_alerta = {}                              # id -> color de la alerta
    for evento in eventos_activos:
        for id_objeto in evento.ids:
            en_alerta[id_objeto] = COLOR_NIVEL[evento.nivel]

    # Cajas e identificadores
    for d in detecciones:
        if getattr(d, "estimada", False):
            continue                            # oculto dentro de otro: no hay caja real que dibujar
        x1, y1, x2, y2 = (int(v) for v in d.caja)
        color = en_alerta.get(d.id, COLOR_CLASE.get(d.clase, (255, 255, 255)))
        grosor = 4 if d.id in en_alerta else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, grosor)
        etiqueta = f"{d.clase} {d.id}"
        (ancho, alto), _ = cv2.getTextSize(etiqueta, FUENTE, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - alto - 6), (x1 + ancho + 4, y1), color, -1)
        color_texto = (255, 255, 255) if d.id in en_alerta else (0, 0, 0)
        cv2.putText(frame, etiqueta, (x1 + 2, y1 - 4), FUENTE, 0.5, color_texto, 1, cv2.LINE_AA)

    # Contador de objetos vistos hasta ahora (esquina inferior izquierda)
    objetos = resumen["objetos"]
    texto = (
        f"Vehiculos: {objetos.get('vehiculo', 0)}  Motos: {objetos.get('motocicleta', 0)}  "
        f"Bicicletas: {objetos.get('bicicleta', 0)}  Peatones: {objetos.get('persona', 0)}  "
        f"Eventos: {sum(resumen['eventos'].values())}"
    )
    alto_frame = frame.shape[0]
    cv2.rectangle(frame, (0, alto_frame - 28), (frame.shape[1], alto_frame), (0, 0, 0), -1)
    cv2.putText(frame, texto, (8, alto_frame - 9), FUENTE, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Aviso de alerta (franja superior): se muestra el evento más grave
    if eventos_activos:
        evento = max(eventos_activos, key=lambda e: (e.nivel == "ALTO", e.tiempo_s))
        color = COLOR_NIVEL[evento.nivel]
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 54), color, -1)
        cv2.putText(frame, f"ALERTA: {ETIQUETA_TIPO[evento.tipo]}", (10, 23), FUENTE, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, sin_tildes(evento.descripcion)[:95], (10, 45), FUENTE, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def guardar_resultados(resumen, eventos, carpeta=config.CARPETA_SALIDAS):
    """Guarda eventos.csv, eventos.json, resumen.json e informe.txt en la carpeta."""
    from src.asistente import construir_informe   # import tardío: evita ciclo

    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    filas = [e.como_dict() for e in eventos]

    with open(carpeta / "eventos.csv", "w", newline="", encoding="utf-8") as archivo:
        columnas = ["numero", "tipo", "nivel", "tiempo_s", "frame", "ids", "clases", "descripcion"]
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        for numero, fila in enumerate(filas, start=1):
            escritor.writerow({
                "numero": numero,
                "tipo": fila["tipo"],
                "nivel": fila["nivel"],
                "tiempo_s": fila["tiempo_s"],
                "frame": fila["frame"],
                "ids": " ".join(str(i) for i in fila["ids"]),
                "clases": " ".join(fila["clases"]),
                "descripcion": fila["descripcion"],
            })

    with open(carpeta / "eventos.json", "w", encoding="utf-8") as archivo:
        json.dump(filas, archivo, ensure_ascii=False, indent=2)
    with open(carpeta / "resumen.json", "w", encoding="utf-8") as archivo:
        json.dump(resumen, archivo, ensure_ascii=False, indent=2)
    with open(carpeta / "informe.txt", "w", encoding="utf-8") as archivo:
        archivo.write(construir_informe(resumen, filas))
    return carpeta


def cargar_resultados(carpeta=config.CARPETA_SALIDAS):
    """Lee el resumen y los eventos de una ejecución anterior."""
    carpeta = Path(carpeta)
    with open(carpeta / "resumen.json", encoding="utf-8") as archivo:
        resumen = json.load(archivo)
    with open(carpeta / "eventos.json", encoding="utf-8") as archivo:
        eventos = json.load(archivo)
    return resumen, eventos
