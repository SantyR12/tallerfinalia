"""
Programa de consola.

Ejemplos:
    python main.py --video data/accidente.mp4
    python main.py --camara 0
    python main.py --video data/accidente.mp4 --asistente
    python main.py --consultar salidas        (preguntar sobre un análisis ya hecho)

Mientras se muestra la ventana, la tecla Q detiene el análisis.
"""

import argparse
import sys

import cv2

import config
from src.alertas import cargar_resultados
from src.asistente import Asistente, conversar


def leer_argumentos():
    parser = argparse.ArgumentParser(description="Detección de accidentes de tránsito")
    fuente = parser.add_mutually_exclusive_group(required=True)
    fuente.add_argument("--video", help="ruta de un archivo de video")
    fuente.add_argument("--camara", type=int, help="número de cámara (0 = principal)")
    fuente.add_argument("--consultar", metavar="CARPETA",
                        help="abrir el asistente sobre resultados ya guardados")
    parser.add_argument("--salida", default=config.CARPETA_SALIDAS, help="carpeta de resultados")
    parser.add_argument("--sin-ventana", action="store_true", help="no mostrar el video")
    parser.add_argument("--max-frames", type=int, help="procesar solo esta cantidad de fotogramas")
    parser.add_argument("--asistente", action="store_true", help="abrir el asistente al terminar")
    parser.add_argument("--sin-llm", action="store_true",
                        help="asistente sin el modelo de lenguaje")
    parser.add_argument("--sin-transformer", action="store_true",
                        help="asistente sin el clasificador de intenciones")
    return parser.parse_args()


def analizar(args):
    from src.video import procesar      # import tardío: --consultar no necesita la CNN

    fuente = args.camara if args.camara is not None else args.video
    pasos = procesar(fuente, carpeta_salida=args.salida, max_frames=args.max_frames)
    analizador = None
    try:
        for anotado, nuevos, analizador in pasos:
            for evento in nuevos:
                print(f"[ALERTA {evento.nivel}] {evento.descripcion}")
            if not args.sin_ventana:
                cv2.imshow("Deteccion de accidentes (Q para salir)", anotado)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
    finally:
        pasos.close()                   # guarda los resultados aunque se interrumpa
        cv2.destroyAllWindows()
    return analizador


def main():
    args = leer_argumentos()

    if args.consultar:
        carpeta = args.consultar
    else:
        try:
            analizador = analizar(args)
        except FileNotFoundError as error:      # falta entrenar la CNN
            sys.exit(str(error))
        carpeta = args.salida
        if analizador is None:
            print("No se leyó ningún fotograma.")
            return
        resumen = analizador.resumen()
        print("\n--- Resumen ---")
        print(f"Fotogramas: {resumen['frames']}  Duración: {resumen['duracion_s']} s")
        print(f"Objetos: {resumen['objetos']}")
        print(f"Eventos: {resumen['eventos']}")
        print(f"Resultados guardados en: {carpeta}/")

    resumen, eventos = cargar_resultados(carpeta)
    asistente = Asistente(resumen, eventos, usar_transformer=not args.sin_transformer,
                          usar_llm=not args.sin_llm)
    print("\n--- Informe ---")
    print(asistente.informe)
    if args.asistente or args.consultar:
        conversar(asistente)


if __name__ == "__main__":
    main()
