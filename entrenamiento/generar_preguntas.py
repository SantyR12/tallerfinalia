"""
Crea el dataset del asistente: un CSV con preguntas y su intención.

    python entrenamiento/generar_preguntas.py

Las preguntas se arman combinando formas de preguntar con sinónimos, de modo
que el Transformer vea muchas maneras de pedir lo mismo. El resultado queda en
datos/preguntas_asistente.csv con dos columnas: pregunta, intencion.
"""

import csv
import itertools
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

CUANTOS = ["cuantos", "cuantas", "que cantidad de", "numero de", "dime cuantos", "cuantos fueron los",
           "total de", "cantidad de"]
HUBO = ["hubo", "se detectaron", "se registraron", "ocurrieron", "pasaron", "se presentaron", "hay",
        "aparecen", "detecto el sistema"]
CUANDO = ["cuando", "en que segundo", "en que momento", "a que hora", "en que instante", "en que minuto"]
COLA = ["", "", "", "en el video", "en la grabacion", "por favor", "en total"]

TEMAS = {
    "colisiones": ["colisiones", "choques", "accidentes", "impactos", "colision", "choque", "accidente"],
    "caidas": ["caidas", "caida", "personas caidas", "motociclistas caidos", "gente que se cayo"],
    "riesgos": ["situaciones de riesgo", "riesgos", "situaciones peligrosas", "casi choques",
                "peligros", "alertas de riesgo"],
}
OBJETOS = {
    "vehiculos": ["vehiculos", "vehiculos en total", "automotores", "carros", "autos", "coches",
                  "automoviles", "carro"],
    "motos": ["motos", "motocicletas", "motociclistas", "moto"],
    "peatones": ["peatones", "personas", "gente", "transeuntes", "peaton"],
    "bicicletas": ["bicicletas", "ciclistas", "bicis", "bicicleta"],
}
# El sistema no separa carros, buses y camiones: estas preguntas tienen su propia respuesta
TIPOS_DE_VEHICULO = ["buses", "autobuses", "busetas", "bus", "camiones", "camion",
                     "vehiculos de carga", "tractomulas", "volquetas"]
FIJAS = {
    "resumen": ["resumen", "dame un resumen", "que paso", "que paso en el video", "que ocurrio",
                "informe general", "cuentame que paso", "resume el video", "dame el informe",
                "que sucedio", "haz un resumen", "resumen general", "que se detecto",
                "explicame que paso", "como estuvo el video", "que vio el sistema"],
    "ultimo_evento": ["cual fue el ultimo evento", "ultimo evento", "que fue lo ultimo que paso",
                      "cual fue la ultima alerta", "ultima alerta", "que paso al final",
                      "el evento mas reciente", "cual es el evento final", "lo ultimo que se detecto"],
    "primer_evento": ["cual fue el primer evento", "primer evento", "que fue lo primero que paso",
                      "cual fue la primera alerta", "primera alerta", "que paso al principio",
                      "el evento inicial", "como empezo", "lo primero que se detecto"],
    "eventos_graves": ["cuales fueron los eventos graves", "eventos graves", "que fue lo mas grave",
                       "eventos de nivel alto", "alertas graves", "que requiere atencion inmediata",
                       "cuales son los eventos mas importantes", "eventos urgentes",
                       "hay algo grave", "alertas de nivel alto", "lo mas peligroso que paso"],
    "listar_eventos": ["lista los eventos", "que eventos hubo", "cuales fueron los eventos",
                       "muestrame los eventos", "todas las alertas", "lista de alertas",
                       "que alertas hubo", "dime todos los eventos", "enumera los eventos",
                       "cuantos eventos hubo", "cuantas alertas hubo", "eventos detectados"],
    "tipo_vehiculo": ["que tipo de vehiculos pasaron", "que clase de vehiculos habia",
                      "eran carros o camiones", "habia buses o camiones", "que tipos de vehiculos hubo",
                      "pasaron buses o camiones", "eran carros buses o camiones",
                      "que tipo de vehiculos se detectaron", "distingue carros de camiones",
                      "cuales eran buses y cuales carros", "de que tipo eran los vehiculos",
                      "paso algun bus", "paso algun camion", "hubo algun bus", "hubo algun camion",
                      "vio algun bus", "vio algun camion"],
    "duracion": ["cuanto dura el video", "duracion del video", "cuantos segundos dura",
                 "cuanto tiempo se analizo", "que tan largo es el video", "cuantos fotogramas tiene",
                 "duracion", "cuanto duro la grabacion", "tiempo total del video"],
}
ADORNOS = ["", "", "", "oye ", "hola ", "por favor ", "me puedes decir ", "quisiera saber ", "dime "]
# Formas directas de preguntar si pasó algo, con verbos en vez de sustantivos
DIRECTAS = {
    "colisiones": ["alguien choco", "se estrello alguien", "chocaron", "se chocaron dos carros",
                   "hubo un estrellon", "se estrellaron", "atropellaron a alguien", "alguien se estrello",
                   "chocaron los vehiculos", "hubo algun golpe entre vehiculos", "se accidento alguien",
                   "paso algun accidente", "un carro golpeo a otro", "atropellaron a un peaton"],
    "caidas": ["alguien se cayo", "se cayo alguien", "se cayo un motociclista", "se cayo una persona",
               "alguien se fue al piso", "hubo alguna persona en el suelo", "se cayeron",
               "se cayo el de la moto", "alguien termino en el suelo", "se cayo un peaton"],
    "riesgos": ["hubo algo peligroso", "casi chocan", "casi se estrellan", "estuvieron a punto de chocar",
                "hubo algun susto", "casi atropellan a alguien", "paso algo riesgoso",
                "hubo maniobras peligrosas", "alguien estuvo en peligro", "casi hay un accidente"],
}


def generar():
    filas = set()
    for tema, sinonimos in TEMAS.items():
        for cuantos, palabra, hubo, cola in itertools.product(CUANTOS, sinonimos, HUBO, COLA):
            filas.add((f"{cuantos} {palabra} {hubo} {cola}".strip(), f"contar_{tema}"))
        for adorno, frase, cola in itertools.product(ADORNOS, DIRECTAS[tema], COLA):
            filas.add((f"{adorno}{frase} {cola}".strip(), f"contar_{tema}"))
        for hubo, palabra in itertools.product(["hubo", "hay", "se detecto", "ocurrio"], sinonimos):
            filas.add((f"{hubo} {palabra}", f"contar_{tema}"))
            filas.add((f"{hubo} algun {palabra}", f"contar_{tema}"))
        for cuando, palabra, cola in itertools.product(CUANDO, sinonimos, COLA):
            for verbo in ["fue", "ocurrio", "paso", "se detecto", "fueron", "ocurrieron"]:
                filas.add((f"{cuando} {verbo} {palabra} {cola}".strip(), f"momento_{tema}"))
    for objeto, sinonimos in OBJETOS.items():
        for cuantos, palabra, hubo, cola in itertools.product(CUANTOS, sinonimos, HUBO, COLA):
            filas.add((f"{cuantos} {palabra} {hubo} {cola}".strip(), f"contar_{objeto}"))
    for cuantos, palabra, hubo, cola in itertools.product(CUANTOS, TIPOS_DE_VEHICULO, HUBO, COLA):
        filas.add((f"{cuantos} {palabra} {hubo} {cola}".strip(), "tipo_vehiculo"))
    for intencion, frases in FIJAS.items():
        for adorno, frase, cola in itertools.product(ADORNOS, frases, COLA):
            filas.add((f"{adorno}{frase} {cola}".strip(), intencion))
    return sorted(filas)


def main(por_intencion=400, semilla=7):
    random.seed(semilla)
    por_clase = {}
    for pregunta, intencion in generar():
        por_clase.setdefault(intencion, []).append(pregunta)

    filas = []
    for intencion, preguntas in sorted(por_clase.items()):
        random.shuffle(preguntas)
        filas += [(p, intencion) for p in preguntas[:por_intencion]]     # clases equilibradas
    random.shuffle(filas)

    os.makedirs(os.path.dirname(config.RUTA_PREGUNTAS), exist_ok=True)
    with open(config.RUTA_PREGUNTAS, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["pregunta", "intencion"])
        escritor.writerows(filas)
    print(f"{len(filas)} preguntas de {len(por_clase)} intenciones -> {config.RUTA_PREGUNTAS}")
    for intencion, preguntas in sorted(por_clase.items()):
        print(f"  {intencion:22s} {min(len(preguntas), por_intencion)}")


if __name__ == "__main__":
    main()
