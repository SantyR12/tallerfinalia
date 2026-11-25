"""
El "idioma" del modelo de lenguaje propio: qué lee, qué escribe y cómo se
convierten sus palabras en la respuesta final.

El modelo de lenguaje (src/modelo_lenguaje.py) recibe una secuencia así:

    <datos> colisiones_uno riesgos_cero ... <pregunta> cuantas colisiones hubo <respuesta>

y la continúa palabra por palabra:

    se detectó <n_colisiones> colisión : <lista_colisiones> <fin>

Dos decisiones de diseño:

1. DATOS RESUMIDOS. Al modelo no se le dan las cifras exactas, solo si cada
   cantidad es cero, uno o varios. Con eso decide la redacción ("no hubo",
   "se detectó una", "se detectaron varias").

2. RANURAS. El modelo no escribe los números: escribe ranuras como
   <n_colisiones> o <lista_colisiones>, y el programa las llena con los datos
   del registro. Un modelo de lenguaje pequeño puede equivocarse al copiar una
   cifra; así las cifras de la respuesta siempre son las reales.
"""

import random
import re

from src.texto import normalizar

DATOS, PREGUNTA, RESPUESTA, FIN = "<datos>", "<pregunta>", "<respuesta>", "<fin>"

# Cantidades que describen un video, en el orden en que se le presentan al modelo
CANTIDADES = ["colisiones", "riesgos", "caidas", "vehiculos", "motos", "peatones",
              "bicicletas", "objetos", "eventos", "graves"]
TIPO_DE = {"colisiones": "colision", "riesgos": "riesgo", "caidas": "caida"}
CLASE_DE = {"vehiculos": "vehiculo", "motos": "motocicleta", "peatones": "persona",
            "bicicletas": "bicicleta"}


# ----------------------------------------------------------------------------
# Del registro de eventos a los datos que lee el modelo
# ----------------------------------------------------------------------------
def contar(resumen, eventos):
    """Devuelve el diccionario cantidad -> número a partir del registro."""
    objetos = resumen.get("objetos", {})
    cuenta = {c: sum(1 for e in eventos if e["tipo"] == t) for c, t in TIPO_DE.items()}
    cuenta.update({c: objetos.get(clase, 0) for c, clase in CLASE_DE.items()})
    cuenta["objetos"] = sum(objetos.values())
    cuenta["eventos"] = len(eventos)
    cuenta["graves"] = sum(1 for e in eventos if e["nivel"] == "ALTO")
    return cuenta


def categoria(n):
    """Resume un número en la palabra que ve el modelo: 0 -> 'cero', 1 -> 'uno', 2 o más -> 'varios'."""
    return "cero" if n == 0 else "uno" if n == 1 else "varios"


def fichas_de_datos(cuenta):
    """['colisiones_uno', 'riesgos_cero', ...]: una ficha por cantidad."""
    return [f"{c}_{categoria(cuenta[c])}" for c in CANTIDADES]


# ----------------------------------------------------------------------------
# Las respuestas que el modelo aprende a escribir
# ----------------------------------------------------------------------------
EVENTO = {    # tema -> (singular, plural, "ninguna ...")
    "colisiones": ("colisión", "colisiones", "ninguna colisión"),
    "caidas": ("caída", "caídas", "ninguna caída"),
    "riesgos": ("situación de riesgo", "situaciones de riesgo", "ninguna situación de riesgo"),
}
OBJETO = {    # tema -> (singular, plural, "ningún ...")
    "vehiculos": ("vehículo", "vehículos", "ningún vehículo"),
    "motos": ("motocicleta", "motocicletas", "ninguna motocicleta"),
    "peatones": ("peatón", "peatones", "ningún peatón"),
    "bicicletas": ("bicicleta", "bicicletas", "ninguna bicicleta"),
}


def respuestas_validas(intencion, cuenta):
    """
    Todas las formas correctas de responder a una intención dado un video.
    Se usan para crear el corpus de entrenamiento y para comprobar que lo que
    escribe el modelo es coherente con los datos.
    """
    # "contar_colisiones" se separa en la acción ("contar") y el tema ("colisiones")
    accion, _, tema = intencion.partition("_")

    if tema in EVENTO:
        singular, plural, ninguna = EVENTO[tema]
        cat = categoria(cuenta[tema])
        if cat == "cero":
            return [f"no se detectó {ninguna} en el video", f"no hubo {plural}",
                    f"el sistema no registró {ninguna}"]
        if accion == "contar":
            if cat == "uno":
                return [f"se detectó <n_{tema}> {singular} : <lista_{tema}>",
                        f"hubo <n_{tema}> {singular} : <lista_{tema}>",
                        f"el sistema registró <n_{tema}> {singular} : <lista_{tema}>"]
            return [f"se detectaron <n_{tema}> {plural} : <lista_{tema}>",
                    f"hubo <n_{tema}> {plural} : <lista_{tema}>",
                    f"el sistema registró <n_{tema}> {plural} : <lista_{tema}>"]
        if cat == "uno":
            return [f"la {singular} ocurrió en el <t_{tema}>", f"fue en el <t_{tema}>",
                    f"hubo <n_{tema}> {singular} , en el <t_{tema}>"]
        return [f"las {plural} ocurrieron en : <t_{tema}>", f"fueron en : <t_{tema}>",
                f"hubo <n_{tema}> {plural} , en : <t_{tema}>"]

    if tema in OBJETO:
        singular, plural, ninguno = OBJETO[tema]
        cat = categoria(cuenta[tema])
        if cat == "cero":
            return [f"no se detectó {ninguno} en el video", f"no pasó {ninguno}"]
        if cat == "uno":
            return [f"se detectó <n_{tema}> {singular} en el video", f"pasó <n_{tema}> {singular}"]
        return [f"se detectaron <n_{tema}> {plural} en el video", f"pasaron <n_{tema}> {plural}"]

    if intencion == "tipo_vehiculo":
        # Preguntas por buses, camiones o el tipo de vehículo: el sistema no los separa
        cat = categoria(cuenta["vehiculos"])
        if cat == "cero":
            return ["el sistema no distingue el tipo de vehículo , y no se detectó ningún vehículo",
                    "no se distingue entre carros , buses y camiones ; no pasó ningún vehículo"]
        if cat == "uno":
            return ["el sistema no distingue el tipo de vehículo ; se detectó <n_vehiculos> vehículo",
                    "no se distingue entre carros , buses y camiones ; pasó <n_vehiculos> vehículo"]
        return ["el sistema no distingue el tipo de vehículo ; se detectaron <n_vehiculos> vehículos",
                "no se distingue entre carros , buses y camiones ; pasaron <n_vehiculos> vehículos"]

    if intencion == "duracion":
        return ["el video dura <duracion> segundos y tiene <fotogramas> fotogramas",
                "se analizaron <fotogramas> fotogramas , que son <duracion> segundos"]

    eventos = categoria(cuenta["eventos"])
    if intencion == "resumen":
        inicio = ["el video dura <duracion> segundos .", "se analizaron <duracion> segundos de video ."]
        if cuenta["objetos"] == 0:
            medio = ["no se detectó ningún vehículo ni peatón ."]
        elif cuenta["objetos"] == 1:
            medio = ["se detectó <n_objetos> objeto : <lista_objetos> .",
                     "pasó <n_objetos> objeto : <lista_objetos> ."]
        else:
            medio = ["se detectaron <n_objetos> objetos : <lista_objetos> .",
                     "pasaron <n_objetos> objetos : <lista_objetos> ."]
        if eventos == "cero":
            final = ["no se registró ningún evento", "no hubo colisiones , caídas ni situaciones de riesgo"]
        elif eventos == "uno":
            final = ["se registró <n_eventos> evento : <lista_eventos>"]
        else:
            final = ["se registraron <n_eventos> eventos : <lista_eventos>"]
        return [f"{a} {b} {c}" for a in inicio for b in medio for c in final]

    if eventos == "cero" and intencion in ("ultimo_evento", "primer_evento", "listar_eventos"):
        return ["no se registró ningún evento en el video", "no hubo eventos"]
    if intencion == "ultimo_evento":
        return ["el último evento fue : <ultimo_evento>", "lo último que se detectó fue : <ultimo_evento>"]
    if intencion == "primer_evento":
        return ["el primer evento fue : <primer_evento>", "lo primero que se detectó fue : <primer_evento>"]
    if intencion == "listar_eventos":
        if eventos == "uno":
            return ["se registró <n_eventos> evento : <lista_eventos>", "hubo <n_eventos> evento : <lista_eventos>"]
        return ["se registraron <n_eventos> eventos : <lista_eventos>",
                "hubo <n_eventos> eventos : <lista_eventos>"]
    if intencion == "eventos_graves":
        graves = categoria(cuenta["graves"])
        if graves == "cero":
            return ["no hubo eventos de nivel alto", "no se registró ningún evento grave"]
        if graves == "uno":
            return ["hay <n_graves> evento de nivel alto : <lista_graves>",
                    "se registró <n_graves> evento grave : <lista_graves>"]
        return ["hay <n_graves> eventos de nivel alto : <lista_graves>",
                "se registraron <n_graves> eventos graves : <lista_graves>"]
    raise ValueError(f"Intención desconocida: {intencion}")


def cuenta_al_azar(rng=random):
    """Inventa un video: cuántos eventos y objetos de cada tipo hubo."""
    def cantidad():
        """Un número al azar: 0 (35 % de las veces), 1 (25 %) o entre 2 y 9 (40 %)."""
        return rng.choices([0, 1, rng.randint(2, 9)], weights=[0.35, 0.25, 0.40])[0]

    cuenta = {c: cantidad() for c in list(TIPO_DE) + list(CLASE_DE)}
    cuenta["objetos"] = sum(cuenta[c] for c in CLASE_DE)
    cuenta["eventos"] = sum(cuenta[c] for c in TIPO_DE)
    cuenta["graves"] = cuenta["colisiones"] + cuenta["caidas"]
    return cuenta


# ----------------------------------------------------------------------------
# Texto <-> fichas
# ----------------------------------------------------------------------------
def fichas_de_pregunta(pregunta, largo):
    """'¿Cuántas colisiones hubo?' -> ['cuantas', 'colisiones', 'hubo'] (máximo `largo` palabras)."""
    return normalizar(pregunta).split()[:largo]


def armar_entrada(cuenta, pregunta, largo_pregunta):
    """La secuencia que se le entrega al modelo para que la continúe."""
    return ([DATOS] + fichas_de_datos(cuenta) + [PREGUNTA]
            + fichas_de_pregunta(pregunta, largo_pregunta) + [RESPUESTA])


# ----------------------------------------------------------------------------
# De lo que escribe el modelo a la respuesta final
# ----------------------------------------------------------------------------
def _lista(elementos):
    """['a', 'b', 'c'] -> 'a, b y c'"""
    elementos = list(elementos)
    if len(elementos) <= 1:
        return "".join(elementos)
    return ", ".join(elementos[:-1]) + " y " + elementos[-1]


NOMBRE_CLASE = {    # clase -> (singular, plural), como se escriben en las respuestas
    "vehiculo": ("vehículo", "vehículos"), "motocicleta": ("motocicleta", "motocicletas"),
    "bicicleta": ("bicicleta", "bicicletas"), "persona": ("persona", "personas"),
}


def con_nombre(n, clase):
    """'1 vehículo', '3 personas'."""
    singular, plural = NOMBRE_CLASE.get(clase, (clase, clase + "s"))
    return f"{n} {singular if n == 1 else plural}"


def valores_de_ranuras(resumen, eventos):
    """Calcula, a partir del registro, el texto que va en cada ranura."""
    cuenta = contar(resumen, eventos)
    # Ranuras de cantidad: <n_colisiones>, <n_vehiculos>, ...
    valores = {f"<n_{c}>": str(n) for c, n in cuenta.items()}
    # Ranuras de lista y de momento, por tipo de evento: <lista_colisiones>, <t_colisiones>, ...
    for tema, tipo in TIPO_DE.items():
        del_tipo = [e for e in eventos if e["tipo"] == tipo]
        valores[f"<lista_{tema}>"] = "; ".join(e["descripcion"] for e in del_tipo)
        valores[f"<t_{tema}>"] = _lista(f"segundo {e['tiempo_s']}" for e in del_tipo)
    graves = [e for e in eventos if e["nivel"] == "ALTO"]
    valores["<lista_graves>"] = "; ".join(e["descripcion"] for e in graves)
    valores["<lista_eventos>"] = "; ".join(e["descripcion"] for e in eventos)
    valores["<primer_evento>"] = eventos[0]["descripcion"] if eventos else ""
    valores["<ultimo_evento>"] = eventos[-1]["descripcion"] if eventos else ""
    valores["<lista_objetos>"] = _lista(
        con_nombre(n, clase)
        for clase, n in sorted(resumen.get("objetos", {}).items(), key=lambda par: -par[1])
    )
    valores["<duracion>"] = str(resumen.get("duracion_s", 0))
    valores["<fotogramas>"] = str(resumen.get("frames", 0))
    return valores


def rellenar(texto, resumen, eventos):
    """Cambia las ranuras por los datos reales y arregla mayúsculas y espacios."""
    for ranura, valor in valores_de_ranuras(resumen, eventos).items():
        texto = texto.replace(ranura, valor)
    texto = re.sub(r"\s+([:,.;])", r"\1", texto).strip()        # "evento : x" -> "evento: x"
    frases = [f.strip() for f in re.split(r"(?<=\.)\s+", texto) if f.strip()]
    texto = " ".join(f[0].upper() + f[1:] for f in frases)
    return texto if texto.endswith(".") else texto + "."
