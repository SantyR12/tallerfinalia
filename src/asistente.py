"""
Asistente para consultar los eventos detectados. Usa dos redes propias,
entrenadas desde cero, y ningún modelo preentrenado ni servicio externo:

1. MODELO DE LENGUAJE (src/modelo_lenguaje.py): un Transformer decodificador
   tipo GPT en pequeño. Recibe los datos resumidos del video y la pregunta, y
   ESCRIBE la respuesta palabra por palabra. Donde va una cifra escribe una
   ranura (<n_colisiones>) que el programa llena con el dato real.

2. CLASIFICADOR DE INTENCIONES (src/transformer.py): un Transformer
   codificador que dice qué se está preguntando. Sirve de control: la
   respuesta del modelo de lenguaje solo se acepta si es coherente con esa
   intención y con los datos del registro.

Si el modelo de lenguaje no está entrenado, o escribe algo incoherente, la
respuesta se arma a partir de la intención. Si tampoco está el clasificador,
la intención se deduce por palabras clave. Cada respuesta indica su origen.
"""

import json
import unicodedata

import config
from src.lenguaje import armar_entrada, con_nombre, contar, rellenar, respuestas_validas

INTENCIONES = [
    "contar_colisiones", "contar_caidas", "contar_riesgos",
    "momento_colisiones", "momento_caidas", "momento_riesgos",
    "contar_vehiculos", "contar_motos", "contar_peatones", "contar_bicicletas", "tipo_vehiculo",
    "resumen", "duracion", "primer_evento", "ultimo_evento", "eventos_graves", "listar_eventos",
]

NOMBRE_TIPO = {"colision": "colisión", "riesgo": "situación de riesgo", "caida": "caída"}
PLURAL_TIPO = {"colision": "colisiones", "riesgo": "situaciones de riesgo", "caida": "caídas"}


def _cantidad(n, singular, plural):
    return f"{n} {singular if n == 1 else plural}"


def _se(n, singular, plural):
    """Concordancia del verbo: 'Se detectó 1 vehículo' / 'Se detectaron 2 vehículos'."""
    return singular if n == 1 else plural


def _normalizar(texto):
    """Minúsculas y sin tildes, para comparar palabras clave."""
    texto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def _lista(elementos):
    """['a', 'b', 'c'] -> 'a, b y c'"""
    elementos = list(elementos)
    if len(elementos) <= 1:
        return "".join(elementos)
    return ", ".join(elementos[:-1]) + " y " + elementos[-1]


# ----------------------------------------------------------------------------
# Paso 1: del registro de eventos a un informe en texto
# ----------------------------------------------------------------------------
def construir_informe(resumen, eventos):
    """Redacta el informe que el Transformer usará como contexto."""
    objetos = resumen.get("objetos", {})
    conteo = resumen.get("eventos", {})
    total_objetos = sum(objetos.values())
    total_eventos = len(eventos)
    frases = [
        f"El video analizado dura {resumen.get('duracion_s', 0)} segundos "
        f"y tiene {resumen.get('frames', 0)} fotogramas."
    ]

    if total_objetos:
        detalle = _lista(
            con_nombre(n, clase) for clase, n in sorted(objetos.items(), key=lambda par: -par[1])
        )
        frases.append(
            f"En total {_se(total_objetos, 'se detectó', 'se detectaron')} "
            f"{_cantidad(total_objetos, 'objeto', 'objetos')}: {detalle}."
        )
    else:
        frases.append("No se detectó ningún vehículo ni peatón.")

    if total_eventos == 0:
        frases.append(
            "No se registró ningún evento: no hubo colisiones, "
            "caídas ni situaciones de riesgo."
        )
        return " ".join(frases)

    detalle = _lista(
        _cantidad(conteo.get(tipo, 0), NOMBRE_TIPO[tipo], PLURAL_TIPO[tipo])
        for tipo in ("colision", "riesgo", "caida")
    )
    frases.append(
        f"{_se(total_eventos, 'Se registró', 'Se registraron')} "
        f"{_cantidad(total_eventos, 'evento', 'eventos')}: {detalle}."
    )

    for numero, e in enumerate(eventos, start=1):
        quienes = _lista(f"{config.NOMBRE_CLASE.get(clase, clase)} {id_}"
                         for clase, id_ in zip(e["clases"], e["ids"]))
        frases.append(
            f"El evento {numero} fue una {NOMBRE_TIPO[e['tipo']]} de nivel {e['nivel']}, "
            f"ocurrió en el segundo {e['tiempo_s']} y participaron: {quienes}."
        )

    for tipo in ("colision", "riesgo", "caida"):
        del_tipo = [e for e in eventos if e["tipo"] == tipo]
        if del_tipo:
            frases.append(
                f"La primera {NOMBRE_TIPO[tipo]} ocurrió en el segundo {del_tipo[0]['tiempo_s']}."
            )
    frases.append(f"El último evento ocurrió en el segundo {eventos[-1]['tiempo_s']}.")

    graves = [n for n, e in enumerate(eventos, start=1) if e["nivel"] == "ALTO"]
    if graves:
        frases.append(
            f"Los eventos de nivel ALTO, que requieren atención inmediata, son: "
            f"{_lista(f'el evento {n}' for n in graves)}."
        )
    return " ".join(frases)


# ----------------------------------------------------------------------------
# El asistente
# ----------------------------------------------------------------------------
class Asistente:
    def __init__(self, resumen, eventos, usar_transformer=True, usar_llm=True,
                 ruta_modelo=config.RUTA_ASISTENTE, ruta_vocabulario=config.RUTA_VOCABULARIO,
                 ruta_llm=config.RUTA_LLM, ruta_vocabulario_llm=config.RUTA_VOCABULARIO_LLM):
        self.resumen = resumen
        self.eventos = eventos
        self.informe = construir_informe(resumen, eventos)
        self.cuenta = contar(resumen, eventos)      # cuántos eventos y objetos de cada tipo
        self.usar_transformer = usar_transformer
        self.usar_llm = usar_llm
        self.ruta_modelo, self.ruta_vocabulario = ruta_modelo, ruta_vocabulario
        self.ruta_llm, self.ruta_vocabulario_llm = ruta_llm, ruta_vocabulario_llm
        self.modelo = None              # clasificador de intenciones
        self.vocabulario = None
        self.intenciones = None
        self.largo = config.LARGO_PREGUNTA
        self.error_modelo = None        # mensaje si el clasificador no cargó
        self.llm = None                 # modelo de lenguaje
        self.error_llm = None           # mensaje si el modelo de lenguaje no cargó

    # ------------------------------------------------------------------ API
    def responder(self, pregunta):
        """Devuelve {'respuesta', 'fuente', 'confianza', 'intencion'}.

        fuente = 'llm'         la escribió el modelo de lenguaje
                 'transformer' se armó con la intención que dio el clasificador
                 'reglas'      se armó con la intención deducida por palabras clave
        """
        pregunta = pregunta.strip()
        if not pregunta:
            return {"respuesta": "Escribe una pregunta sobre el video.", "fuente": "reglas",
                    "confianza": None, "intencion": None}

        # 1. ¿Qué se está preguntando? (clasificador; si no, palabras clave)
        intencion, confianza, origen = None, None, "reglas"
        if self.usar_transformer and self._cargar_modelo():
            candidata, seguridad = self._intencion_transformer(pregunta)
            if candidata and seguridad >= config.CONFIANZA_ASISTENTE:
                intencion, confianza, origen = candidata, round(seguridad, 3), "transformer"
        if intencion is None:
            intencion = self._intencion_por_palabras(pregunta)

        # 2. El modelo de lenguaje escribe la respuesta
        if self.usar_llm and self._cargar_llm():
            escrito = self._escribir_con_llm(pregunta)
            if escrito is not None and self._es_coherente(escrito, intencion):
                return {"respuesta": rellenar(escrito, self.resumen, self.eventos),
                        "fuente": "llm", "confianza": confianza, "intencion": intencion}

        # 3. Respaldo: respuesta armada a partir de la intención
        return {"respuesta": self._responder_intencion(intencion), "fuente": origen,
                "confianza": confianza, "intencion": intencion}

    # ----------------------------------------------------- modelo de lenguaje
    def _cargar_llm(self):
        """Carga el modelo de lenguaje la primera vez. Devuelve True si está listo."""
        if self.llm is not None:
            return True
        if self.error_llm is not None:
            return False
        try:
            from src.modelo_lenguaje import ModeloLenguaje

            self.llm = ModeloLenguaje(self.ruta_llm, self.ruta_vocabulario_llm)
            return True
        except Exception as error:      # no está entrenado, falta Keras...
            self.error_llm = (
                f"{type(error).__name__}: {error}. Entrénalo con: "
                "python entrenamiento/entrenar_llm.py"
            )
            return False

    def _escribir_con_llm(self, pregunta):
        """El modelo continúa '<datos> ... <pregunta> ... <respuesta>'. None si no terminó."""
        entrada = armar_entrada(self.cuenta, pregunta, config.LARGO_PREGUNTA)
        fichas, termino = self.llm.generar(entrada)
        return " ".join(fichas) if termino and fichas else None

    def _es_coherente(self, escrito, intencion):
        """
        Control de lo que escribió el modelo de lenguaje. Solo se acepta si es
        una respuesta correcta para la intención detectada y para los datos
        del video: no puede hablar de la duración si se preguntó por las
        colisiones, ni decir "no hubo colisiones" si hubo dos.
        """
        return escrito in respuestas_validas(intencion, self.cuenta)

    # ------------------------------------------- clasificador de intenciones
    def _cargar_modelo(self):
        """Carga el clasificador la primera vez. Devuelve True si está listo."""
        if self.modelo is not None:
            return True
        if self.error_modelo is not None:
            return False
        try:
            import keras

            import src.transformer  # noqa: F401  (registra las capas propias)

            with open(self.ruta_vocabulario, encoding="utf-8") as archivo:
                datos = json.load(archivo)
            self.vocabulario = datos["vocabulario"]
            self.intenciones = datos["intenciones"]
            self.largo = datos["largo"]
            self.modelo = keras.models.load_model(self.ruta_modelo)
            return True
        except Exception as error:      # no está entrenado, falta Keras...
            self.error_modelo = (
                f"{type(error).__name__}: {error}. Entrénalo con: "
                "python entrenamiento/entrenar_asistente.py"
            )
            return False

    def _intencion_transformer(self, pregunta):
        """Devuelve (intención, confianza) según el clasificador."""
        import numpy as np

        from src.texto import codificar

        indices = codificar(pregunta, self.vocabulario, self.largo)
        if not any(i > 1 for i in indices):
            return None, 0.0            # ninguna palabra conocida: no se puede opinar
        probabilidades = np.asarray(self.modelo(np.array([indices], dtype="int32"), training=False))[0]
        mejor = int(probabilidades.argmax())            # salida softmax: la más probable
        return self.intenciones[mejor], float(probabilidades[mejor])

    # --------------------------------------------------- respaldo por reglas
    def _intencion_por_palabras(self, pregunta):
        """Deduce la intención buscando palabras clave."""
        p = _normalizar(pregunta)
        pide_momento = any(palabra in p for palabra in
                           ("cuando", "segundo", "momento", "minuto", "que hora", "instante"))
        temas = {
            "colisiones": ("colision", "choque", "choco", "chocar", "accidente", "impacto",
                           "estrell", "atropell"),
            "caidas": ("caida", "cayo", "caer", "cae"),
            "riesgos": ("riesgo", "peligro", "casi"),
        }
        for tema, palabras in temas.items():
            if any(palabra in p for palabra in palabras):
                return f"momento_{tema}" if pide_momento else f"contar_{tema}"
        if any(palabra in p for palabra in ("bus", "camion", "tractomula", "volqueta", "tipo de vehiculo")):
            return "tipo_vehiculo"          # el sistema no separa los tipos de vehículo
        objetos = {
            "peatones": ("peaton", "persona", "gente"), "motos": ("moto",),
            "bicicletas": ("bici", "ciclista"),
            "vehiculos": ("vehiculo", "carro", "auto", "coche"),
        }
        for objeto, palabras in objetos.items():
            if any(palabra in p for palabra in palabras):
                return f"contar_{objeto}"
        if "ultim" in p:
            return "ultimo_evento"
        if "primer" in p:
            return "primer_evento"
        if any(palabra in p for palabra in ("grave", "alto", "urgente", "importante")):
            return "eventos_graves"
        if any(palabra in p for palabra in ("dura", "fotograma", "largo")):
            return "duracion"
        if any(palabra in p for palabra in ("evento", "alerta", "cuantos", "cuantas")):
            return "listar_eventos"
        return "resumen"

    def _responder_intencion(self, intencion):
        """Respuesta armada sin el modelo de lenguaje: la primera forma válida."""
        return rellenar(respuestas_validas(intencion, self.cuenta)[0], self.resumen, self.eventos)


def conversar(asistente):
    """Chat por consola. Línea vacía o 'salir' para terminar."""
    print("\nAsistente de eventos. Pregunta sobre el video (escribe 'salir' para terminar).")
    while True:
        try:
            pregunta = input("\nPregunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not pregunta or pregunta.lower() == "salir":
            break
        resultado = asistente.responder(pregunta)
        origen = {"llm": "modelo de lenguaje", "transformer": "clasificador",
                  "reglas": "palabras clave"}[resultado["fuente"]]
        print(f"Respuesta ({origen}): {resultado['respuesta']}")
    for nombre, error in (("modelo de lenguaje", asistente.error_llm),
                          ("clasificador de intenciones", asistente.error_modelo)):
        if error:
            print(f"\nNota: el {nombre} no está disponible ({error}).")
