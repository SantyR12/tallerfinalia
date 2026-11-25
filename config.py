"""
Parámetros del sistema de detección de accidentes de tránsito.

Todos los umbrales están aquí para poder ajustarlos sin tocar la lógica.
Las velocidades y distancias se miden en "tamaños de objeto": los píxeles se
dividen entre el tamaño de la caja del objeto. Así el mismo umbral sirve para
un carro cercano (caja grande) y uno lejano (caja pequeña), y para videos de
distinta resolución.

El proyecto NO usa modelos preentrenados: la red convolucional y el
Transformer se entrenan desde cero con los scripts de la carpeta entrenamiento/.
"""

# ----------------------------------------------------------------------------
# Clases
# ----------------------------------------------------------------------------
# Clases con las que se entrena la CNN. "fondo" sirve para descartar lo que se
# mueve pero no es un usuario de la vía (sombras, árboles, reflejos, "fantasmas").
CLASES = ["bicicleta", "bus", "camion", "carro", "fondo", "motocicleta", "persona"]
# Lo que el sistema reporta. La red aprende a separar carro, bus y camión, pero
# en videos de cámaras distintas a las del dataset confunde esos tres tipos
# entre sí; en cambio casi nunca los confunde con una persona o una moto. Por
# eso sus tres probabilidades se suman en un solo grupo: "vehiculo".
GRUPOS = {"carro": "vehiculo", "bus": "vehiculo", "camion": "vehiculo"}
NOMBRE_CLASE = {"vehiculo": "vehículo"}     # cómo se escribe en los mensajes
VEHICULOS_MOTOR = {"vehiculo", "motocicleta"}
VEHICULOS_LIVIANOS = {"motocicleta", "bicicleta"}

# ----------------------------------------------------------------------------
# Paso 1. Detección de movimiento (sustracción de fondo, sin aprendizaje)
# ----------------------------------------------------------------------------
ANCHO_DE_TRABAJO = 640          # el movimiento se busca en la imagen reducida a este ancho
SUAVIZADO = 5                   # tamaño del filtro gaussiano previo (impar)
HISTORIAL_FONDO = 500           # fotogramas que recuerda el modelo de fondo
UMBRAL_FONDO = 25               # qué tan distinto debe ser un píxel para ser "movimiento"
DESCARTAR_SOMBRAS = True        # no contar como movimiento las sombras que proyectan los objetos
SEGUNDOS_HASTA_FONDO = 5        # un objeto que se detiene sigue visible este tiempo antes de
                                # pasar a ser parte del fondo (alcanza para ver un choque)
PROPORCION_FONDO = 0.7          # peso que debe reunir un aspecto de la escena para ser "el fondo";
                                # con valores altos los fondos viejos tardan mucho en olvidarse
SEGUNDOS_APRENDIZAJE = 1.0      # al empezar, el fondo se aprende rápido durante este tiempo
CAMBIO_GLOBAL = 0.40            # si más de esta fracción de la imagen "se mueve"...
SEGUNDOS_CAMBIO_GLOBAL = 0.5    # ...durante este tiempo, la cámara se movió o cambió la luz:
                                # se descarta el fondo y se aprende de nuevo
MARGEN_BORDE = 2                # píxeles; una caja a esta distancia del borde se ve cortada
AREA_MINIMA = 0.0006            # fracción del fotograma; manchas más pequeñas son ruido
AREA_MAXIMA = 0.50              # una mancha de medio fotograma es un cambio de luz, no un objeto
RELLENO_MINIMO = 0.30           # la mancha debe llenar al menos esta parte de su caja

# ----------------------------------------------------------------------------
# Paso 2. Clasificación (CNN propia, entrenada desde cero)
# ----------------------------------------------------------------------------
TAM_IMAGEN = 64                 # los recortes se llevan a 64 x 64 píxeles
NITIDEZ_MINIMA = 12             # al entrenar, cada imagen se emborrona hasta este tamaño como mínimo
RUTA_CNN = "modelos/clasificador_objetos.keras"
RUTA_CLASES = "modelos/clases.json"
CONFIANZA_MINIMA = 0.50         # por debajo, el voto de ese fotograma no cuenta
VOTOS_PARA_MOSTRAR = 3          # votos antes de dar por buena la clase de un objeto

# ----------------------------------------------------------------------------
# Paso 3. Seguimiento (seguidor propio) y velocidad
# ----------------------------------------------------------------------------
IOU_EMPAREJAR = 0.10            # solape mínimo para decir "es el mismo objeto"
PERDIDA_MAXIMA_S = 2.0          # tiempo que se conserva un objeto sin verlo
RECORRIDO_MINIMO = 0.50         # un objeto se confirma cuando se desplaza medio tamaño propio
SALTO_DE_AREA = 1.4             # si el área cambia más que esto de golpe, la caja es inestable
VENTANA_VELOCIDAD_S = 0.20      # la velocidad se promedia sobre este tiempo
MEMORIA_PISTA_S = 2.0           # historial que se guarda de cada objeto
FRAMES_MINIMOS_CONTEO = 8       # un objeto cuenta si se vio al menos estas veces

# ----------------------------------------------------------------------------
# Colisión
# ----------------------------------------------------------------------------
SOLAPE_MINIMO = 0.15            # intersección / área de la caja más pequeña
TOLERANCIA_PROFUNDIDAD = 0.45   # diferencia permitida entre las bases de las cajas
V_CIERRE_MINIMA = 0.60          # velocidad de acercamiento (tamaños por segundo)
V_MOVIMIENTO = 0.30             # por debajo de esto el objeto se considera quieto
CONFIRMACION_S = 0.30           # espera tras el primer contacto antes de confirmar
VENTANA_CONFIRMACION_S = 1.50   # tiempo máximo, desde el contacto, para ver el cambio brusco
FRENADO_MINIMO = 0.50           # caída de velocidad: 0.5 = pierde la mitad
GIRO_MINIMO_GRADOS = 45.0       # o cambio brusco de dirección
VARIACION_ESTABLE = 0.20        # dos objetos fundidos "quedaron quietos" si su caja casi no cambia
RIESGO_MEMORIA_S = 2.5          # una alerta de riesgo "respalda" un contacto durante este tiempo
FUSION_SOSTENIDA_S = 1.2        # tras una alerta de riesgo, manchas fundidas este tiempo = choque
IMPACTO_V_MINIMA = 1.50         # frenada imposible: venía al menos a esta velocidad (tamaños/s)...
IMPACTO_ESPERA_S = 0.40         # ...y este tiempo después está casi quieto

# ----------------------------------------------------------------------------
# Situación de riesgo (casi colisión)
# ----------------------------------------------------------------------------
TTC_UMBRAL_S = 1.0              # tiempo estimado para el contacto
DISTANCIA_CRUCE = 0.35          # qué tan cerca pasarían (en tamaños)

# ----------------------------------------------------------------------------
# Caída de una persona
# ----------------------------------------------------------------------------
ASPECTO_DE_PIE = 0.75           # ancho/alto de una persona de pie (menor a esto)
ASPECTO_CAIDA = 1.30            # ancho/alto de una persona en el suelo (mayor)
CAIDA_ALTO_MAXIMO = 0.70        # al caer, el alto de la caja baja al menos a esta fracción...
CAIDA_ANCHO_MINIMO = 1.20       # ...el ancho crece al menos en esta proporción...
CAIDA_CAMBIO_AREA = 2.0         # ...y el área no cambia más que este factor
VENTANA_CAIDA_S = 1.0           # el cambio debe ocurrir dentro de este tiempo
ALTURA_MINIMA_PERSONA = 40      # píxeles; personas más pequeñas son poco fiables

# ----------------------------------------------------------------------------
# Alertas
# ----------------------------------------------------------------------------
ENFRIAMIENTO_S = 3.0            # no repetir la misma alerta antes de este tiempo
DURACION_ALERTA_S = 2.0         # tiempo que la alerta queda visible en pantalla
CARPETA_SALIDAS = "salidas"

# ----------------------------------------------------------------------------
# Videos de demostración
# ----------------------------------------------------------------------------
# Aparecen en la lista de la aplicación web (solo los que existan en la carpeta
# data/). Cada uno indica su archivo y cuántos fotogramas analizar (0 = todo).
# Los videos no se suben al repositorio: se copian a mano en data/.
VIDEOS_DEMO = {
    "Choque en una intersección": ("data/demo_choque_interseccion.mp4", 0),
    "Choque en una avenida": ("data/demo_choque_avenida.mp4", 0),
    "Tráfico normal (sin accidentes)": ("data/demo_trafico_normal.mp4", 240),
}

# ----------------------------------------------------------------------------
# Clasificador de intenciones (Transformer codificador propio, desde cero)
# ----------------------------------------------------------------------------
RUTA_PREGUNTAS = "datos/preguntas_asistente.csv"
RUTA_ASISTENTE = "modelos/asistente_transformer.keras"
RUTA_VOCABULARIO = "modelos/asistente_vocabulario.json"
LARGO_PREGUNTA = 16             # palabras por pregunta (se rellena o se recorta)
CONFIANZA_ASISTENTE = 0.70      # por debajo, se responde por palabras clave

# ----------------------------------------------------------------------------
# Modelo de lenguaje propio (Transformer decodificador tipo GPT, desde cero)
# ----------------------------------------------------------------------------
RUTA_CORPUS = "datos/corpus_asistente.csv"
RUTA_LLM = "modelos/modelo_lenguaje.keras"
RUTA_VOCABULARIO_LLM = "modelos/modelo_lenguaje_vocabulario.json"
LARGO_SECUENCIA = 56            # fichas por ejemplo: datos + pregunta + respuesta
DIM_LLM = 64                    # tamaño del vector de cada ficha
CABEZAS_LLM = 4                 # cabezas de atención
BLOQUES_LLM = 2                 # bloques decodificadores apilados
MAX_FICHAS_RESPUESTA = 45       # tope de fichas que puede escribir
