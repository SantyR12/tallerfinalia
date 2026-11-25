"""
Aplicación web del proyecto.

    streamlit run app.py

Permite elegir un video de demostración, subir uno propio o usar la cámara;
ver el análisis en vivo con las alertas, revisar la tabla de eventos y
consultar al asistente.

Cómo funciona Streamlit: cada vez que se pulsa un botón o se escribe algo, este
archivo se ejecuta de nuevo de arriba abajo. Lo que debe conservarse entre una
ejecución y la siguiente se guarda en st.session_state (por ejemplo los mensajes
del chat) o se marca con @st.cache_resource (las redes, que tardan en cargar).
"""

import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

import config
from src.alertas import cargar_resultados
from src.asistente import Asistente

st.set_page_config(page_title="Detección de accidentes de tránsito", layout="wide")
st.title("Detección de accidentes de tránsito y comportamientos peligrosos")
st.caption("Detección de movimiento, seguimiento y reglas propias · red convolucional, "
           "Transformer y modelo de lenguaje entrenados desde cero")


# ------------------------------------------------------------ redes (en caché)
@st.cache_resource(show_spinner="Cargando la red convolucional...")
def obtener_clasificador():
    """Carga la CNN una sola vez; las siguientes ejecuciones reutilizan la misma."""
    from src.clasificador import ClasificadorCNN
    return ClasificadorCNN()


@st.cache_resource(show_spinner="Cargando el asistente...")
def obtener_asistente(firma, usar_transformer):
    """`firma` cambia cuando hay un análisis nuevo, para reconstruir el asistente."""
    resumen, eventos = cargar_resultados(config.CARPETA_SALIDAS)
    return Asistente(resumen, eventos, usar_transformer=usar_transformer, usar_llm=usar_transformer)


def videos_de_demostracion():
    """Los videos de config.VIDEOS_DEMO que de verdad están en la carpeta data/."""
    return {nombre: (ruta, fotogramas) for nombre, (ruta, fotogramas) in config.VIDEOS_DEMO.items()
            if Path(ruta).exists()}


# ------------------------------------------------------------------ controles
# Barra lateral: de dónde sale el video y con qué opciones se analiza
demos = videos_de_demostracion()
with st.sidebar:
    st.header("Fuente de video")
    opciones = (["Video de demostración"] if demos else []) + ["Subir un video", "Cámara web"]
    origen = st.radio("Origen", opciones, label_visibility="collapsed")
    archivo = None          # video subido por el usuario
    demo = None             # nombre del video de demostración elegido
    if origen == "Video de demostración":
        demo = st.selectbox("Video", list(demos))
        limite = demos[demo][1]             # fotogramas a analizar (0 = todo el video)
    elif origen == "Subir un video":
        archivo = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"])
        limite = st.number_input("Máximo de fotogramas (0 = todo el video)", 0, 100000, 0, step=100)
    else:
        limite = st.number_input("Fotogramas a capturar", 50, 5000, 300, step=50)
    usar_transformer = st.checkbox("Asistente con las redes entrenadas", value=True,
                                   help="Modelo de lenguaje y clasificador de intenciones propios. "
                                        "Si se desactiva, responde solo con palabras clave.")
    iniciar = st.button("Analizar", type="primary", width="stretch")

# Zona principal: el video a la izquierda y las alertas a la derecha
columna_video, columna_alertas = st.columns([2, 1])
visor = columna_video.empty()               # espacio donde se va pintando cada fotograma
columna_alertas.subheader("Alertas")
caja_alertas = columna_alertas.container(height=420)

# ------------------------------------------------------------------- análisis
if iniciar:
    from src.detector import Detector       # se importa aquí para que la página abra rápido
    from src.video import procesar

    try:
        detector = Detector(clasificador=obtener_clasificador())
    except FileNotFoundError as error:      # falta entrenar la CNN
        st.error(str(error))
        st.stop()

    # `fuente` es lo que recibe OpenCV: la ruta de un archivo o el número de la cámara
    if origen == "Video de demostración":
        fuente = demos[demo][0]
    elif origen == "Subir un video":
        if archivo is None:
            st.warning("Sube primero un video.")
            st.stop()
        # OpenCV necesita un archivo en el disco: el video subido se copia a uno temporal
        temporal = tempfile.NamedTemporaryFile(delete=False, suffix=Path(archivo.name).suffix)
        temporal.write(archivo.read())
        temporal.close()
        fuente = temporal.name
    else:
        fuente = 0                          # 0 = cámara principal del computador

    # procesar() entrega un fotograma anotado por vuelta, junto con las alertas nuevas
    pasos = procesar(fuente, max_frames=limite or None, detector=detector)
    try:
        for anotado, nuevos, analizador in pasos:
            # OpenCV trabaja en BGR y Streamlit espera RGB: se convierte antes de mostrar
            visor.image(cv2.cvtColor(anotado, cv2.COLOR_BGR2RGB), width="stretch")
            for evento in nuevos:
                mensaje = f"**{evento.tipo.upper()}** · segundo {evento.tiempo_s}  \n{evento.descripcion}"
                # Rojo para el nivel ALTO (colisión, caída) y amarillo para el MEDIO (riesgo)
                (caja_alertas.error if evento.nivel == "ALTO" else caja_alertas.warning)(mensaje)
    finally:
        pasos.close()                       # al cerrar, procesar() guarda los resultados
    st.session_state["analisis"] = st.session_state.get("analisis", 0) + 1
    st.session_state["mensajes"] = []       # chat nuevo para el video nuevo
    st.success("Análisis terminado.")

# ----------------------------------------------------------------- resultados
# Se muestran los del último análisis guardado en la carpeta salidas/
if (Path(config.CARPETA_SALIDAS) / "resumen.json").exists():
    resumen, eventos = cargar_resultados(config.CARPETA_SALIDAS)
    objetos = resumen["objetos"]

    st.subheader("Resultados del último análisis")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Vehículos", objetos.get("vehiculo", 0))
    m2.metric("Motocicletas", objetos.get("motocicleta", 0))
    m3.metric("Bicicletas", objetos.get("bicicleta", 0))
    m4.metric("Peatones", objetos.get("persona", 0))
    m5.metric("Colisiones y caídas", resumen["eventos"]["colision"] + resumen["eventos"]["caida"])
    m6.metric("Situaciones de riesgo", resumen["eventos"]["riesgo"])

    if eventos:
        tabla = pd.DataFrame(eventos)[["tipo", "nivel", "tiempo_s", "clases", "descripcion"]]
        tabla["clases"] = tabla["clases"].apply(" / ".join)
        st.dataframe(tabla, width="stretch", hide_index=True)
        capturas = sorted((Path(config.CARPETA_SALIDAS) / "capturas").glob("*.jpg"))
        if capturas:
            with st.expander("Capturas de los eventos"):
                st.image([str(c) for c in capturas], width=320)
    else:
        st.info("No se registraron eventos en este video.")

    # ---------------------------------------------------------- asistente
    st.subheader("Asistente")
    asistente = obtener_asistente(st.session_state.get("analisis", 0), usar_transformer)
    with st.expander("Informe que lee el asistente"):
        st.write(asistente.informe)

    # Se vuelven a pintar los mensajes anteriores (la página se ejecuta de nuevo en cada pregunta)
    for mensaje in st.session_state.setdefault("mensajes", []):
        st.chat_message(mensaje["rol"]).write(mensaje["texto"])

    pregunta = st.chat_input("Pregunta sobre el video, por ejemplo: ¿cuántas colisiones hubo?")
    if pregunta:
        st.chat_message("user").write(pregunta)
        resultado = asistente.responder(pregunta)
        # Cada respuesta dice cuál de las tres vías la produjo
        origen_respuesta = {"llm": "escrita por el modelo de lenguaje",
                            "transformer": "armada con la intención del clasificador",
                            "reglas": "armada por palabras clave"}[resultado["fuente"]]
        texto = f"{resultado['respuesta']}\n\n_Respuesta {origen_respuesta}._"
        st.chat_message("assistant").write(texto)
        st.session_state["mensajes"] += [
            {"rol": "user", "texto": pregunta},
            {"rol": "assistant", "texto": texto},
        ]
        if usar_transformer:                # si alguna red no cargó, se avisa por qué
            for nombre, error in (("modelo de lenguaje", asistente.error_llm),
                                  ("clasificador", asistente.error_modelo)):
                if error:
                    st.caption(f"El {nombre} no está disponible: {error}")
else:
    st.info("Elige un video y pulsa **Analizar**.")
