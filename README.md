# Detección de accidentes de tránsito y comportamientos peligrosos

Proyecto final de Inteligencia Artificial (Proyecto 2). El sistema analiza un video de tránsito o la cámara web y:

1. **Detecta** vehículos (carros, buses y camiones como un solo grupo), motocicletas, bicicletas y peatones.
2. **Sigue** a cada objeto de un fotograma al siguiente y le asigna un número.
3. **Identifica** posibles colisiones, situaciones de riesgo y caídas a partir de las trayectorias.
4. **Genera alertas** visuales sobre el video y guarda una captura de cada evento.
5. Ofrece un **asistente** que responde preguntas sobre lo ocurrido.

**No se usa ningún modelo preentrenado ni ningún servicio externo** (nada de YOLO, ResNet, ChatGPT, Gemini o Claude). Las tres redes del proyecto están definidas en este repositorio y se entrenan desde cero con los scripts de `entrenamiento/`:

- una **red convolucional (CNN)** que clasifica los objetos;
- un **modelo de lenguaje** (Transformer decodificador tipo GPT, en pequeño) que escribe las respuestas del asistente;
- un **Transformer codificador** que clasifica la intención de la pregunta y controla lo que escribe el modelo de lenguaje.

El seguimiento también es código propio.

Integrantes: Santiago Ramos, Paulo Tibanta, Nelson Barrera y Julian Dorado.

## Guía rápida

Para probar el proyecto de principio a fin, en este orden:

```bash
# 1. Descargar el repositorio y entrar a la carpeta
git clone https://github.com/SantyR12/tallerfinalia.git
cd tallerfinalia

# 2. Crear el entorno e instalar las librerías (Python 3.10 o superior)
python3 -m venv .venv
source .venv/bin/activate            # en Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Comprobar la instalación: debe terminar con "58 passed"
python -m pytest tests -q

# 4. Entrenar las redes (los modelos no vienen en el repositorio)
python entrenamiento/generar_preguntas.py
python entrenamiento/entrenar_asistente.py      # clasificador de intenciones, 1 minuto
python entrenamiento/generar_corpus.py
python entrenamiento/entrenar_llm.py            # modelo de lenguaje, unos 5 minutos
python entrenamiento/preparar_dataset.py --origen RUTA/MIO-TCD-Classification/train
python entrenamiento/entrenar_cnn.py            # CNN; requiere descargar el dataset

# 5. Copiar un video de cámara fija a data/ y abrir la aplicación
streamlit run app.py
```

En el navegador: elegir un video de demostración o subir uno, pulsar **Analizar** y, al terminar, preguntar al asistente en el chat.

- **Los modelos entrenados no se incluyen en el repositorio** (están en `.gitignore`): el paso 4 es obligatorio la primera vez. Los dos del asistente tardan unos 6 minutos en total y no descargan nada; el de la CNN necesita el dataset de imágenes.
- **Si falta alguna red**, el programa lo dice y muestra el comando para entrenarla.
- Con Anaconda, el paso 2 puede hacerse con `conda create -n ia python=3.11` y `conda activate ia` antes del `pip install`.

Las secciones siguientes explican cada paso con más detalle.

## Cómo funciona

```
video o cámara
      │
      ▼
 Sustracción de fondo ──► cajas de lo que se mueve         src/movimiento.py   (OpenCV clásico)
      │
      ▼
 Seguidor propio ───────► cada objeto conserva su ID       src/seguimiento.py
      │
      ▼
 CNN propia ────────────► qué es cada objeto               src/clasificador.py
      │
      ▼
 Reglas de trayectoria ─► colisión / riesgo / caída        src/eventos.py
      │
      ├──► alertas en pantalla, capturas, CSV y JSON       src/alertas.py
      │
      ▼
 Asistente              ─► modelo de lenguaje propio: escribe la respuesta
                           clasificador propio: controla que sea coherente
                           src/modelo_lenguaje.py, src/transformer.py, src/asistente.py
```

| Parte | Qué es | ¿Se entrena? |
|---|---|---|
| Detección de movimiento | Sustracción de fondo MOG2 y morfología (procesamiento clásico de imágenes) | No, no es un modelo de aprendizaje |
| Clasificación | CNN de 3 bloques convolucionales, 1,1 millones de parámetros | Sí, desde cero, con un dataset de imágenes |
| Seguimiento | Emparejamiento por solape (IoU) con predicción de posición | No, es un algoritmo |
| Eventos | Reglas geométricas sobre las trayectorias | No |
| Modelo de lenguaje | Transformer decodificador (tipo GPT) de 2 bloques, 107 mil parámetros | Sí, desde cero, con un CSV de 24 000 ejemplos |
| Clasificador de intenciones | Transformer codificador de 1 bloque, 15 mil parámetros | Sí, desde cero, con un CSV de 7 300 preguntas |

La explicación detallada está en [`docs/como_funciona.md`](docs/como_funciona.md).

## Instalación

Requiere Python 3.10 o superior.

```bash
python -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Entrenamiento (obligatorio antes del primer uso)

El proyecto no usa pesos de terceros ni incluye modelos ya entrenados: las tres redes se entrenan con estos scripts y quedan en `modelos/`, que no se sube al repositorio.

### 1. Clasificador de intenciones (un minuto, sin descargar nada)

```bash
python entrenamiento/generar_preguntas.py     # crea datos/preguntas_asistente.csv
python entrenamiento/entrenar_asistente.py    # crea modelos/asistente_transformer.keras
```

### 2. Modelo de lenguaje (unos cinco minutos, sin descargar nada)

```bash
python entrenamiento/generar_corpus.py        # crea datos/corpus_asistente.csv
python entrenamiento/entrenar_llm.py          # crea modelos/modelo_lenguaje.keras
```

Al terminar imprime qué porcentaje de respuestas escribe bien en casos nuevos.

### 3. CNN de objetos (necesita un dataset de imágenes)

Se necesita un dataset de **recortes de objetos de tránsito organizados en una carpeta por clase**. El recomendado es **MIO-TCD Classification**: recortes tomados por cámaras de tránsito, con las clases carro, motocicleta, peatón, bus, camión, bicicleta y fondo. La red se entrena con esas siete clases; al usarla, las probabilidades de carro, bus y camión se suman en un solo grupo, "vehículo" (ver `GRUPOS` en `config.py`). Se descarga en <https://tcd.miovision.com/challenge/dataset.html> (licencia CC BY-NC-SA 4.0).

```bash
# convierte las carpetas en un CSV (ruta, clase, conjunto) con 4000 imágenes por clase
python entrenamiento/preparar_dataset.py --origen RUTA/MIO-TCD-Classification/train

# entrena la red y guarda modelos/clasificador_objetos.keras
python entrenamiento/entrenar_cnn.py
```

El entrenamiento deja en `modelos/entrenamiento_cnn/` las curvas de pérdida y exactitud, la matriz de confusión y el reporte por clase, e imprime la exactitud con las siete clases y por grupos. Con unas 24 000 imágenes de 64 × 64 tarda entre 20 y 35 minutos en CPU; en Google Colab con GPU, mucho menos.

Si se cambia la forma de preparar las imágenes (`encuadrar` en `src/clasificador.py`) hay que volver a entrenar: la red debe ver al detectar imágenes preparadas igual que al entrenar.

Para usar otro dataset basta con que tenga una carpeta por clase; los nombres de carpeta se relacionan con las clases del proyecto en el diccionario `EQUIVALENCIAS` de `entrenamiento/preparar_dataset.py`.

## Uso

Coloca los videos de prueba en `data/`. Deben ser de **cámara fija**.

**Aplicación web**

```bash
streamlit run app.py
```

En la barra lateral se elige la fuente: un **video de demostración**, un video subido o la cámara web. Los videos de demostración son los que figuran en `VIDEOS_DEMO` (archivo `config.py`) y estén en la carpeta `data/`; como los videos no se suben al repositorio, hay que copiarlos allí con estos nombres:

| Archivo en `data/` | Qué muestra |
|---|---|
| `demo_choque_interseccion.mp4` | Choque en una intersección: alerta de riesgo y después colisión |
| `demo_choque_avenida.mp4` | Choque en una avenida: alerta de riesgo y después colisión |
| `demo_trafico_normal.mp4` | Tráfico sin accidentes: conteo de vehículos, sin alertas |

**Consola**

```bash
python main.py --video data/accidente.mp4              # analizar un video
python main.py --camara 0                              # analizar la cámara web
python main.py --video data/accidente.mp4 --asistente  # y abrir el asistente al final
python main.py --consultar salidas                     # preguntar sobre un análisis ya hecho
```

La tecla `Q` detiene el análisis. Opciones: `--sin-ventana`, `--max-frames 300`, `--sin-llm`, `--sin-transformer`.

**Resultados** (carpeta `salidas/`)

| Archivo | Contenido |
|---|---|
| `video_anotado.mp4` | El video con cajas, IDs y alertas |
| `capturas/` | Una imagen por cada evento detectado |
| `eventos.csv` y `eventos.json` | Tipo, nivel, segundo y objetos de cada evento |
| `resumen.json` | Objetos contados por clase y eventos por tipo |
| `informe.txt` | El informe en texto del análisis |

## Estructura

```
├── app.py                    Aplicación web (Streamlit)
├── main.py                   Programa de consola
├── config.py                 Todos los umbrales y rutas
├── src/
│   ├── movimiento.py         Sustracción de fondo: cajas de lo que se mueve
│   ├── seguimiento.py        Seguidor propio (IDs, predicción, fusión de manchas)
│   ├── clasificador.py       Uso de la CNN entrenada
│   ├── detector.py           Une movimiento + seguimiento + clasificación
│   ├── eventos.py            Reglas de colisión, riesgo y caída
│   ├── alertas.py            Dibujo de alertas y guardado de resultados
│   ├── video.py              Recorrido del video fotograma a fotograma
│   ├── texto.py              Texto -> números (normalizar, codificar)
│   ├── transformer.py        Capas Transformer propias y clasificador de intenciones
│   ├── lenguaje.py           Qué lee y qué escribe el modelo de lenguaje; ranuras
│   ├── modelo_lenguaje.py    Modelo de lenguaje (decodificador) y generación
│   └── asistente.py          Une modelo de lenguaje, clasificador y registro
├── entrenamiento/
│   ├── preparar_dataset.py   Carpetas de imágenes -> CSV
│   ├── entrenar_cnn.py       Define y entrena la CNN
│   ├── generar_preguntas.py  Crea el CSV de preguntas
│   ├── entrenar_asistente.py Entrena el clasificador de intenciones
│   ├── generar_corpus.py     Crea el CSV de (datos, pregunta, respuesta)
│   └── entrenar_llm.py       Entrena el modelo de lenguaje
├── datos/                    CSV de los datasets
├── modelos/                  Redes entrenadas (no se suben) y resultados del entrenamiento
├── tests/                    Pruebas automáticas
├── docs/                     Informe, presentación, guion y explicación del código
├── data/                     Videos de prueba (no se suben al repositorio)
└── salidas/                  Resultados (no se suben al repositorio)
```

## Requisitos del proyecto y dónde se cumplen

| Requisito | Dónde |
|---|---|
| Detectar vehículos, motocicletas y peatones | `src/movimiento.py` (dónde están) y CNN propia en `src/clasificador.py` (qué son) |
| Identificar posibles colisiones o caídas | `src/eventos.py` (`_revisar_colision`, `_revisar_caida`) |
| Realizar seguimiento de los objetos | `src/seguimiento.py` |
| Generar alertas ante una situación de riesgo | `src/eventos.py` (`_revisar_riesgo`) y `src/alertas.py` |
| Asistente para consultar los eventos | `src/asistente.py`, sección de chat en `app.py` |
| Redes neuronales, Transformer y modelo de lenguaje | CNN en `entrenamiento/entrenar_cnn.py`; Transformer en `src/transformer.py`; modelo de lenguaje en `src/modelo_lenguaje.py` |
| Demostración con video o cámara web | `app.py` y `main.py` |

## Pruebas

```bash
python -m pytest tests -q
```

- `test_eventos.py`: reglas con trayectorias construidas a mano (choques, cruces, adelantamientos, caídas).
- `test_deteccion.py`: de punta a punta con videos sintéticos: sustracción de fondo, seguidor y reglas (choque frontal, alcance, cruce en carriles distintos, fila de carros, frenada ante un peatón).
- `test_cnn.py`: la arquitectura, que la CNN aprende desde cero un problema sencillo y que el clasificador la usa bien.
- `test_transformer.py`: tokenización, que el clasificador aprende, que se guarda y se carga igual, y que la máscara de relleno funciona.
- `test_llm.py`: que el modelo de lenguaje aprende a escribir respuestas correctas, que la máscara causal le impide ver las palabras futuras, y que el asistente descarta lo que escribe cuando no es coherente.
- `test_asistente.py` y `test_video.py`: respuestas, alertas en pantalla y archivos de resultados.

## Evaluación con videos reales

`evaluar.py` pasa por el sistema los videos de una carpeta y compara las alertas de colisión con lo que realmente ocurre, anotado a mano en un CSV (`video,segundo_choque`; el segundo se deja vacío si no hay accidente). Un choque cuenta como detectado si hay una alerta de colisión entre un segundo antes y cuatro después del momento anotado.

`datos/choques_reales.csv` trae anotados 17 videos de choques grabados por cámaras fijas, del repositorio público [ashin-coder/vehicle-crash-detector](https://github.com/ashin-coder/vehicle-crash-detector) (carpeta `test_videos`). Los segundos son aproximados.

```bash
git clone --depth 1 https://github.com/ashin-coder/vehicle-crash-detector ../videos_choques
python evaluar.py --carpeta ../videos_choques/test_videos
```

Al final imprime cuántos choques detectó, las alertas de colisión falsas, la precisión y el recall. El script solo compara el momento de la alerta, no qué vehículos chocaron, así que hay que confirmar cada colisión mirando su captura: con los 17 videos cuenta 3 detectados, pero en `test (17).mp4` la alerta es entre dos vehículos ajenos al accidente. El resultado verificado es de 2 choques detectados de 17 y 1 alerta falsa. El resultado puede variar un poco de un computador a otro: cada instalación decodifica el video con diferencias mínimas en los píxeles, y eso basta para que el seguimiento pierda a un vehículo un instante antes o después. Por eso, cuando ya hubo una alerta de riesgo, la colisión se confirma aunque uno de los dos vehículos se haya perdido de vista (ver "Riesgo cumplido" en `docs/como_funciona.md`).

En `test (4).mp4` y `test (12).mp4` el sistema sigue a los dos vehículos, avisa del riesgo y después reporta la colisión. Son los videos de la demostración: se copian a `data/` con los nombres `demo_choque_interseccion.mp4` y `demo_choque_avenida.mp4` (deben ser los archivos originales; un video reenviado por una aplicación de mensajería queda recomprimido y el resultado puede cambiar).

```bash
cp "../videos_choques/test_videos/test (4).mp4"  data/demo_choque_interseccion.mp4
cp "../videos_choques/test_videos/test (12).mp4" data/demo_choque_avenida.mp4
python main.py --video data/demo_choque_avenida.mp4 --asistente
```

## Ajuste

Todos los umbrales están en `config.py`.

| Parámetro | Efecto |
|---|---|
| `UMBRAL_FONDO` | Más alto: menos sensible al movimiento (menos ruido, puede perder objetos poco contrastados) |
| `AREA_MINIMA` | Tamaño mínimo de una mancha para ser objeto; subirlo quita ruido, bajarlo detecta objetos lejanos |
| `SEGUNDOS_HASTA_FONDO` | Cuánto tiempo sigue visible un objeto que se detiene |
| `V_CIERRE_MINIMA` | Más alto: menos falsas alarmas de colisión; puede ignorar choques lentos |
| `TTC_UMBRAL_S` | Con cuánta anticipación se avisa del riesgo |
| `CONFIANZA_MINIMA` | Seguridad que debe tener la CNN para que su voto cuente |

## Limitaciones

- **Cámara fija.** La sustracción de fondo no funciona con cámara en movimiento.
- **Solo ve lo que se mueve.** Un vehículo parqueado desde el inicio es parte del fondo; uno que se detiene sigue visible unos 5 segundos.
- **El primer segundo no cuenta.** Se usa para aprender el fondo. Lo mismo ocurre durante un segundo si la cámara se mueve o la luz cambia de golpe.
- **Los objetos cortados por el borde de la imagen** no se clasifican ni generan alertas: su caja no es confiable.
- **Choques reales.** La regla de colisión entre dos objetos exige haber seguido a los dos antes del contacto. En choques reales, que suelen durar menos de un segundo, eso se cumple solo a veces: ver "Evaluación con videos reales".
- **Objetos lejanos.** Cerca del horizonte los vehículos son manchas pequeñas que se juntan entre sí; ahí el sistema detecta menos y puede dar alguna alerta falsa. Las reglas descuentan la perspectiva (dos carros que se alejan por la misma vía se juntan en la imagen sin acercarse de verdad), pero con cajas tan pequeñas la corrección no es exacta.
- **Objetos juntos se ven como uno.** Cuando dos objetos se tocan, sus manchas se funden. El sistema lo aprovecha para detectar choques, pero en tráfico denso varios vehículos pueden quedar agrupados.
- **Sombras y cambios de luz** producen manchas falsas; la clase "fondo" de la CNN y el filtro de desplazamiento mínimo descartan la mayoría, no todas.
- **La CNN es pequeña** y se entrena con una fracción del dataset: se equivoca más que un detector moderno. En videos de cámaras distintas a las del dataset no distingue de forma confiable entre carro, bus y camión; por eso el sistema los reporta como un solo grupo, "vehículo", y el asistente lo dice cuando se le pregunta por buses o camiones.
- El sistema marca eventos **posibles**: es una ayuda para quien revisa el video, no un dictamen.
- **El modelo de lenguaje es pequeño** (107 mil parámetros; un LLM comercial tiene miles de millones). Solo sabe hablar de los eventos de un video de tránsito, con frases parecidas a las de su corpus. No escribe las cifras: escribe ranuras que el programa llena con los datos del registro.
- El asistente entiende las preguntas parecidas a las de su dataset; si lo que escribe el modelo de lenguaje no es coherente con la pregunta y los datos, se usa una respuesta de respaldo.
