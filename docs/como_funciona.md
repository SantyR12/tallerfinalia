# Cómo funciona el sistema, paso a paso

Este documento explica cada parte del proyecto con el nivel de detalle necesario para sustentarlo. Al final hay una lista de preguntas frecuentes con su respuesta.

Regla del proyecto: **ningún modelo preentrenado ni servicio externo**. Las tres redes neuronales (la CNN, el modelo de lenguaje y el clasificador de intenciones) están definidas en este repositorio y se entrenan desde cero.

## 1. Recorrido de un fotograma

Para cada fotograma ocurre lo siguiente (`src/video.py`, función `procesar`):

1. `DetectorMovimiento` encuentra las zonas que se mueven y devuelve sus cajas.
2. `SeguidorObjetos` decide a qué objeto ya conocido corresponde cada caja y le mantiene su ID.
3. `ClasificadorCNN` recorta cada caja y la red dice qué es.
4. `AnalizadorEventos` agrega las cajas al historial de cada objeto y aplica las reglas.
5. `dibujar` pinta cajas, IDs, contador y, si hay un evento reciente, la franja de alerta.

Al terminar se guardan `eventos.csv`, `eventos.json`, `resumen.json` e `informe.txt`.

## 2. Encontrar lo que se mueve: sustracción de fondo

Archivo `src/movimiento.py`. No es aprendizaje automático: es procesamiento clásico de imágenes, como el del taller de OpenCV.

0. **Reducción** de la imagen a 640 píxeles de ancho: los filtros se miden en píxeles y así actúan igual en cualquier resolución.
1. **Suavizado** con un filtro gaussiano, para que el ruido de la cámara no cuente como movimiento, e **igualación del brillo**: las cámaras ajustan solas la exposición y toda la imagen se aclara u oscurece; cada fotograma se lleva al brillo del fondo aprendido. La corrección es la mediana del cociente entre el fondo y el fotograma: un cambio de exposición altera todos los píxeles en la misma proporción, y un objeto, por grande que sea, solo los que tapa.
2. **MOG2** (`cv2.createBackgroundSubtractorMOG2`). Para cada píxel mantiene un modelo estadístico (una mezcla de gaussianas) de los colores que ha tenido. Si el color actual no encaja en ese modelo, el píxel es primer plano. El resultado es una máscara: 255 = movimiento, 127 = sombra, 0 = fondo.
3. **Umbral** a 200 para descartar las sombras.
4. **Morfología.** Apertura (erosión y luego dilatación) para borrar puntos sueltos; cierre (dilatación y luego erosión) para rellenar huecos dentro de un objeto.
5. **Contornos** (`cv2.findContours`) y su caja envolvente (`cv2.boundingRect`).
6. **Filtros.** Se descartan las manchas muy pequeñas (ruido), las enormes (cambio de luz) y las que llenan menos del 30 % de su caja (líneas delgadas).

**Cuánto tarda algo en volverse fondo.** La tasa de aprendizaje se fija para que un objeto que se detiene tarde unos 5 segundos en pasar a ser parte del fondo (`SEGUNDOS_HASTA_FONDO`). Ese tiempo alcanza para ver dos vehículos que chocaron y quedaron quietos, y no es tan largo como para que un error del fondo dure medio video. `PROPORCION_FONDO` = 0,7 hace que el aspecto viejo de la escena se olvide poco después de aprender el nuevo.

**Vehículos que se detienen (opción apagada).** Un carro que espera en un cruce se vuelve fondo a los 5 segundos y puede desaparecer antes de recibir el golpe. Para ese caso existe `PROTECCION_S` en `config.py`: con el valor 10, el fondo deja de actualizarse dentro de las cajas de los vehículos ya reconocidos durante 10 segundos desde su último movimiento. Viene en 0 (apagada) porque en un trancón hace daño: toda la fila detenida sigue a la vista y cada carro que frena detrás del último se confunde con un choque por alcance. En las pruebas, encendida subió de 5 a 19 las alertas en un video de trancón sin ningún accidente.

**Primer segundo.** Al empezar, el fondo se aprende rápido durante un segundo y en ese tramo no se entregan detecciones.

**Cambios de toda la imagen.** Si más del 40 % de la imagen aparece "en movimiento" durante medio segundo, no son objetos: la cámara se movió o cambió la luz. El fondo se descarta y se aprende de nuevo.

## 3. Seguimiento propio

Archivo `src/seguimiento.py`. En cada fotograma:

1. **Predicción.** Cada objeto conocido se desplaza con su velocidad: ahí debería estar ahora.
2. **Emparejamiento.** Se calcula el IoU (intersección sobre unión) entre cada predicción y cada caja nueva, y se asignan parejas de mayor a menor IoU. Es un emparejamiento voraz.
3. **Objetos nuevos.** Las cajas sin pareja reciben un ID nuevo.
4. **Confirmación.** Un objeto solo se reporta cuando se ha desplazado al menos medio tamaño propio. Así se ignoran las manchas que parpadean en un mismo sitio.
5. **Objetos perdidos.** Un objeto sin pareja se conserva hasta 2 segundos con su posición estimada.
6. **Fusión.** Si la posición estimada de un objeto perdido queda dentro de la caja de otro, sus manchas se fundieron: se marca como "oculto en" ese objeto. Esta es la señal que usan las reglas de colisión.

La velocidad de cada objeto es un promedio móvil del desplazamiento de su centro. Cuando el área de la caja cambia de golpe (una fusión o una separación) ese desplazamiento no es movimiento real y no se usa.

## 4. La CNN propia

Definida en `entrenamiento/entrenar_cnn.py`, función `construir_cnn`.

| Capa | Salida | Parámetros |
|---|---|---|
| Entrada | 64 × 64 × 3 | — |
| Aumento de datos (volteo, desplazamiento, zoom, brillo) | 64 × 64 × 3 | 0 |
| Conv2D(32, 3×3, ReLU) | 64 × 64 × 32 | 896 |
| MaxPooling 2×2 | 32 × 32 × 32 | 0 |
| Conv2D(64, 3×3, ReLU) | 32 × 32 × 64 | 18 496 |
| MaxPooling 2×2 | 16 × 16 × 64 | 0 |
| Conv2D(128, 3×3, ReLU) | 16 × 16 × 128 | 73 856 |
| MaxPooling 2×2 | 8 × 8 × 128 | 0 |
| Flatten | 8192 | 0 |
| Dropout(0.4) | 8192 | 0 |
| Dense(128, ReLU) | 128 | 1 048 704 |
| Dense(7, softmax) | 7 | 903 |
| **Total** | | **1 142 855** |

Cómo se calculan: una convolución tiene (alto × ancho × canales de entrada + 1) × filtros parámetros; por ejemplo (3·3·3 + 1) × 32 = 896. Una capa densa tiene entradas × neuronas + neuronas; por ejemplo 8192 × 128 + 128 = 1 048 704. Las convoluciones usan `padding="same"`, por eso no reducen el tamaño; lo reduce el pooling.

**Preparación de las imágenes (`encuadrar`, en `src/clasificador.py`).** Cada recorte se lleva a 64 × 64 sin deformarlo: se escala hasta que el lado más largo mida 64 y lo que falta se rellena con gris. Así una persona sigue viéndose alta y angosta y un carro, ancho. La misma función se usa al entrenar y al detectar.

**Entrenamiento.** Optimizador Adam, función de costo `sparse_categorical_crossentropy` (clasificación multiclase con etiquetas enteras), lotes de 64, hasta 40 épocas con parada temprana (se detiene si la pérdida de validación no mejora en 6 épocas y recupera los mejores pesos). Datos: 70 % entrenamiento, 15 % validación, 15 % prueba. Si una clase tiene menos imágenes, sus errores pesan más en la pérdida.

**Aumento de nitidez (`bajar_nitidez`).** En cada época, cada imagen de entrenamiento se reduce a un tamaño al azar y se devuelve a 64 × 64, de modo que queda más o menos borrosa. En el dataset los peatones son recortes pequeños (borrosos al ampliarlos) y los buses son grandes (nítidos); sin este aumento la red aprende el atajo "nítido = bus o camión" y falla con cámaras de otra resolución.

**Contra el sobreajuste:** aumento de datos, Dropout y parada temprana.

**De siete clases a cinco grupos.** La red se entrena con siete clases, pero el sistema reporta cinco grupos: vehículo, motocicleta, bicicleta, persona y fondo. Las probabilidades de carro, bus y camión se suman en "vehículo" (`GRUPOS` en `config.py`). La razón: en videos de cámaras distintas a las del dataset la red confunde esos tres tipos entre sí, pero casi nunca los confunde con una persona o una moto.

**La clase "fondo".** El dataset incluye recortes sin ningún objeto. Con ellos la red aprende a reconocer lo que se mueve pero no es un usuario de la vía (sombras, vegetación, reflejos), y esas manchas se descartan.

**Votación.** La clase de un objeto no se decide con un solo fotograma: cada fotograma en que la red supera el 50 % de confianza aporta un voto y gana la mayoría. Un error aislado no cambia la clase.

## 5. Velocidad en "tamaños por segundo"

La velocidad de un objeto es el desplazamiento del centro de su caja en los últimos 0,2 segundos. Como un mismo carro recorre más píxeles cerca de la cámara que lejos, se divide entre el tamaño de la caja (la raíz de su área). Así un umbral como "0,6 tamaños por segundo" significa lo mismo para un objeto cercano que para uno lejano.

## 6. Las reglas de eventos

Las reglas no se aplican a un objeto cuya caja toca el borde de la imagen. Un objeto que entra o sale se ve a medias: su caja se encoge contra el borde, su centro parece frenar y el que viene detrás parece alcanzarlo. Por la misma razón esos objetos tampoco se clasifican hasta que se ven completos.

### Colisión (`_revisar_colision`)

1. **Venían acercándose.** Antes del contacto, la velocidad de acercamiento es al menos `V_CIERRE_MINIMA`, y los dos objetos ya tenían historial.
2. **Contacto.** Sus manchas se funden en una (lo habitual) o sus cajas se solapan.
3. **Misma profundidad.** Antes del contacto, las bases de las dos cajas estaban a una altura parecida en la imagen. Si un objeto está más lejos su base queda más arriba: se están tapando, no chocando.
4. **Desenlace**, observado entre 0,3 y 1,5 segundos después del contacto:
   - siguen fundidos y la mancha conjunta, con tamaño estable, **queda quieta**: chocaron y quedaron detenidos;
   - siguen fundidos y **el que estaba quieto avanza**: lo empujaron (alcance por detrás); o
   - se separan y al menos uno **frenó de golpe, cambió de dirección o salió impulsado**.

En un simple cruce las manchas también se funden, pero la mancha conjunta sigue cambiando de tamaño mientras uno pasa junto al otro, y al separarse cada uno conserva su velocidad: no hay alerta. Tampoco la hay cuando una moto alcanza a un carro lento y se funde con él: la mancha conjunta va más despacio que la moto, pero sigue avanzando.

### Riesgo cumplido (`_revisar_riesgo_cumplido`)

En los videos reales el seguimiento suele perder al vehículo que viene rápido justo antes del golpe: su mancha se funde con la del otro sin que se registre el contacto entre dos objetos con historial. Por eso, cuando el sistema ya dio una alerta de riesgo, queda pendiente de sus dos participantes. Hay colisión si la mancha de uno de ellos crece de golpe (algo se le fundió), sigue crecida al menos 1,2 segundos, el otro no está a la vista entero y separado, y esa mancha ya no se mueve como venía cuando se dio el aviso. Si el otro pasa de largo, las manchas se separan antes y no hay alerta.

La misma idea se aplica cuando los dos sí se ven fundidos: con una alerta de riesgo previa basta que sigan juntos 1,2 segundos y que la mancha haya cambiado su movimiento. Una moto que alcanza a un carro y sigue a su lado no cambia el movimiento del carro, y no cuenta.

### Frenada imposible (`_revisar_impacto`)

Un vehículo que venía de forma sostenida a más de 1,5 tamaños por segundo y 0,4 segundos después está casi quieto. Ningún freno detiene un carro así; solo un golpe. Sirve para los choques contra un poste o un muro, y contra un vehículo que el sistema no venía siguiendo. Se exige que la velocidad sea sostenida (tres mediciones seguidas) porque un salto aislado de la caja también parece velocidad.

### Riesgo (`_revisar_riesgo`)

Para dos objetos que no se tocan, con al menos un vehículo de motor. Con la posición relativa $p$ y la velocidad relativa $v$, el instante de máximo acercamiento es $t^* = -\dfrac{p \cdot v}{\lVert v \rVert^2}$ y la distancia en ese instante es $\lVert p + v\,t^* \rVert$. Hay alerta si $t^*$ es menor que un segundo, si pasarían muy cerca y si estarían a la misma profundidad.

**Perspectiva (`_cierre_en_tamanos`).** Dos carros que se alejan de la cámara por la misma vía se juntan en la imagen aunque mantengan su distancia, porque todo se encoge hacia el horizonte. El tamaño de una caja es inversamente proporcional a la distancia a la cámara, así que la separación en la imagen dividida entre el producto de los dos tamaños no cambia mientras la distancia real no cambie. Solo hay riesgo, o contacto que pueda ser choque, si esa medida también se reduce.

### Caída (`_revisar_caida`)

La caja de una persona de pie es más alta que ancha; en el suelo, más ancha que alta. Se reporta una caída si la relación ancho/alto pasa de menos de 0,75 a más de 1,3 dentro de un segundo y, además, la caja se acortó al menos un 30 %, se ensanchó al menos un 20 % y su área no cambió en más del doble. Sin esas condiciones también contaría como caída una mancha que se le pega a la persona (se ensancha sin acortarse) o algo que la tapa por debajo (se acorta sin ensancharse). Se usa la mediana de varios fotogramas, y no se evalúan las cajas cortadas por el borde ni las fundidas con otra.

## 7. El asistente: un modelo de lenguaje y un clasificador, los dos propios

El asistente no usa ChatGPT, Gemini, Claude ni ningún modelo descargado. Tiene dos redes Transformer entrenadas desde cero. Las dos comparten las mismas capas (`src/transformer.py`).

### 7.1. Las capas comunes

1. **Embedding con posición.** Cada ficha (palabra) se convierte en un vector que la red aprende, y se le suma otro vector según la posición que ocupa. Sin esa suma, la atención no sabría en qué orden vienen las palabras.
2. **Bloque Transformer.** Autoatención multi-cabeza, conexión residual y normalización; después una red densa con ReLU y otra vez residual y normalización. La atención es $\text{Attention}(Q,K,V) = \text{softmax}\left(\dfrac{QK^T}{\sqrt{d_k}}\right)V$ con $Q = K = V$ = los vectores de la secuencia. Una máscara impide atender a las posiciones de relleno.

El bloque tiene un interruptor, `causal`:

- `causal=False`: bloque **codificador**. Cada palabra mira a todas las demás. Sirve para entender una frase completa.
- `causal=True`: bloque **decodificador**. Cada palabra solo mira a las anteriores. Sirve para generar texto, porque al predecir la palabra siguiente no se puede haber visto.

### 7.2. El modelo de lenguaje (`src/modelo_lenguaje.py`)

Es un Transformer de solo decodificador, la arquitectura de GPT, en tamaño pequeño: 2 bloques, 4 cabezas de atención, vectores de 64, vocabulario de 278 fichas, 106 390 parámetros.

**Qué aprende.** A predecir la ficha siguiente. La salida es una capa densa con softmax sobre todo el vocabulario: para cada posición, la probabilidad de cada ficha como continuación.

**Con qué se entrena.** `entrenamiento/generar_corpus.py` crea `datos/corpus_asistente.csv` con 24 000 ejemplos de tres columnas: datos, pregunta, respuesta. Cada ejemplo se vuelve una secuencia:

```
<datos> colisiones_uno riesgos_cero caidas_varios ... <pregunta> cuando fue el choque <respuesta> la colisión ocurrió en el <t_colisiones> <fin>
```

La entrada es la secuencia y la salida esperada es la misma secuencia corrida una posición. Función de costo: entropía cruzada; optimizador: Adam.

**Cómo escribe (generación autorregresiva).** Se le da `<datos> ... <pregunta> ... <respuesta>`. El modelo predice la ficha siguiente, esa ficha se agrega al final, y se le vuelve a preguntar. Así, ficha por ficha, hasta que escribe `<fin>`. Es la misma forma en que escribe cualquier LLM. Por defecto se toma siempre la ficha más probable; con temperatura mayor que cero se sortea entre las probables.

**Dos decisiones de diseño** (`src/lenguaje.py`):

- **Datos resumidos.** Al modelo no se le dan las cifras, solo si cada cantidad es cero, uno o varios (`colisiones_uno`). Con eso elige la redacción: "no hubo", "se detectó una", "se detectaron varias".
- **Ranuras.** El modelo no escribe números: escribe ranuras como `<n_colisiones>` o `<lista_colisiones>`, y el programa las llena con el registro. Un modelo pequeño puede equivocarse al copiar una cifra; así las cifras siempre son las reales.

### 7.3. El clasificador de intenciones (`src/transformer.py`)

Un Transformer codificador de un bloque, 2 cabezas y vectores de 32 (15 921 parámetros). Lee la pregunta completa, promedia los vectores de sus palabras y, con una capa softmax, dice cuál de 17 intenciones es. Se entrena con `datos/preguntas_asistente.csv` (unas 6 500 preguntas). Una de las intenciones, `tipo_vehiculo`, reúne las preguntas por buses o camiones: la respuesta dice que el sistema no distingue el tipo de vehículo y da el total de vehículos.

### 7.4. Cómo trabajan juntos (`src/asistente.py`)

1. El clasificador dice la intención de la pregunta. Si no está seguro, se deduce por palabras clave.
2. El modelo de lenguaje escribe la respuesta.
3. **Control:** lo escrito se acepta solo si es una respuesta correcta para esa intención y para los datos del video. No puede hablar de la duración si se preguntó por las colisiones, ni decir "no hubo" si hubo dos.
4. Si pasa el control, se llenan las ranuras y esa es la respuesta. Si no, se arma una respuesta de respaldo a partir de la intención.

La interfaz indica siempre el origen: escrita por el modelo de lenguaje, armada con la intención del clasificador, o armada por palabras clave.

## 8. Relación con los temas del curso

| Tema del curso | Dónde aparece |
|---|---|
| Procesamiento de imágenes con OpenCV | Sustracción de fondo, umbral, erosión y dilatación, contornos |
| Redes convolucionales | La CNN: Conv2D, MaxPooling, Flatten, Dense, softmax |
| Funciones de activación y de costo | ReLU, softmax, entropía cruzada |
| Backpropagation y descenso de gradiente | Así se entrenan la CNN y el Transformer (optimizador Adam) |
| Sobreajuste, Dropout y Early Stopping | Entrenamiento de la CNN |
| Matriz de confusión, precisión y recall | Evaluación de la CNN en el conjunto de prueba |
| Transformers y atención | El clasificador (codificador) y el modelo de lenguaje (decodificador) |
| Modelos de lenguaje (LLM) | El modelo de lenguaje propio: predicción de la siguiente ficha y generación autorregresiva |
| Agentes | El sistema percibe (cámara), mantiene estado (trayectorias) y actúa (alertas) |

## 9. Preguntas frecuentes

**¿Usaron algún modelo preentrenado?**
No. La CNN, el modelo de lenguaje y el clasificador se definen capa por capa en el repositorio y sus pesos empiezan al azar. Tampoco se llama a ningún servicio como ChatGPT, Gemini o Claude. OpenCV se usa para operaciones clásicas (leer video, sustracción de fondo, morfología), que no son modelos entrenados.

**¿Por qué no entrenar una red que detecte y localice directamente?**
Un detector de objetos necesita muchas más imágenes anotadas con cajas y mucho más tiempo de entrenamiento. Se separó el problema: el movimiento dice dónde está el objeto y la CNN dice qué es, que es una tarea de clasificación mucho más fácil de aprender desde cero.

**¿Qué dataset usaron y cómo lo dividieron?**
MIO-TCD Classification (Luo et al., 2018), recortes de objetos tomados por cámaras de tránsito. Sus categorías se agruparon en las siete clases con las que se entrena la red (carro = car + pickup truck + work van; camión = articulated truck + single unit truck) y se tomaron hasta 4.000 imágenes por clase: 24.266 en total (bicicleta 2.284 y motocicleta 1.982, porque el dataset no tiene más). División por clase: 16.985 para entrenamiento (70 %), 3.640 para validación (15 %) y 3.641 para prueba (15 %).

**¿Por qué imágenes de 64 × 64?**
Los objetos de tránsito vistos por una cámara suelen ser pequeños; 64 × 64 conserva su forma y mantiene la red liviana para correr en CPU en tiempo real. La imagen no se estira para llenar el cuadro: se escala y se rellena con gris, para no perder la proporción del objeto.

**¿Qué exactitud obtuvo la CNN y dónde falla?**
0,901 de exactitud con las siete clases sobre las 3.641 imágenes de prueba, y 0,938 por grupos (carro, bus y camión contados como vehículo), que es como el sistema usa la red. Lo que mejor reconoce es fondo (precisión 0,97, recall 0,95) y bus (0,94 en ambas). Lo que más le cuesta es bicicleta (precisión 0,78) y persona (recall 0,85). Se entrenó las 40 épocas completas; la pérdida de validación seguía bajando (mínimo 0,303 en la época 39), así que con más épocas podría mejorar un poco. La exactitud final fue 0,901 tanto en entrenamiento como en validación: no hay sobreajuste.

**¿Por qué el sistema dice "vehículo" y no carro, bus o camión?**
En el primer entrenamiento la red obtuvo 0,898 de exactitud en las imágenes de prueba, pero al usarla con un video propio llamó "bus" o "camión" a muchos carros. La exactitud de prueba se mide con imágenes de las mismas cámaras del dataset; con otra cámara, otro ángulo y otra resolución la red generaliza peor. Se hicieron tres cambios: las imágenes ya no se deforman, al entrenar se varía su nitidez, y las probabilidades de carro, bus y camión se suman en un solo grupo. Distinguir un vehículo de una persona o de una moto sí resulta confiable; distinguir el tipo de vehículo, no, con una red de este tamaño y esta cantidad de imágenes. El enunciado pide detectar vehículos, motocicletas y peatones, y eso es lo que el sistema reporta.

**¿Por qué una exactitud alta en prueba no garantiza que funcione en un video real?**
Porque el conjunto de prueba sale del mismo dataset que el de entrenamiento: mismas cámaras, mismos ángulos, misma calidad de imagen. La red puede apoyarse en pistas que solo valen para ese dataset (por ejemplo, que los buses aparecen nítidos y los peatones borrosos). Por eso, además de medir en prueba, hay que mirar los resultados sobre videos reales.

**¿Cómo midieron el sistema con accidentes reales?**
Con `evaluar.py`: se anotó a mano el segundo del choque en 17 grabaciones de cámaras fijas y se contó en cuántas el sistema dio una alerta de colisión cerca de ese momento. [Completar con la salida de `evaluar.py`: detectados de 17, falsas alarmas, precisión y recall.] Los que no detecta tienen casi siempre la misma causa: uno de los vehículos entra a la imagen y choca en menos de un segundo, antes de que el sistema alcance a confirmarlo, o su mancha se pierde entre el ruido del fondo. Cuando sí venía siguiendo a los dos, avisa del riesgo y luego reporta la colisión.

**¿Cómo saben si hay sobreajuste?**
Por las curvas: si la pérdida de entrenamiento sigue bajando mientras la de validación sube. Para evitarlo se usan aumento de datos, Dropout y parada temprana.

**¿Qué es el IoU y para qué lo usan?**
Intersección sobre unión de dos cajas. El seguidor lo usa para decidir qué caja nueva corresponde a cada objeto.

**¿Cómo distingue un choque de un cruce?**
Por la profundidad antes del contacto y por el desenlace: en un choque quedan detenidos juntos o alguno cambia bruscamente de movimiento; en un cruce cada uno sigue como venía.

**¿La red neuronal detecta el accidente?**
No directamente. La red dice qué es cada objeto; el accidente se deduce con reglas sobre las trayectorias. Es un diseño híbrido que permite explicar cada alerta.

**¿Su modelo de lenguaje es un LLM?**
Es un modelo de lenguaje con la misma arquitectura y el mismo modo de entrenamiento y de generación que un LLM (Transformer decodificador, predicción de la siguiente ficha, generación autorregresiva), pero no es "grande": tiene 107 mil parámetros y un vocabulario de 281 fichas, frente a miles de millones de parámetros de un LLM comercial. Por eso solo sabe hablar de los eventos de un video de tránsito.

**¿Qué diferencia hay entre el codificador y el decodificador?**
La máscara de atención. En el codificador cada palabra ve toda la frase; en el decodificador solo ve las anteriores (máscara causal). El primero sirve para entender, el segundo para generar.

**¿Cómo genera texto el modelo?**
Predice la ficha más probable como continuación, la agrega a la secuencia y repite, hasta escribir la ficha de fin.

**¿Por qué el modelo escribe ranuras y no los números?**
Porque un modelo de lenguaje pequeño puede equivocarse al copiar una cifra. El modelo decide la redacción y el programa pone los datos del registro, de modo que las cifras siempre son correctas. Los asistentes reales hacen algo parecido cuando consultan una herramienta para obtener un dato.

**¿Para qué sirve entonces el clasificador?**
Para controlar al modelo de lenguaje: lo que este escribe solo se acepta si corresponde a la intención de la pregunta y a los datos. Si no, hay una respuesta de respaldo.

**¿Con qué datos entrenaron el modelo de lenguaje?**
Con un corpus propio de 24 000 ejemplos generado por `generar_corpus.py`: para cada uno se inventa un video, se elige una pregunta y una de las formas correctas de responderla.

**¿Qué hace la máscara de atención?**
Las secuencias se rellenan con ceros hasta un largo fijo. La máscara de relleno evita que esas posiciones vacías influyan; la máscara causal, en el modelo de lenguaje, evita que una posición vea las siguientes.

**¿Qué pasa con una pregunta que el asistente no entiende?**
Si el clasificador no está seguro, la intención se deduce por palabras clave; si no coincide ninguna, se entrega un resumen del video. En todos los casos lo que escribe el modelo de lenguaje pasa por el control.

**¿Cuáles son las principales limitaciones?**
Necesita cámara fija; solo ve lo que se mueve; los objetos que se tocan se ven como uno; las sombras generan manchas falsas; y la CNN es pequeña.

**¿Cómo se podría mejorar?**
Entrenar con más imágenes; entrenar una red propia que también localice los objetos; calibrar la cámara para medir velocidades reales; agregar una LSTM que analice la secuencia de posiciones para confirmar las alertas.
