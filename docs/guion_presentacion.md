# Guion de la sustentación

Proyecto final 2 de Inteligencia Artificial: detección de accidentes de tránsito y comportamientos peligrosos.

Integrantes: Santiago Ramos, Paulo Tibanta, Nelson Barrera y Julian Dorado.

Duración total: unos 13 minutos (9 de exposición y 4 de demostración). Las diapositivas están en `docs/Presentacion_Proyecto_2.pdf`.

El texto de cada diapositiva es una guía de lo que se dice, con palabras sencillas. No hay que aprenderlo de memoria: cada uno lo cuenta a su manera.

## Reparto

| Quién | Parte | Diapositivas | Tiempo |
|---|---|---|---|
| Santiago Ramos | El problema y cómo se detecta y se sigue cada objeto | 1 a 4 | 2 minutos |
| Paulo Tibanta | La red convolucional: arquitectura, entrenamiento y métricas | 5 a 9 | 4 minutos |
| Nelson Barrera | Las reglas de eventos y el asistente | 10 a 11 | 2 minutos |
| Julian Dorado | Demostración y cierre | 12 a 15 | 5 minutos |

Cada uno responde las preguntas de su parte. Si una pregunta es de otra parte, se le pasa la palabra a quien la explicó.

## Santiago Ramos: El problema y cómo se detecta y se sigue cada objeto

Abre la presentación, explica qué pide el proyecto y la condición de no usar nada preentrenado, y recorre los cinco pasos del sistema. Después explica los dos primeros: cómo se encuentra lo que se mueve y cómo se le conserva el número a cada objeto. Durante la demostración maneja el computador.

Archivos que debe conocer: `src/movimiento.py`, `src/seguimiento.py`, `src/detector.py` y `config.py`.

### Diapositiva 1. Detección de accidentes de tránsito y comportamientos peligrosos

> Buenos días, somos Santiago, Paulo, Nelson y Julian. Nuestro proyecto detecta accidentes de tránsito en un video: sigue a los vehículos y a las personas, avisa cuando ve un posible choque o una caída, y tiene un asistente al que se le puede preguntar qué pasó.

### Diapositiva 2. Lo que nos pidieron

> Esto era lo que pedía el proyecto. Lo difícil fue la condición: no podíamos usar ningún modelo ya entrenado, ni para detectar ni para el asistente. Entonces partimos el problema en partes pequeñas que sí podíamos resolver con modelos nuestros.

### Diapositiva 3. Cómo funciona

> El sistema tiene cinco pasos. Primero busca lo que se mueve. Después le pone un número a cada objeto para seguirlo. Nuestra red dice si es un vehículo, una moto, una bicicleta o una persona. Unas reglas revisan si hay riesgo, choque o caída. Y al final el asistente responde preguntas sobre lo que pasó.

### Diapositiva 4. Detectar y seguir

> Este es un video nuestro. El sistema aprende cómo se ve la vía vacía y lo que cambia es un objeto. A cada uno le pone una caja y un número, y ese número se mantiene mientras avanza; eso lo hace un seguidor que programamos nosotros. Sirve para cámaras fijas, como las de tránsito. Ahora Paulo explica la red.

### Preguntas que le pueden hacer a Santiago

**¿Qué es la sustracción de fondo?** Un algoritmo clásico de imagen (MOG2, de OpenCV). Aprende píxel por píxel cómo se ve la escena vacía; lo que se aparta de ese fondo es un objeto en movimiento. No es una red neuronal ni un modelo preentrenado.

**¿Qué es IoU?** Intersección sobre unión: el área que comparten dos cajas dividida entre el área que ocupan entre las dos. Vale 0 si no se tocan y 1 si son iguales. El seguidor empareja cada objeto con la caja nueva que más se le solapa.

**¿Por qué no usaron un detector como YOLO?** Porque la condición del proyecto era no usar modelos preentrenados. Por eso se separó el problema: el movimiento dice dónde está el objeto y la CNN propia dice qué es.

**¿Qué pasa con un carro que se detiene?** Sigue visible unos cinco segundos y después pasa a ser parte del fondo. El sistema está pensado para vías con tráfico en movimiento; ver objetos quietos es el primer punto de los próximos pasos.

## Paulo Tibanta: La red convolucional: arquitectura, entrenamiento y métricas

Explica la CNN con las capturas del código real, cómo se entrena y qué resultados dio: las curvas, el reporte con precisión, recall y F1, y la matriz de confusión.

Archivos que debe conocer: `entrenamiento/entrenar_cnn.py`, `src/clasificador.py` y la carpeta `modelos/entrenamiento_cnn/` (reporte.txt, curvas.png, matriz_confusion.png).

### Diapositiva 5. Nuestra red (CNN)

> Este es el código de nuestra red. Recibe un recorte de 64 por 64 píxeles y tiene tres capas convolucionales con ReLU, cada una con su MaxPooling, que reduce la imagen a la mitad. Al final hay dos capas densas y la salida con softmax, que da la probabilidad de cada una de las siete clases: bicicleta, bus, camión, carro, fondo, moto y persona. Son un poco más de un millón de parámetros y todos empiezan al azar.

### Diapositiva 6. Así la entrenamos

> Aquí se entrena. Usamos el optimizador Adam, la entropía cruzada como función de pérdida, que es la que se usa para clasificar en varias clases, y medimos el accuracy. Entrenamos 40 épocas con unas 17 mil imágenes del dataset MIO-TCD, que son recortes de cámaras de tránsito. Si la validación deja de mejorar, el entrenamiento se detiene solo.

### Diapositiva 7. Cómo aprendió

> Estas son las curvas. A la izquierda la pérdida, que baja, y a la derecha el accuracy, que sube hasta 0,90. La línea azul es el entrenamiento y la naranja la validación. Como van juntas, sabemos que la red no está memorizando, o sea que no hay sobreajuste.

### Diapositiva 8. Resultados de la red

> Este es el reporte con las 3.641 imágenes de prueba, que la red nunca vio. El accuracy es 0,90: acierta 90 de cada 100. La precisión dice, de lo que la red marcó como bus, cuánto era bus de verdad. El recall dice, de los buses que había, cuántos encontró. El F1 junta las dos. La clase más difícil es la bicicleta. Y si contamos carro, bus y camión como un solo grupo, que es como lo usa el sistema, sube a 0,94.

### Diapositiva 9. Matriz de confusión

> En la matriz las filas son la clase real y las columnas lo que dijo la red. La diagonal son los aciertos. Por ejemplo, 52 camiones los tomó por carros. Como lo que más confunde es carro, bus y camión, y para detectar un choque da igual cuál sea, el sistema los junta como «vehículo». Sigue Nelson con las reglas.

### Preguntas que le pueden hacer a Paulo

**¿Por qué ReLU y por qué softmax?** ReLU es la activación de las capas internas: deja pasar los valores positivos y anula los negativos; es simple y evita que el gradiente se desvanezca. Softmax va en la salida: convierte los siete valores en probabilidades que suman 1.

**¿Qué es la entropía cruzada?** La función de pérdida para clasificar en varias clases. Mira la probabilidad que la red le dio a la clase correcta: si fue alta, la pérdida es pequeña; si fue baja, es grande. El entrenamiento ajusta los pesos para reducirla.

**¿Qué diferencia hay entre precisión y recall?** Precisión: de lo que la red marcó como una clase, cuánto era de esa clase. Recall: de todo lo que realmente era de esa clase, cuánto encontró. F1 combina las dos.

**¿Cómo saben que no hay sobreajuste?** Por las curvas: la pérdida de validación baja junto con la de entrenamiento y las dos exactitudes terminan en 0,90. Además hay Dropout de 0,4, aumento de datos y parada temprana.

**¿Por qué juntan carro, bus y camión?** Porque son las clases que más se confunden entre sí (135 de los 359 errores) y para detectar un choque basta con saber que es un vehículo. El sistema suma las tres probabilidades.

**¿De dónde es el dataset?** MIO-TCD, un dataset público de recortes tomados por cámaras de tránsito. Se usaron 24.266 imágenes: 70 % para entrenar, 15 % para validar y 15 % para probar.

## Nelson Barrera: Las reglas de eventos y el asistente

Explica cuándo avisa el sistema (riesgo, colisión y caída) y cómo funciona el asistente.

Archivos que debe conocer: `src/eventos.py`, `src/transformer.py`, `src/modelo_lenguaje.py`, `src/lenguaje.py` y `src/asistente.py`.

### Diapositiva 10. Cuándo avisa

> Las alertas salen de reglas sobre el movimiento, no de otra red. Riesgo: con la posición y la velocidad de dos objetos calculamos en cuánto tiempo se tocarían; si es menos de un segundo, avisa. Colisión: no basta con que se toquen, porque dos carros que se cruzan también se tocan en la imagen; tiene que haber un cambio brusco después: frenan de golpe o uno sale empujado. Caída: la caja de la persona pasa de vertical a horizontal.

### Diapositiva 11. El asistente

> El asistente no es ChatGPT: son dos redes pequeñas que entrenamos nosotros. La primera es un Transformer que entiende qué se está preguntando, por ejemplo cuántas colisiones hubo. La segunda es un modelo de lenguaje tipo GPT, pero pequeño, que escribe la respuesta palabra por palabra. Los números no los inventa: el programa los pone desde el registro de eventos. Esta es una captura de la aplicación respondiendo. Ahora Julian muestra la demostración.

### Preguntas que le pueden hacer a Nelson

**¿Cómo distinguen un choque de dos carros que solo se cruzan?** Por lo que pasa después del contacto. En un cruce cada uno sigue su camino; en un choque frenan de golpe, quedan detenidos juntos o uno sale impulsado. Además deben estar a la misma profundidad en la imagen.

**¿Qué es el TTC?** Tiempo hasta el contacto. Con la posición y la velocidad relativas de dos objetos se calcula cuándo estarían más cerca si siguen como van. Si es menos de un segundo, se avisa del riesgo.

**¿El asistente es un LLM de verdad?** Es un modelo de lenguaje con la misma arquitectura de GPT (Transformer decodificador que predice la palabra siguiente), pero en pequeño: 106.390 parámetros y un vocabulario de 278 fichas. Solo sabe hablar de los eventos de un video.

**¿Qué es la atención?** El mecanismo con que cada palabra mira a las demás para entender la frase. En el decodificador lleva una máscara causal: cada palabra solo puede mirar a las anteriores, y por eso el modelo puede escribir de izquierda a derecha.

**¿Cómo evitan que invente datos?** De dos formas: no escribe cifras sino ranuras que el programa llena con el registro, y su respuesta solo se acepta si coincide con una de las respuestas válidas para la intención detectada. Si no, se usa una respuesta de respaldo.

## Julian Dorado: Demostración y cierre

Narra las tres demostraciones mientras Santiago maneja el computador y hace el cierre.

Archivos que debe conocer: `app.py`, `main.py`, `evaluar.py` y el `README.md`.

### Diapositiva 12. Demo 1: choque en una intersección

> Vamos a probarlo. En la aplicación elegimos «Choque en una intersección» y Analizar. Fíjense en las cajas y en el número de cada carro. Primero sale la alerta amarilla de riesgo y un segundo después la roja de colisión. Al terminar le preguntamos al asistente: ¿cuántas colisiones hubo?

### Diapositiva 13. Demo 2: choque en una avenida

> Otro choque, con otra cámara. Elegimos «Choque en una avenida». Otra vez avisa primero del riesgo y después de la colisión. Le pedimos al asistente un resumen y le preguntamos en qué segundo fue el choque.

### Diapositiva 14. Demo 3: tráfico normal

> Y este es un video nuestro sin accidentes. Elegimos «Tráfico normal». El sistema cuenta los vehículos y no da ninguna alerta, que es lo que debe pasar. Si le preguntamos al asistente si hubo algún accidente, responde que no.

### Diapositiva 15. Conclusiones

> Para cerrar: todo lo que aprende en este proyecto lo entrenamos nosotros, sin modelos de otros. Vimos que el sistema sigue a los vehículos, avisa del riesgo y del choque, y que el asistente responde con los datos del video. Funciona mejor con cámara fija y cuando los vehículos se venían moviendo; lo que le falta es ver los que están quietos, y para eso el siguiente paso sería una red nuestra que también los ubique. Es un prototipo: marca eventos posibles para que una persona los revise. Muchas gracias, ¿tienen preguntas?

### Preguntas que le pueden hacer a Julian

**¿Funciona con cualquier video?** Está pensado para cámara fija, como las de tránsito, y rinde mejor cuando los vehículos se venían moviendo antes del contacto. Con cámara en movimiento la sustracción de fondo no sirve.

**¿Cuántos choques detecta?** Se evaluó con 17 videos de choques de cámaras distintas, de un repositorio público: confirmó 2 y dio 1 alerta que no correspondía (informe, sección 7.5). En los demás, uno de los vehículos estaba detenido o muy lejos. Por eso el primer paso siguiente es una red propia que localice los objetos.

**¿Por qué el video de tráfico se analiza solo 8 segundos?** Se eligió ese tramo para la demostración. El video completo dura un minuto; más adelante, con vehículos lejanos que se juntan en la imagen por la perspectiva, el sistema da una alerta que no corresponde. Es justo el punto de calibrar la perspectiva que aparece en los próximos pasos. Los fotogramas de cada video de demostración se fijan en `config.py`.

**¿Se puede usar con la cámara web?** Sí: en la aplicación se elige «Cámara web», o por consola `python main.py --camara 0`.

## Antes de empezar

1. Abrir la terminal en la carpeta del proyecto y activar el entorno: `source .venv/bin/activate`.
2. Comprobar que están los tres videos: `ls data` debe mostrar `demo_choque_interseccion.mp4`, `demo_choque_avenida.mp4` y `demo_trafico_normal.mp4`.
3. Abrir la aplicación con `streamlit run app.py` y dejarla lista en «Video de demostración».
4. Analizar una vez cada video antes de la sustentación, para confirmar que el computador da las mismas alertas que las diapositivas.
5. Tener abierta otra terminal por si hay que usar la consola: `python main.py --video data/demo_choque_avenida.mp4 --asistente`.
6. Conectar el cargador: con batería baja el análisis va más lento.

## Dónde está cada cosa

| Qué | Archivo |
|---|---|
| Arquitectura de la CNN | `entrenamiento/entrenar_cnn.py`, función `construir_cnn` (líneas 33 a 65) |
| Entrenamiento de la CNN (optimizador, pérdida, épocas) | `entrenamiento/entrenar_cnn.py`, función `main` (líneas 157 a 173) |
| Cálculo de accuracy, precisión, recall, F1 y matriz de confusión | `entrenamiento/entrenar_cnn.py` (líneas 175 a 237) |
| Resultados de la CNN | `modelos/entrenamiento_cnn/reporte.txt`, `curvas.png` y `matriz_confusion.png` |
| Entrenamiento del clasificador de intenciones | `entrenamiento/entrenar_asistente.py` |
| Entrenamiento del modelo de lenguaje | `entrenamiento/entrenar_llm.py` |
| Capas del Transformer | `src/transformer.py` |
| Reglas de riesgo, colisión y caída | `src/eventos.py` |
| Umbrales y lista de videos de demostración | `config.py` |
| Evaluación con videos reales | `evaluar.py` y `datos/choques_reales.csv` |
