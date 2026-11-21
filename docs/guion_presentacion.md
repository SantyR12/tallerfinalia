# Guion de la presentación

Diez minutos, once diapositivas. Las diapositivas están en `docs/Presentacion_Proyecto_2.pdf`.

## 1. Detección de accidentes de tránsito y comportamientos peligrosos

30 segundos. Presentar al equipo y decir en una frase qué hace el sistema: mira un video de tránsito, sigue cada vehículo y persona, avisa cuando ve un posible choque o una caída, y responde preguntas sobre lo que pasó.

## 2. El reto: todo hecho desde cero

45 segundos. La condición cambia todo el diseño: un detector moderno se entrena con millones de imágenes, y nosotros no podíamos usar ninguno. Hubo que dividir el problema en partes que sí se pueden resolver con modelos pequeños.

## 3. Cinco pasos, del píxel a la respuesta

1 minuto. La idea central: el movimiento resuelve dónde está el objeto, la CNN resuelve qué es, y las reglas deciden si hay un evento. Así ninguna parte necesita un modelo grande. Aclarar que OpenCV solo aporta operaciones clásicas de imagen.

## 4. Detectar y seguir sin una red

45 segundos. El seguidor es código nuestro, no una librería. La votación evita que un cuadro mal clasificado cambie la clase del objeto. Límite honesto: solo ve lo que se mueve; un carro que se detiene pasa a ser fondo a los 5 segundos.

## 5. Una CNN propia, entrenada desde cero

1 minuto. Sin pesos preentrenados: la red arranca con pesos al azar. La exactitud en el dataset era alta, pero en nuestro propio video marcaba buses donde no había. La causa: la red se apoyaba en pistas del dataset, como la nitidez de los recortes. Lo corregimos y agrupamos las clases que no distingue bien.

## 6. Tres reglas deciden las alertas

1 minuto. Las reglas son explícitas: se puede explicar por qué saltó cada alerta. Lo que distingue un choque de un cruce no es el contacto sino lo que pasa después. La última frase es la mejora que hizo que los choques reales se detectaran también en otro computador.

## 7. Asistente: dos Transformers propios

1 minuto. No es ChatGPT ni ningún modelo comercial: los dos se entrenan desde cero en pocos minutos. El mismo bloque de atención sirve para entender o para generar, según lleve o no una máscara causal. Como las cifras vienen del registro, el modelo no puede inventar datos. Si lo que redacta no es coherente con la pregunta, se usa una respuesta de respaldo.

## 8. Tráfico normal, video propio

1 minuto y medio. Comando:

```bash
python main.py --video data/tu_video.mp4
```

El video dura un minuto; se pueden mostrar unos 30 segundos y cerrar la ventana. Si aparece alguna alerta falsa, decirlo sin rodeos: el sistema marca eventos posibles, que una persona debe confirmar.

## 9. Un choque real

1 minuto y medio. Comando:

```bash
python main.py --video "../videos_choques/test_videos/test (12).mp4" --asistente
```

El video dura 12 segundos. Al terminar se abre el asistente: preguntar «quién chocó con quién» y «cuántos vehículos hubo». Para repetir preguntas sin volver a correr el video:

```bash
python main.py --consultar salidas
```

## 10. Resultados con 17 choques reales

45 segundos. Decir las cifras tal como son. Los 14 choques no detectados fallan casi siempre por lo mismo: el sistema no alcanza a seguir a los dos vehículos antes del impacto, porque entran a la imagen y chocan en menos de un segundo.

## 11. Lo que aprendimos

45 segundos. Cerrar con la idea de que las alertas son confiables pero la ausencia de alerta no descarta un accidente. Abrir a preguntas.
