# Subir el proyecto a GitHub por etapas

La idea es subir el proyecto en varios commits, uno por cada parte, para que el historial muestre cómo se construyó. Todos los comandos se escriben en la Terminal, dentro de la carpeta del proyecto.

## 0. Una sola vez

1. En [github.com](https://github.com) pulsa **New repository**. Nombre sugerido: `deteccion-accidentes-transito`. Déjalo **sin** README, sin .gitignore y sin licencia (el proyecto ya los trae).
2. En la Terminal:

```bash
cd ~/Downloads/examIA/proyecto-final-accidentes-transito
git init -b main
git remote add origin https://github.com/SantyR12/tallerfinalia.git
```

Si es la primera vez que usas git en este computador:

```bash
git config --global user.name "Tu Nombre"
git config --global user.email "tu_correo_de_github"
```

## 1. Antes de cada etapa

Prueba que lo que vas a subir funciona:

```bash
python -m pytest tests -q
```

## 2. Las etapas

Cada bloque es un commit. Súbelos en este orden; puedes hacer uno por día o varios seguidos.

**Etapa 1 — Estructura y configuración**

```bash
git add README.md requirements.txt .gitignore config.py data/.gitkeep salidas/.gitkeep modelos/.gitkeep src/__init__.py
git commit -m "Estructura inicial del proyecto y parámetros"
git push -u origin main
```

**Etapa 2 — Detección de movimiento y seguimiento propio**

```bash
git add src/movimiento.py src/seguimiento.py
git commit -m "Sustracción de fondo y seguidor de objetos propio"
git push
```

**Etapa 3 — Dataset y CNN entrenada desde cero**

Entrena primero la red (ver README) y revisa `modelos/entrenamiento_cnn/reporte.txt`.

```bash
git add entrenamiento/__init__.py entrenamiento/preparar_dataset.py entrenamiento/entrenar_cnn.py
git add src/clasificador.py src/detector.py tests/__init__.py tests/test_cnn.py
git add modelos/entrenamiento_cnn/
git commit -m "CNN propia para clasificar objetos: scripts de entrenamiento y resultados"
git push
```

**Etapa 4 — Reglas de eventos y sus pruebas**

```bash
git add src/eventos.py tests/test_eventos.py tests/test_deteccion.py tests/escenas.py
git commit -m "Reglas de colisión, riesgo y caída con pruebas"
git push
```

**Etapa 5 — Alertas, recorrido del video y programa de consola**

```bash
git add src/alertas.py src/video.py main.py tests/test_video.py
git commit -m "Alertas visuales, guardado de resultados y programa de consola"
git push
```

**Etapa 6 — Clasificador de intenciones (Transformer codificador)**

Genera el dataset de preguntas y entrena el clasificador (ver README).

```bash
git add entrenamiento/generar_preguntas.py entrenamiento/entrenar_asistente.py datos/preguntas_asistente.csv
git add src/texto.py src/transformer.py tests/test_transformer.py
git commit -m "Transformer codificador propio para clasificar la intención de las preguntas"
git push
```

**Etapa 7 — Modelo de lenguaje propio y asistente**

Genera el corpus y entrena el modelo de lenguaje (ver README).

```bash
git add entrenamiento/generar_corpus.py entrenamiento/entrenar_llm.py datos/corpus_asistente.csv
git add src/lenguaje.py src/modelo_lenguaje.py src/asistente.py tests/test_llm.py tests/test_asistente.py
git commit -m "Modelo de lenguaje propio (Transformer decodificador) y asistente"
git push
```

**Etapa 8 — Aplicación web**

```bash
git add app.py
git commit -m "Aplicación web con Streamlit"
git push
```

**Etapa 9 — Documentación e informe**

```bash
git add docs/
git commit -m "Documentación e informe del proyecto"
git push
```

## 3. Si GitHub pide contraseña

GitHub no acepta la contraseña de la cuenta en la Terminal. Hay dos caminos:

- **GitHub Desktop** (el más sencillo): instálalo, inicia sesión, usa *Add existing repository* con la carpeta del proyecto y publica desde ahí.
- **Token personal:** en GitHub ve a *Settings → Developer settings → Personal access tokens*, crea uno con permiso `repo` y pégalo cuando la Terminal pida la contraseña.

## 4. Qué no se sube

El archivo `.gitignore` se encarga de que no se suba lo que no hace falta. Aunque escribas `git add .`, git lo deja por fuera:

- **Los modelos entrenados** (`modelos/*.keras` y sus archivos `.json`). Quien descargue el proyecto los genera con los scripts de `entrenamiento/`.
- **Los videos** de `data/` y **los resultados** de `salidas/`.
- **El dataset de imágenes** y su CSV, que tiene rutas de tu computador. El README indica dónde descargarlo.
- **Lo que crea Python o el sistema**: `__pycache__/`, `.pytest_cache/`, el entorno `.venv/`, `.DS_Store`, configuraciones del editor.

Sí se suben las gráficas y el reporte del entrenamiento de la CNN (`modelos/entrenamiento_cnn/`), porque son la evidencia de los resultados, y los dos CSV propios de `datos/` (preguntas y corpus), que son los datasets del asistente.

Para comprobar qué va a subir git antes de cada commit:

```bash
git status
```

## 5. Cambios posteriores

Cada vez que ajustes algo (por ejemplo un umbral en `config.py` después de probar con tus videos):

```bash
git add config.py
git commit -m "Ajuste del umbral de velocidad de acercamiento tras pruebas con videos reales"
git push
```
