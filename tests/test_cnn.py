"""
Pruebas de la CNN propia: que la arquitectura tiene la forma esperada, que
aprende un problema sencillo desde cero y que el clasificador la usa bien.
Se entrena unos segundos con imágenes sintéticas (rectángulos anchos, altos y
fondo vacío); no mide la calidad con imágenes reales.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

keras = pytest.importorskip("keras")
np = pytest.importorskip("numpy")

import config  # noqa: E402
from entrenamiento.entrenar_cnn import construir_cnn  # noqa: E402
from src.clasificador import ClasificadorCNN, encuadrar, preparar_recorte  # noqa: E402

CLASES = ["carro", "fondo", "persona"]


def imagen(clase, rng, tam=config.TAM_IMAGEN):
    """Imagen sintética de 64 x 64: un rectángulo ancho (carro), uno alto (persona) o solo ruido (fondo)."""
    img = np.clip(rng.normal(95, 8, (tam, tam, 3)), 0, 255).astype(np.uint8)
    color = rng.integers(20, 235, 3)
    d = int(rng.integers(-4, 5))
    if clase == "carro":
        img[24 + d:44 + d, 8 + d:56 + d] = color        # rectángulo ancho
    elif clase == "persona":
        img[6 + d:58 + d, 24 + d:40 + d] = color        # rectángulo alto
    return img


def test_arquitectura():
    """La red tiene 3 capas convolucionales, 7 salidas y 1.142.855 parámetros."""
    modelo = construir_cnn(7)
    assert modelo.output_shape == (None, 7)
    assert sum(1 for capa in modelo.layers if isinstance(capa, keras.layers.Conv2D)) == 3
    assert modelo.count_params() == 1_142_855   # el mismo número que muestra el resumen


def test_preparar_recorte():
    """El recorte queda de 64 x 64, entre 0 y 1, en RGB y sin deformar."""
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[20:60, 50:150] = (255, 0, 0)                  # azul en BGR
    recorte = preparar_recorte(frame, (50, 20, 150, 60))    # caja de 100 x 40: queda de 64 x 26
    assert recorte.shape == (config.TAM_IMAGEN, config.TAM_IMAGEN, 3)
    assert recorte.max() <= 1.0
    centro = recorte[22:42, 4:60]                       # la franja que ocupa la imagen
    assert centro[..., 2].mean() > 0.99 and centro[..., 0].mean() < 0.01       # RGB: azul al final


def test_encuadrar_no_deforma_la_imagen():
    """Una imagen alta y angosta sigue siéndolo; lo que sobra se rellena con gris."""
    alta = np.full((100, 40, 3), 255, dtype=np.uint8)
    cuadro = encuadrar(alta, 64)
    assert cuadro.shape == (64, 64, 3)
    blancas = np.where((cuadro == 255).all(axis=2).any(axis=0))[0]      # columnas con imagen
    assert len(blancas) == 26                           # 40 * 64 / 100, no las 64
    assert (cuadro[:, 0] == 128).all() and (cuadro[:, -1] == 128).all()     # relleno gris
    ancha = encuadrar(np.full((30, 120, 3), 255, dtype=np.uint8), 64)
    assert (ancha == 255).all(axis=2).any(axis=1).sum() == 16           # 30 * 64 / 120 filas


def test_la_cnn_aprende_desde_cero_y_el_clasificador_la_usa(tmp_path):
    """Se entrena la red unos segundos con imágenes sintéticas y se comprueba que el clasificador acierta."""
    keras.utils.set_random_seed(3)
    rng = np.random.default_rng(3)
    etiquetas = np.array([i % 3 for i in range(240)], dtype="int32")
    x = np.stack([imagen(CLASES[e], rng) for e in etiquetas]).astype("float32") / 255.0

    modelo = construir_cnn(len(CLASES))
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    modelo.fit(x[:180], etiquetas[:180], epochs=8, batch_size=32, verbose=0)
    assert modelo.evaluate(x[180:], etiquetas[180:], verbose=0)[1] > 0.9

    ruta_modelo, ruta_clases = str(tmp_path / "cnn.keras"), str(tmp_path / "clases.json")
    modelo.save(ruta_modelo)
    with open(ruta_clases, "w", encoding="utf-8") as archivo:
        json.dump(CLASES, archivo)

    clasificador = ClasificadorCNN(ruta_modelo, ruta_clases)
    # Dos zonas de 128 x 128 dentro de un fotograma, cada una con un objeto
    # dibujado igual que en el entrenamiento (al doble de tamaño).
    frame = np.full((200, 400, 3), 95, dtype=np.uint8)
    frame[36 + 48:36 + 88, 20 + 16:20 + 112] = (40, 40, 200)        # ancho: "carro"
    frame[36 + 12:36 + 116, 220 + 48:220 + 80] = (60, 160, 60)      # alto: "persona"
    resultado = clasificador.clasificar(frame, [(20, 36, 148, 164), (220, 36, 348, 164)])
    # "carro" pertenece al grupo "vehiculo", que es lo que reporta el sistema
    assert [clase for clase, _ in resultado] == ["vehiculo", "persona"]
    assert clasificador.clasificar(frame, []) == []


def test_las_probabilidades_de_carro_bus_y_camion_se_suman(tmp_path):
    """Ninguna de las tres gana sola, pero juntas superan a "persona"."""
    clases = ["bus", "camion", "carro", "persona"]
    with open(tmp_path / "clases.json", "w", encoding="utf-8") as archivo:
        json.dump(clases, archivo)
    modelo = keras.Sequential([keras.Input(shape=(config.TAM_IMAGEN, config.TAM_IMAGEN, 3)),
                               keras.layers.Flatten(), keras.layers.Dense(4, activation="softmax")])
    capa = modelo.layers[-1]
    capa.set_weights([np.zeros_like(capa.get_weights()[0]),
                      np.log(np.array([0.2, 0.2, 0.25, 0.35], dtype="float32"))])
    modelo.save(str(tmp_path / "cnn.keras"))

    clasificador = ClasificadorCNN(str(tmp_path / "cnn.keras"), str(tmp_path / "clases.json"))
    assert clasificador.grupos == ["persona", "vehiculo"]
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    (clase, confianza), = clasificador.clasificar(frame, [(10, 10, 60, 60)])
    assert clase == "vehiculo" and abs(confianza - 0.65) < 1e-4


def test_sin_modelo_entrenado_el_error_dice_que_hacer(tmp_path):
    """Si falta el archivo de la red, el mensaje de error indica el comando para entrenarla."""
    with pytest.raises(FileNotFoundError, match="entrenar_cnn.py"):
        ClasificadorCNN(str(tmp_path / "falta.keras"), str(tmp_path / "falta.json"))
