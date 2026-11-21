"""
Entrena desde cero la red convolucional que clasifica los objetos.

    python entrenamiento/preparar_dataset.py --origen RUTA/AL/DATASET/train
    python entrenamiento/entrenar_cnn.py

Lee datos/dataset_objetos.csv (ruta, clase, conjunto), entrena la CNN y guarda:
    modelos/clasificador_objetos.keras     la red entrenada
    modelos/clases.json                    el orden de las clases
    modelos/entrenamiento_cnn/             curvas, matriz de confusión y reporte

No se usa ningún modelo preentrenado: todos los pesos empiezan al azar.
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import keras  # noqa: E402
from keras import layers  # noqa: E402

import config  # noqa: E402
from src.clasificador import encuadrar  # noqa: E402


def construir_cnn(clases, tam=config.TAM_IMAGEN):
    """
    Tres bloques convolución + agrupamiento, y dos capas densas.
    Entrada: imagen RGB de tam x tam con valores entre 0 y 1.
    """
    return keras.Sequential([
        keras.Input(shape=(tam, tam, 3)),
        # Aumento de datos (solo actúa al entrenar): la red ve cada imagen
        # volteada, desplazada o con otro brillo, y así generaliza mejor.
        # (El aumento de nitidez se hace aparte, en la clase Lotes.)
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomZoom(0.10),
        layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
        # Extracción de características
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),                    # 64 -> 32
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),                    # 32 -> 16
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),                    # 16 -> 8
        # Clasificación
        layers.Flatten(),
        layers.Dropout(0.4),                            # contra el sobreajuste
        layers.Dense(128, activation="relu"),
        layers.Dense(clases, activation="softmax"),     # una probabilidad por clase
    ], name="clasificador_objetos")


def cargar_imagenes(filas, clases, tam=config.TAM_IMAGEN):
    """Lee las imágenes del CSV y las deja como arreglos listos para la red."""
    x, y = [], []
    for ruta, clase in filas:
        imagen = cv2.imread(ruta)
        if imagen is None:
            continue                                    # archivo dañado o ausente
        x.append(encuadrar(imagen, tam))                # sin deformar, igual que al detectar
        y.append(clases.index(clase))
    return np.array(x, dtype=np.uint8), np.array(y, dtype=np.int32)


def bajar_nitidez(imagen, rng, minimo=config.NITIDEZ_MINIMA):
    """
    Reduce la imagen a un tamaño al azar y la devuelve a su tamaño: queda igual
    de grande pero más borrosa. En el dataset los peatones son recortes pequeños
    (borrosos al ampliarlos) y los buses, grandes (nítidos); sin este aumento la
    red aprende el atajo "nítido = bus o camión" y falla con otras cámaras.
    """
    tam = imagen.shape[0]
    lado = int(rng.integers(minimo, tam + 1))
    if lado >= tam:
        return imagen
    pequena = cv2.resize(imagen, (lado, lado), interpolation=cv2.INTER_AREA)
    return cv2.resize(pequena, (tam, tam), interpolation=cv2.INTER_LINEAR)


class Lotes(keras.utils.PyDataset):
    """Entrega los lotes de entrenamiento, con una nitidez distinta en cada época."""

    def __init__(self, x, y, pesos, lote, semilla=7):
        super().__init__()
        self.x, self.y, self.pesos, self.lote = x, y, pesos, lote
        self.rng = np.random.default_rng(semilla)
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.x) / self.lote))

    def on_epoch_end(self):
        self.orden = self.rng.permutation(len(self.x))      # barajar en cada época

    def __getitem__(self, i):
        indices = self.orden[i * self.lote:(i + 1) * self.lote]
        x = np.stack([bajar_nitidez(self.x[j], self.rng) for j in indices])
        # El tercer elemento es el peso de cada imagen en la pérdida
        return x.astype(np.float32) / 255.0, self.y[indices], self.pesos[self.y[indices]]


def main():
    parser = argparse.ArgumentParser(description="Entrena la CNN clasificadora de objetos")
    parser.add_argument("--csv", default="datos/dataset_objetos.csv")
    parser.add_argument("--epocas", type=int, default=40)
    parser.add_argument("--lote", type=int, default=64)
    args = parser.parse_args()
    keras.utils.set_random_seed(7)

    if not os.path.exists(args.csv):
        sys.exit(
            f"No existe '{args.csv}'. Créalo primero con:\n"
            "  python entrenamiento/preparar_dataset.py --origen RUTA/AL/DATASET/train"
        )
    with open(args.csv, encoding="utf-8") as archivo:
        filas = list(csv.DictReader(archivo))
    clases = sorted({f["clase"] for f in filas})
    conjuntos = {}
    for nombre in ("entrenamiento", "validacion", "prueba"):
        seleccion = [(f["ruta"], f["clase"]) for f in filas if f["conjunto"] == nombre]
        conjuntos[nombre] = cargar_imagenes(seleccion, clases)
        print(f"{nombre}: {len(conjuntos[nombre][1])} imágenes")
    (x_ent, y_ent), (x_val, y_val), (x_pru, y_pru) = (conjuntos[n] for n in
                                                      ("entrenamiento", "validacion", "prueba"))
    x_val, x_pru = x_val.astype(np.float32) / 255.0, x_pru.astype(np.float32) / 255.0

    # Si hay clases con menos imágenes, sus errores pesan más
    cuenta = np.bincount(y_ent, minlength=len(clases))
    pesos = (len(y_ent) / (len(clases) * np.maximum(cuenta, 1))).astype(np.float32)

    modelo = construir_cnn(len(clases))
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    modelo.summary()
    historia = modelo.fit(
        Lotes(x_ent, y_ent, pesos, args.lote), validation_data=(x_val, y_val),
        epochs=args.epocas, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=6,
                                                 restore_best_weights=True)],
    )

    # ------------------------------------------------------------ evaluación
    perdida, exactitud = modelo.evaluate(x_pru, y_pru, verbose=0)
    predicho = np.asarray(modelo.predict(x_pru, verbose=0)).argmax(axis=1)
    print(f"\nExactitud en prueba: {exactitud:.3f}")

    # Exactitud como la usa el sistema: carro, bus y camión cuentan como "vehiculo"
    grupo_de = [config.GRUPOS.get(c, c) for c in clases]
    grupos = sorted(set(grupo_de))
    grupo = np.array([grupos.index(g) for g in grupo_de])       # clase -> número de grupo
    exactitud_grupos = float((grupo[predicho] == grupo[y_pru]).mean())
    print(f"Exactitud por grupos (carro, bus y camión como 'vehiculo'): {exactitud_grupos:.3f}")

    carpeta = "modelos/entrenamiento_cnn"
    os.makedirs(carpeta, exist_ok=True)
    modelo.save(config.RUTA_CNN)
    with open(config.RUTA_CLASES, "w", encoding="utf-8") as archivo:
        json.dump(clases, archivo, ensure_ascii=False)

    try:
        from sklearn.metrics import classification_report, confusion_matrix
        reporte = classification_report(y_pru, predicho, labels=list(range(len(clases))),
                                        target_names=clases, zero_division=0)
        matriz = confusion_matrix(y_pru, predicho, labels=list(range(len(clases))))
        print(reporte)
        with open(f"{carpeta}/reporte.txt", "w", encoding="utf-8") as archivo:
            archivo.write(f"Exactitud en prueba: {exactitud:.4f}\n")
            archivo.write(f"Exactitud por grupos (carro, bus y camión como 'vehiculo'): "
                          f"{exactitud_grupos:.4f}\n\n{reporte}\n")
            archivo.write("Matriz de confusión (filas = real, columnas = predicho)\n")
            archivo.write("clases: " + ", ".join(clases) + "\n" + str(matriz) + "\n")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figura, ejes = plt.subplots(1, 2, figsize=(11, 4))
        ejes[0].plot(historia.history["loss"], label="entrenamiento")
        ejes[0].plot(historia.history["val_loss"], label="validación")
        ejes[0].set_title("Pérdida"); ejes[0].set_xlabel("época"); ejes[0].legend()
        ejes[1].plot(historia.history["accuracy"], label="entrenamiento")
        ejes[1].plot(historia.history["val_accuracy"], label="validación")
        ejes[1].set_title("Exactitud"); ejes[1].set_xlabel("época"); ejes[1].legend()
        figura.tight_layout(); figura.savefig(f"{carpeta}/curvas.png", dpi=120); plt.close(figura)

        figura, eje = plt.subplots(figsize=(6.5, 5.5))
        eje.imshow(matriz, cmap="Blues")
        eje.set_xticks(range(len(clases)), clases, rotation=45, ha="right")
        eje.set_yticks(range(len(clases)), clases)
        eje.set_xlabel("Predicción"); eje.set_ylabel("Real"); eje.set_title("Matriz de confusión")
        for i in range(len(clases)):
            for j in range(len(clases)):
                eje.text(j, i, matriz[i, j], ha="center", va="center",
                         color="white" if matriz[i, j] > matriz.max() / 2 else "black")
        figura.tight_layout(); figura.savefig(f"{carpeta}/matriz_confusion.png", dpi=120)
        plt.close(figura)
    except ImportError as error:
        print(f"(Sin reporte detallado ni gráficas: falta {error.name})")

    print(f"Red guardada en {config.RUTA_CNN}; resultados en {carpeta}/")


if __name__ == "__main__":
    main()
