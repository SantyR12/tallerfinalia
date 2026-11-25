"""
Piezas del Transformer propio del asistente (se entrena desde cero).

El modelo recibe una pregunta convertida en números (un índice por palabra) y
predice su INTENCIÓN: "contar colisiones", "cuándo fue la caída", "resumen"...
Con la intención, el asistente arma la respuesta a partir del registro de eventos.

Arquitectura (un codificador Transformer pequeño, como en "Attention Is All
You Need" pero con un solo bloque):

    índices de palabras
        │
    EmbeddingConPosicion      vector de la palabra + vector de su posición
        │
    BloqueTransformer         autoatención multi-cabeza + red densa,
        │                     con conexiones residuales y normalización
    PromedioSinRelleno        un solo vector para toda la pregunta
        │
    Dense + softmax           probabilidad de cada intención
"""

import keras
from keras import layers, ops


# ----------------------------------------------------------------------------
# Capas
# ----------------------------------------------------------------------------
@keras.saving.register_keras_serializable(package="proyecto")
class EmbeddingConPosicion(layers.Layer):
    """Suma el vector de cada palabra con el vector de la posición que ocupa.

    La autoatención no sabe en qué orden vienen las palabras; el vector de
    posición es lo que le permite distinguir 'carro golpea moto' de 'moto
    golpea carro'.
    """

    def __init__(self, largo, vocabulario, dim, **kwargs):
        # largo: palabras por secuencia; vocabulario: cuántas palabras distintas
        # existen; dim: tamaño del vector con que se representa cada palabra
        super().__init__(**kwargs)
        self.largo, self.vocabulario, self.dim = largo, vocabulario, dim
        # Un Embedding es una tabla que se aprende: una fila (vector) por palabra
        self.palabras = layers.Embedding(vocabulario, dim)
        self.posiciones = layers.Embedding(largo, dim)      # y otra tabla: una fila por posición

    def build(self, forma):
        """Keras llama a build() una vez para crear los pesos de la capa."""
        self.palabras.build(forma)
        self.posiciones.build((self.largo,))
        self.built = True

    def call(self, indices):
        """El cálculo de la capa: vector de la palabra + vector de su posición."""
        posiciones = ops.arange(0, self.largo)              # 0, 1, 2, ..., largo - 1
        return self.palabras(indices) + self.posiciones(posiciones)

    def get_config(self):
        """Datos para volver a crear la capa al cargar el modelo guardado."""
        return {**super().get_config(), "largo": self.largo,
                "vocabulario": self.vocabulario, "dim": self.dim}


@keras.saving.register_keras_serializable(package="proyecto")
class BloqueTransformer(layers.Layer):
    """Un bloque Transformer: autoatención + red densa, con residuales y normalización.

    Con causal=False es un bloque CODIFICADOR: cada palabra mira a todas las
    demás (se usa en el clasificador de intenciones).
    Con causal=True es un bloque DECODIFICADOR como los de GPT: cada palabra
    solo puede mirar a las anteriores, que es lo que permite generar texto
    prediciendo la palabra siguiente (se usa en el modelo de lenguaje).
    """

    def __init__(self, dim, cabezas, dim_densa, abandono=0.1, causal=False, **kwargs):
        # dim: tamaño del vector de cada palabra; cabezas: cuántas "miradas"
        # distintas hace la atención; dim_densa: neuronas de la red densa;
        # abandono: fracción de valores que el Dropout apaga al entrenar
        super().__init__(**kwargs)
        self.dim, self.cabezas, self.dim_densa, self.abandono = dim, cabezas, dim_densa, abandono
        self.causal = causal
        # Autoatención multi-cabeza: la pieza central del Transformer
        self.atencion = layers.MultiHeadAttention(num_heads=cabezas, key_dim=dim // cabezas)
        # Red densa de dos capas que procesa cada palabra después de la atención
        self.densa1 = layers.Dense(dim_densa, activation="relu")
        self.densa2 = layers.Dense(dim)
        # Normalización: mantiene los valores en una escala estable
        self.norma1 = layers.LayerNormalization(epsilon=1e-6)
        self.norma2 = layers.LayerNormalization(epsilon=1e-6)
        # Dropout: apaga valores al azar al entrenar, contra el sobreajuste
        self.abandono1 = layers.Dropout(abandono)
        self.abandono2 = layers.Dropout(abandono)

    def build(self, x_shape, indices_shape=None):
        """Crea los pesos de las capas internas (Keras lo llama una vez)."""
        # Keras exige que estos nombres coincidan con los argumentos de call()
        self.atencion.build(x_shape, x_shape)
        self.densa1.build(x_shape)
        self.densa2.build((*x_shape[:-1], self.dim_densa))
        self.norma1.build(x_shape)
        self.norma2.build(x_shape)
        self.built = True

    def call(self, x, indices, training=None):
        """x: vectores de las palabras; indices: las palabras como números (para saber cuál es relleno)."""
        # Máscara: las posiciones de relleno (índice 0) no reciben atención
        mascara = ops.expand_dims(ops.not_equal(indices, 0), axis=1)
        # Autoatención: cada palabra mira a todas las demás (consulta = clave = valor = x)
        atendido = self.atencion(x, x, attention_mask=mascara, use_causal_mask=self.causal)
        x = self.norma1(x + self.abandono1(atendido, training=training))    # residual + norma
        denso = self.densa2(self.densa1(x))
        return self.norma2(x + self.abandono2(denso, training=training))    # residual + norma

    def get_config(self):
        """Datos para volver a crear la capa al cargar el modelo guardado."""
        return {**super().get_config(), "dim": self.dim, "cabezas": self.cabezas,
                "dim_densa": self.dim_densa, "abandono": self.abandono, "causal": self.causal}


@keras.saving.register_keras_serializable(package="proyecto")
class PromedioSinRelleno(layers.Layer):
    """Promedia los vectores de las palabras reales, ignorando el relleno."""

    def call(self, x, indices):
        # peso vale 1 en las palabras reales y 0 en el relleno (índice 0)
        peso = ops.cast(ops.expand_dims(ops.not_equal(indices, 0), axis=-1), x.dtype)
        return ops.sum(x * peso, axis=1) / ops.maximum(ops.sum(peso, axis=1), 1.0)


def construir_transformer(largo, vocabulario, intenciones, dim=32, cabezas=2, dim_densa=64):
    """
    Arma el clasificador de intenciones uniendo las capas de arriba.
    Entrada: `largo` números (uno por palabra). Salida: una probabilidad por intención.
    """
    entrada = keras.Input(shape=(largo,), dtype="int32", name="pregunta")
    x = EmbeddingConPosicion(largo, vocabulario, dim)(entrada)      # números -> vectores
    x = BloqueTransformer(dim, cabezas, dim_densa)(x, entrada)      # cada palabra mira a las demás
    x = PromedioSinRelleno()(x, entrada)                            # un solo vector por pregunta
    x = layers.Dropout(0.2)(x)                                      # contra el sobreajuste
    # softmax: las salidas son positivas y suman 1, es decir, probabilidades
    salida = layers.Dense(intenciones, activation="softmax", name="intencion")(x)
    return keras.Model(entrada, salida, name="transformer_asistente")
