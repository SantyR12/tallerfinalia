"""
Paso 1 de la detección: encontrar lo que se mueve.

Se usa sustracción de fondo, una técnica clásica de procesamiento de imágenes
(no es una red neuronal ni un modelo preentrenado). El algoritmo MOG2 aprende,
píxel por píxel, cómo se ve la escena vacía; lo que se aparta de ese fondo es
"primer plano". Cada mancha de primer plano es un candidato a objeto.

Requiere cámara fija, como las cámaras de tránsito.
"""

import cv2
import numpy as np

import config


class DetectorMovimiento:
    def __init__(self, fps=30.0):
        self.fps = fps
        # MOG2 incorpora al fondo lo que permanece quieto. Un objeto nuevo pasa
        # a ser fondo cuando su "peso" llega a 1 - PROPORCION_FONDO, y el peso
        # crece `tasa` por fotograma; de ahí sale la tasa para que tarde
        # SEGUNDOS_HASTA_FONDO.
        self.tasa = (1 - config.PROPORCION_FONDO) / (config.SEGUNDOS_HASTA_FONDO * fps)
        self.nucleo_pequeno = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.nucleo_grande = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        self._reiniciar_fondo()

    def _reiniciar_fondo(self):
        """Olvida el fondo aprendido y empieza a aprenderlo de nuevo."""
        self.sustractor = cv2.createBackgroundSubtractorMOG2(
            history=config.HISTORIAL_FONDO,
            varThreshold=config.UMBRAL_FONDO,
            detectShadows=config.DESCARTAR_SOMBRAS,     # si se activa, marca las sombras con gris (127)
        )
        self.sustractor.setBackgroundRatio(config.PROPORCION_FONDO)
        # Durante el primer segundo el fondo se aprende rápido; después, despacio.
        self.por_aprender = int(config.SEGUNDOS_APRENDIZAJE * self.fps)
        self.frames_alterados = 0           # fotogramas seguidos con media imagen "en movimiento"

    def _igualar_brillo(self, imagen):
        """
        Las cámaras ajustan solas la exposición y toda la imagen se aclara u
        oscurece un poco. Para que eso no parezca movimiento, cada fotograma se
        lleva al brillo del fondo aprendido. La corrección es la mediana, sobre
        toda la imagen, del cociente entre el fondo y el fotograma: un cambio
        de exposición altera todos los píxeles en la misma proporción, mientras
        que un objeto, por grande que sea, solo altera los que tapa.
        """
        fondo = self.sustractor.getBackgroundImage()
        ahora = cv2.cvtColor(imagen[::4, ::4], cv2.COLOR_BGR2GRAY).astype(np.float32) + 1.0
        antes = cv2.cvtColor(fondo[::4, ::4], cv2.COLOR_BGR2GRAY).astype(np.float32) + 1.0
        ganancia = min(max(float(np.median(antes / ahora)), 0.8), 1.25)
        return cv2.convertScaleAbs(imagen, alpha=ganancia)

    def detectar(self, frame, proteger=()):
        """
        Devuelve la lista de cajas (x1, y1, x2, y2) de las zonas en movimiento.

        proteger: cajas dentro de las cuales no se actualiza el fondo. Solo se
        usa con la opción PROTECCION_S de config.py, que viene apagada; sin
        cajas, el fondo se compara y se actualiza en un solo paso.
        """
        # El movimiento se busca siempre en una imagen del mismo ancho. Así los
        # filtros (que se miden en píxeles) actúan igual en un video pequeño
        # que en uno de alta resolución, y el proceso es más rápido.
        escala = min(1.0, config.ANCHO_DE_TRABAJO / frame.shape[1])
        if escala < 1.0:
            frame = cv2.resize(frame, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
        # Suavizar antes de comparar: el ruido de la cámara y de la compresión
        # hace "parpadear" los bordes muy contrastados (líneas de la vía).
        suave = cv2.GaussianBlur(frame, (config.SUAVIZADO, config.SUAVIZADO), 0)
        if self.por_aprender > 0:
            # Aprendizaje rápido del fondo (tasa -1 = automática). En este
            # tramo todavía no se entregan detecciones.
            self.por_aprender -= 1
            self.sustractor.apply(suave, learningRate=-1)
            return []
        suave = self._igualar_brillo(suave)
        if not proteger:
            mascara = self.sustractor.apply(suave, learningRate=self.tasa)
        else:
            # Comparar con el fondo sin aprender, y después aprender con el
            # fotograma en el que las zonas protegidas muestran el fondo conocido
            mascara = self.sustractor.apply(suave, learningRate=0)
            fondo = self.sustractor.getBackgroundImage()
            para_aprender = suave.copy()
            for x1, y1, x2, y2 in proteger:
                x1, y1 = max(int(x1 * escala), 0), max(int(y1 * escala), 0)
                x2, y2 = int(x2 * escala) + 1, int(y2 * escala) + 1
                para_aprender[y1:y2, x1:x2] = fondo[y1:y2, x1:x2]
            self.sustractor.apply(para_aprender, learningRate=self.tasa)

        # 255 = movimiento, 127 = sombra, 0 = fondo. Se descartan las sombras.
        _, mascara = cv2.threshold(mascara, 200, 255, cv2.THRESH_BINARY)

        # Si buena parte de la imagen aparece "en movimiento" durante un rato,
        # no son objetos: la cámara se movió o cambió la luz. El fondo aprendido
        # ya no sirve y se aprende otra vez.
        if cv2.countNonZero(mascara) > config.CAMBIO_GLOBAL * mascara.size:
            self.frames_alterados += 1
            if self.frames_alterados >= config.SEGUNDOS_CAMBIO_GLOBAL * self.fps:
                self._reiniciar_fondo()
                return []
        else:
            self.frames_alterados = 0

        # Apertura: borra puntos sueltos. Cierre: rellena huecos dentro del objeto.
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, self.nucleo_pequeno)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, self.nucleo_grande)

        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_total = frame.shape[0] * frame.shape[1]
        cajas = []
        for contorno in contornos:
            x, y, ancho, alto = cv2.boundingRect(contorno)
            fraccion = (ancho * alto) / area_total
            if not config.AREA_MINIMA <= fraccion <= config.AREA_MAXIMA:
                continue                    # muy pequeña (ruido) o enorme (cambio de luz)
            # Un objeto real llena buena parte de su caja; una línea delgada
            # en diagonal ocupa una caja grande pero casi vacía.
            if cv2.contourArea(contorno) < config.RELLENO_MINIMO * ancho * alto:
                continue
            # Las cajas se devuelven en las medidas del fotograma original
            cajas.append((x / escala, y / escala, (x + ancho) / escala, (y + alto) / escala))
        return cajas
