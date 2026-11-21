"""
Análisis de eventos: colisiones, situaciones de riesgo y caídas.

Este módulo no usa redes neuronales. Recibe, fotograma a fotograma, los objetos
que entrega el detector (con su ID de seguimiento) y aplica reglas geométricas
sobre sus trayectorias:

- COLISIÓN: dos objetos se acercaban, entran en contacto a la misma
  profundidad (sus cajas se solapan o sus manchas se funden en una) y, justo
  después, quedan detenidos juntos o al menos uno cambia bruscamente de
  movimiento.
- RIESGO: dos objetos todavía no se tocan, pero si siguen como van se
  encontrarían en menos de un segundo.
- CAÍDA: la caja de una persona pasa de vertical (de pie) a horizontal (en el
  suelo) en poco tiempo.

Unidades: las velocidades se expresan en "tamaños por segundo" (píxeles por
segundo divididos entre el tamaño de la caja), para que los umbrales no
dependan de la resolución ni de qué tan lejos esté el objeto.
"""

import math
from collections import Counter, deque
from dataclasses import asdict, dataclass
from statistics import median

import config


# ----------------------------------------------------------------------------
# Estructuras de datos
# ----------------------------------------------------------------------------
@dataclass
class Evento:
    """Un evento detectado, listo para mostrarse, guardarse o consultarse."""

    tipo: str           # "colision", "riesgo" o "caida"
    nivel: str          # "ALTO" o "MEDIO"
    frame: int
    tiempo_s: float
    ids: tuple          # IDs de los objetos involucrados
    clases: tuple       # clase de cada objeto involucrado
    descripcion: str

    def como_dict(self):
        datos = asdict(self)
        datos["ids"] = list(self.ids)
        datos["clases"] = list(self.clases)
        return datos


class Pista:
    """Historial reciente de un objeto seguido (una "pista" o track)."""

    def __init__(self, id_pista, fps):
        self.id = id_pista
        largo = max(4, int(config.MEMORIA_PISTA_S * fps))
        self.historial = deque(maxlen=largo)   # (frame, x1, y1, x2, y2)
        self.votos_clase = Counter()           # la clase puede variar entre fotogramas
        self.observaciones = 0
        self.caida_reportada = False
        self.estimada = False                  # la última caja es una predicción
        self.oculta_en = None                  # ID del objeto con el que se fundió
        self.en_borde = False                  # la caja está cortada por el borde de la imagen
        self.salto = -10**9                    # último fotograma con un cambio brusco de caja
        self.ultimo_borde = -10**9             # último fotograma con la caja cortada por el borde
        self.impacto_reportado = False

    def agregar(self, frame, clase, caja, estimada=False, oculta_en=None, en_borde=False):
        if self.historial:
            # Si el área cambia de golpe (dos manchas que se funden o se
            # separan) o la caja pasa de real a estimada, el desplazamiento
            # del centro no es movimiento real: se anota el fotograma.
            _, x1, y1, x2, y2 = self.historial[-1]
            razon = ((caja[2] - caja[0]) * (caja[3] - caja[1])) / max((x2 - x1) * (y2 - y1), 1.0)
            if not 1 / config.SALTO_DE_AREA <= razon <= config.SALTO_DE_AREA or estimada != self.estimada:
                self.salto = frame
        self.historial.append((frame, *caja))
        self.votos_clase[clase] += 1
        self.observaciones += 1
        self.estimada = estimada
        self.oculta_en = oculta_en
        self.en_borde = en_borde
        if en_borde:
            self.ultimo_borde = frame

    def estable(self, frame, k):
        """True si la velocidad de los últimos k pasos es confiable."""
        return not self.estimada and self.observaciones > k and frame - self.salto > k

    def muestra(self, saltar=0):
        """Caja de hace `saltar` observaciones (0 = la última)."""
        return self.historial[max(-1 - saltar, -len(self.historial))][1:]

    # --- propiedades de la última observación ---
    @property
    def clase(self):
        return self.votos_clase.most_common(1)[0][0]

    @property
    def ultimo_frame(self):
        return self.historial[-1][0]

    @property
    def caja(self):
        return self.historial[-1][1:]

    @property
    def centro(self):
        x1, y1, x2, y2 = self.caja
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def ancho(self):
        return self.caja[2] - self.caja[0]

    @property
    def alto(self):
        return self.caja[3] - self.caja[1]

    @property
    def tam(self):
        """Tamaño característico: raíz del área de la caja."""
        return math.sqrt(max(self.ancho * self.alto, 1.0))

    @property
    def base(self):
        """Borde inferior de la caja: el punto donde el objeto toca el suelo."""
        return self.caja[3]

    def velocidad(self, k, saltar=0):
        """
        Velocidad (vx, vy) en píxeles por fotograma, promediada en k pasos.
        Con saltar=1 se ignora la última observación (sirve para conocer la
        velocidad que traía un objeto justo antes de fundirse con otro).
        """
        disponibles = len(self.historial) - saltar
        if disponibles < 2:
            return (0.0, 0.0)
        k = min(k, disponibles - 1)
        f0, a1, b1, a2, b2 = self.historial[-1 - saltar - k]
        f1, x1, y1, x2, y2 = self.historial[-1 - saltar]
        dt = max(f1 - f0, 1)
        return (((x1 + x2) - (a1 + a2)) / 2 / dt, ((y1 + y2) - (b1 + b2)) / 2 / dt)

    def rapidez(self, k, fps, saltar=0):
        """Rapidez en tamaños por segundo."""
        vx, vy = self.velocidad(k, saltar)
        x1, y1, x2, y2 = self.muestra(saltar)
        tam = math.sqrt(max((x2 - x1) * (y2 - y1), 1.0))
        return math.hypot(vx, vy) * fps / tam


# ----------------------------------------------------------------------------
# Funciones geométricas
# ----------------------------------------------------------------------------
def nombre(pista):
    """'vehículo 4': la clase del objeto como se escribe en los mensajes, y su ID."""
    return f"{config.NOMBRE_CLASE.get(pista.clase, pista.clase)} {pista.id}"


def solape(caja_a, caja_b):
    """Área de intersección dividida entre el área de la caja más pequeña."""
    ax1, ay1, ax2, ay2 = caja_a
    bx1, by1, bx2, by2 = caja_b
    ancho = min(ax2, bx2) - max(ax1, bx1)
    alto = min(ay2, by2) - max(ay1, by1)
    if ancho <= 0 or alto <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return (ancho * alto) / max(min(area_a, area_b), 1.0)


def misma_profundidad(base_a, alto_a, base_b, alto_b):
    """
    Dos objetos están a la misma distancia de la cámara si sus bases quedan a
    una altura parecida en la imagen. Evita confundir un choque con un objeto
    que simplemente pasa por delante de otro que está más lejos.
    """
    return abs(base_a - base_b) <= config.TOLERANCIA_PROFUNDIDAD * min(alto_a, alto_b)


def angulo_entre(v1, v2):
    """Ángulo en grados entre dos vectores de velocidad."""
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    coseno = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, coseno))))


# ----------------------------------------------------------------------------
# Analizador
# ----------------------------------------------------------------------------
class AnalizadorEventos:
    """Mantiene las pistas y aplica las reglas en cada fotograma."""

    def __init__(self, fps=30.0):
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.k = max(2, round(config.VENTANA_VELOCIDAD_S * self.fps))
        self.pistas = {}            # id -> Pista (objetos visibles hace poco)
        self.contactos = {}         # (id_a, id_b) -> datos del contacto en curso
        self.riesgo_reciente = {}   # id -> momento de su última alerta de riesgo
        self.anuncios = {}          # id -> cómo estaba el objeto cuando se avisó del riesgo
        self.ultima_alerta = {}     # (tipo, ids) -> tiempo de la última alerta
        self.ultima_colision = {}   # id -> momento de la última colisión en que participó
        self.registro = {}          # id -> (clase, observaciones), para el resumen
        self.eventos = []           # todos los eventos detectados
        self.frames_procesados = 0

    # ------------------------------------------------------------------ API
    def actualizar(self, frame, detecciones):
        """
        Recibe las detecciones de un fotograma y devuelve los eventos nuevos.
        `detecciones` es una lista de objetos con atributos id, clase y caja.
        """
        self.frames_procesados += 1
        tiempo = frame / self.fps

        # 1. Actualizar el historial de cada objeto
        for d in detecciones:
            pista = self.pistas.get(d.id)
            if pista is None:
                pista = self.pistas[d.id] = Pista(d.id, self.fps)
            pista.agregar(frame, d.clase, d.caja,
                          estimada=getattr(d, "estimada", False),
                          oculta_en=getattr(d, "oculta_en", None),
                          en_borde=getattr(d, "en_borde", False))
            self.registro[d.id] = (pista.clase, pista.observaciones)

        # 2. Olvidar los objetos que ya no se ven
        limite = frame - config.MEMORIA_PISTA_S * self.fps
        for id_pista in [i for i, p in self.pistas.items() if p.ultimo_frame < limite]:
            del self.pistas[id_pista]
        for clave in [c for c in self.contactos if not set(c) <= self.pistas.keys()]:
            del self.contactos[clave]

        visibles = sorted(
            (p for p in self.pistas.values() if p.ultimo_frame == frame),
            key=lambda p: p.id,
        )
        nuevos = []

        # 3. Revisar cada pareja de objetos
        for i, a in enumerate(visibles):
            for b in visibles[i + 1:]:
                if a.clase == "persona" and b.clase == "persona":
                    continue    # dos peatones juntos no son un evento de tránsito
                if a.en_borde or b.en_borde:
                    # Un objeto que entra o sale de la imagen tiene la caja
                    # cortada: su centro parece frenar y el que viene detrás
                    # parece alcanzarlo. Con esos datos no se evalúa nada.
                    continue
                # Fundidos: la mancha de uno quedó dentro de la del otro
                fundidos = a.oculta_en == b.id or b.oculta_en == a.id
                if (a.estimada or b.estimada) and not fundidos:
                    continue    # uno está oculto en un tercero: no hay datos reales
                en_contacto = fundidos or (
                    solape(a.caja, b.caja) >= config.SOLAPE_MINIMO
                    and misma_profundidad(a.base, a.alto, b.base, b.alto)
                )
                if en_contacto or (a.id, b.id) in self.contactos:
                    # Se tocan, o se tocaron hace un momento: tras un golpe los
                    # objetos pueden separarse, así que se sigue observando.
                    evento = self._revisar_colision(a, b, en_contacto, fundidos, frame, tiempo)
                else:
                    evento = self._revisar_riesgo(a, b, frame, tiempo)
                if evento:
                    nuevos.append(evento)

        # 4. Revisar frenadas imposibles (choque contra algo que no se seguía)
        for p in visibles:
            if p.clase in config.VEHICULOS_MOTOR:
                evento = self._revisar_impacto(p, frame, tiempo)
                if evento:
                    nuevos.append(evento)

        # 5. Revisar si un choque anunciado ocurrió aunque se haya perdido a uno de los dos
        for p in visibles:
            evento = self._revisar_riesgo_cumplido(p, visibles, frame, tiempo)
            if evento:
                nuevos.append(evento)

        # 6. Revisar caídas de personas
        for p in visibles:
            if p.clase == "persona" and not p.estimada:
                evento = self._revisar_caida(p, visibles, frame, tiempo)
                if evento:
                    nuevos.append(evento)

        self.eventos.extend(nuevos)
        return nuevos

    def resumen(self):
        """Estadísticas del video procesado hasta el momento."""
        objetos = Counter(
            clase
            for clase, veces in self.registro.values()
            if veces >= config.FRAMES_MINIMOS_CONTEO
        )
        tipos = Counter(e.tipo for e in self.eventos)
        return {
            "frames": self.frames_procesados,
            "duracion_s": round(self.frames_procesados / self.fps, 2),
            "fps": round(self.fps, 2),
            "objetos": dict(objetos),
            "eventos": {t: tipos.get(t, 0) for t in ("colision", "riesgo", "caida")},
        }

    # -------------------------------------------------------------- reglas
    def _movimiento_relativo(self, a, b, saltar=0):
        """
        Devuelve (p, v, cierre):
        p = posición de b respecto a a (píxeles)
        v = velocidad de b respecto a a (píxeles por fotograma)
        cierre = velocidad de acercamiento en tamaños por segundo (positiva si
                 se acercan)
        Con saltar=1 se calcula con los datos del fotograma anterior.
        """
        ax1, ay1, ax2, ay2 = a.muestra(saltar)
        bx1, by1, bx2, by2 = b.muestra(saltar)
        (avx, avy), (bvx, bvy) = a.velocidad(self.k, saltar), b.velocidad(self.k, saltar)
        p = ((bx1 + bx2 - ax1 - ax2) / 2, (by1 + by2 - ay1 - ay2) / 2)
        v = (bvx - avx, bvy - avy)
        distancia = math.hypot(*p)
        if distancia == 0:
            return p, v, 0.0
        acercamiento = -(p[0] * v[0] + p[1] * v[1]) / distancia    # px por fotograma
        tam_a = math.sqrt(max((ax2 - ax1) * (ay2 - ay1), 1.0))
        tam_b = math.sqrt(max((bx2 - bx1) * (by2 - by1), 1.0))
        return p, v, acercamiento * self.fps / ((tam_a + tam_b) / 2)

    def _cierre_en_tamanos(self, a, b, saltar=0, completo=False):
        """
        Velocidad de acercamiento (en tamaños de objeto por segundo) descontando
        la perspectiva. Dos carros que se alejan de la cámara por la misma vía se
        juntan en la imagen aunque mantengan su distancia, porque todo se encoge
        hacia el horizonte. Como el tamaño de una caja es inversamente
        proporcional a la distancia a la cámara, la separación dividida entre el
        producto de los dos tamaños no cambia mientras la distancia real entre
        ellos no cambie. Se compara esa medida ahora y hace un momento.
        Con saltar=1 se calcula con los datos del fotograma anterior. Con
        completo=True devuelve también la separación actual, en tamaños.
        """
        def medidas(atras):
            ax1, ay1, ax2, ay2 = a.muestra(atras)
            bx1, by1, bx2, by2 = b.muestra(atras)
            distancia = math.hypot((ax1 + ax2 - bx1 - bx2) / 2, (ay1 + ay2 - by1 - by2) / 2)
            tam_a = math.sqrt(max((ax2 - ax1) * (ay2 - ay1), 1.0))
            tam_b = math.sqrt(max((bx2 - bx1) * (by2 - by1), 1.0))
            return distancia, tam_a * tam_b

        distancia, escala = medidas(saltar)
        distancia_antes, escala_antes = medidas(saltar + self.k)
        antes = distancia_antes * escala / escala_antes      # la de antes, a la escala de ahora
        cierre = (antes - distancia) / math.sqrt(escala) * self.fps / self.k
        return (cierre, distancia / math.sqrt(escala)) if completo else cierre

    def _revisar_colision(self, a, b, en_contacto, fundidos, frame, tiempo):
        """Colisión = venían acercándose + contacto + cambio brusco de movimiento."""
        clave = (a.id, b.id)
        estado = self.contactos.get(clave)

        if estado is None:
            # Primer fotograma en contacto: se guarda cómo venían moviéndose.
            # Si las manchas se fundieron, la última caja ya es la de los dos
            # juntos, así que se usan los datos del fotograma anterior.
            saltar = 1 if fundidos else 0
            ca, cb = a.muestra(saltar), b.muestra(saltar)
            # El contacto solo puede llegar a ser un choque si los dos objetos
            # tenían historial (no acaban de aparecer) y estaban a la misma
            # profundidad. Un motociclista sobre su moto, o un objeto al que el
            # seguimiento le cambió el ID, nunca se toma como choque.
            confiable = min(a.observaciones, b.observaciones) > self.k + saltar and (
                misma_profundidad(ca[3], ca[3] - ca[1], cb[3], cb[3] - cb[1])
            )
            # Deben venir acercándose en la imagen y también al descontar la perspectiva
            cierre = min(self._movimiento_relativo(a, b, saltar)[2],
                         self._cierre_en_tamanos(a, b, saltar)) if confiable else 0.0
            self.contactos[clave] = {
                "inicio": tiempo,
                "frames": 1,                # fotogramas en contacto
                "cierre": cierre,
                "previo": {                 # (rapidez, velocidad) antes del contacto
                    a.id: (a.rapidez(self.k, self.fps, saltar), a.velocidad(self.k, saltar)),
                    b.id: (b.rapidez(self.k, self.fps, saltar), b.velocidad(self.k, saltar)),
                },
                "con_historial": {          # ¿se venía siguiendo desde antes del contacto?
                    a.id: a.observaciones > self.k + saltar,
                    b.id: b.observaciones > self.k + saltar,
                },
                "fundidos": fundidos,
                "cambio": frame,            # último fotograma en que se fundieron o separaron
                "reportado": False,
            }
            return None

        if fundidos != estado["fundidos"]:
            estado["fundidos"] = fundidos
            estado["cambio"] = frame
        if en_contacto:
            estado["frames"] += 1
        transcurrido = tiempo - estado["inicio"]
        if transcurrido > config.VENTANA_CONFIRMACION_S and not en_contacto:
            del self.contactos[clave]       # se separaron y no pasó nada: se olvida
            return None
        # ¿El sistema ya había avisado del riesgo para alguno de los dos? Entonces
        # el acercamiento quedó comprobado en ese momento, aunque después el
        # seguimiento haya perdido a uno de ellos y le haya dado otro número.
        anunciado = any(tiempo - self.riesgo_reciente.get(i, -1e9) <= config.RIESGO_MEMORIA_S
                        for i in (a.id, b.id))
        if (
            estado["reportado"]
            or (estado["cierre"] < config.V_CIERRE_MINIMA and not anunciado)   # no venían acercándose
            or estado["frames"] < 2                         # un roce de un solo fotograma
            or transcurrido < config.CONFIRMACION_S
            or transcurrido > config.VENTANA_CONFIRMACION_S
        ):
            return None

        motivo = None
        if fundidos:
            contenedor = a if b.oculta_en == a.id else b
            motivo = self._fundidos_tras_choque(contenedor, estado, frame)
            # Riesgo cumplido: se avisó del choque inminente, las manchas se
            # fundieron, siguen fundidas (dos objetos que solo se cruzan se
            # separan antes) y la mancha ya no se mueve como venía. Una moto que
            # alcanza a un carro y sigue a su lado no cambia el movimiento del carro.
            if (motivo is None and anunciado and transcurrido >= config.FUSION_SOSTENIDA_S
                    and estado["frames"] >= 0.8 * transcurrido * self.fps):
                otro = b if contenedor is a else a
                referencia = contenedor if estado["con_historial"][contenedor.id] else otro
                antes = estado["previo"][referencia.id][0]
                ahora = contenedor.rapidez(round(0.5 * self.fps), self.fps)
                if abs(ahora - antes) >= max(config.V_MOVIMIENTO, 0.4 * antes):
                    motivo = "siguen juntos tras el contacto anunciado y cambiaron su movimiento"
        elif frame - estado["cambio"] > self.k:     # cajas ya estables tras separarse
            for pista in (a, b):
                motivo = self._cambio_brusco(pista, *estado["previo"][pista.id])
                if motivo:
                    motivo = f"{nombre(pista)} {motivo}"
                    break
        if motivo is None:
            return None

        estado["reportado"] = True
        descripcion = (
            f"Posible colisión entre {nombre(a)} y {nombre(b)} "
            f"en el segundo {tiempo:.1f} ({motivo})"
        )
        return self._emitir("colision", "ALTO", frame, tiempo, (a, b), descripcion)

    def _fundidos_tras_choque(self, contenedor, estado, frame):
        """
        Dos objetos cuyas manchas se fundieron chocaron si la mancha conjunta,
        ya con tamaño estable, queda quieta, o si el objeto que estaba quieto
        sale empujado. En un simple cruce, en cambio, la mancha conjunta sigue
        cambiando de tamaño mientras uno pasa junto al otro. Devuelve el motivo
        (texto) o None.
        """
        m = max(self.k, round(config.CONFIRMACION_S * self.fps))
        if frame - estado["cambio"] < m:
            return None                     # aún no hay suficientes datos tras fundirse
        recientes = list(contenedor.historial)[-(m + 1):]
        areas = [(h[3] - h[1]) * (h[4] - h[2]) for h in recientes]
        variacion = (max(areas) - min(areas)) / (sum(areas) / len(areas))
        if variacion > config.VARIACION_ESTABLE:
            return None                     # sigue cambiando: se están cruzando

        # El que estaba quieto ahora avanza: lo empujaron
        rapidez_propia = estado["previo"][contenedor.id][0]
        if (rapidez_propia < config.V_MOVIMIENTO
                and contenedor.rapidez(self.k, self.fps) >= config.V_CIERRE_MINIMA):
            return f"{nombre(contenedor)} salió impulsado"

        rapidez_antes = max(math.hypot(*velocidad) for _, velocidad in estado["previo"].values())
        rapidez_ahora = math.hypot(*contenedor.velocidad(self.k))      # píxeles por fotograma
        venian_moviendose = max(r for r, _ in estado["previo"].values()) >= config.V_MOVIMIENTO
        # Una moto que alcanza a un carro lento también se funde con él, y la
        # mancha conjunta va más despacio que la moto; pero sigue avanzando.
        # Para que sea un choque, la mancha debe quedar prácticamente quieta.
        casi_quieta = contenedor.rapidez(self.k, self.fps) < config.V_MOVIMIENTO
        if (venian_moviendose and casi_quieta
                and rapidez_ahora <= (1 - config.FRENADO_MINIMO) * rapidez_antes):
            return "quedaron detenidos tras el contacto"
        return None

    def _revisar_riesgo_cumplido(self, pista, visibles, frame, tiempo):
        """
        El sistema avisó de un choque inminente entre dos objetos y después el
        seguimiento perdió a uno (algo habitual: el que venía rápido se funde
        con el otro sin que se alcance a registrar el contacto). Si la mancha
        del que quedó crece de golpe, sigue así un buen rato y ya no se mueve
        como venía, el choque ocurrió.
        """
        anuncio = self.anuncios.get(pista.id)
        if anuncio is None:
            return None
        if tiempo - anuncio["tiempo"] > config.RIESGO_MEMORIA_S + config.FUSION_SOSTENIDA_S:
            del self.anuncios[pista.id]             # pasó el tiempo y no ocurrió nada
            return None
        if pista.estimada or pista.en_borde:
            return None
        # Si el otro sigue a la vista, entero y separado, no se fundieron
        otro = self.pistas.get(anuncio["otro"])
        if (otro is not None and otro in visibles and not otro.estimada
                and solape(pista.caja, otro.caja) == 0):
            anuncio["fusion"] = None
            return None
        # ¿La mancha creció (algo se le fundió) y sigue crecida?
        if pista.tam ** 2 < config.SALTO_DE_AREA * anuncio["area"]:
            anuncio["fusion"] = None
            return None
        if anuncio["fusion"] is None:
            anuncio["fusion"] = frame
        if frame - anuncio["fusion"] < config.FUSION_SOSTENIDA_S * self.fps:
            return None
        # ¿Cambió su movimiento respecto a como venía cuando se dio el aviso?
        antes = anuncio["rapidez"]
        ahora = pista.rapidez(round(0.5 * self.fps), self.fps)
        if abs(ahora - antes) < max(config.V_MOVIMIENTO, 0.4 * antes):
            return None

        del self.anuncios[pista.id]
        self.anuncios.pop(anuncio["otro"], None)
        if self._choque_reciente((pista,), tiempo):     # no repetir un choque ya reportado
            return None
        participantes = (pista, otro) if otro is not None else (pista,)
        quienes = " y ".join(nombre(p) for p in participantes)
        descripcion = (
            f"Posible colisión entre {quienes} en el segundo {tiempo:.1f} "
            f"(ocurrió el contacto que se había anunciado)"
        ) if otro is not None else (
            f"Posible colisión de {quienes} en el segundo {tiempo:.1f} "
            f"(ocurrió el contacto que se había anunciado)"
        )
        return self._emitir("colision", "ALTO", frame, tiempo, participantes, descripcion)

    def _revisar_impacto(self, pista, frame, tiempo):
        """
        Frenada imposible: un vehículo que venía rápido queda casi quieto en una
        fracción de segundo. Ningún freno detiene un carro así; solo un golpe.
        Sirve para los choques contra un poste o un muro, y contra un vehículo
        que el sistema no venía siguiendo (acababa de entrar a la imagen).
        """
        m = round(config.IMPACTO_ESPERA_S * self.fps)
        necesarios = m + 3 * self.k
        if pista.rapidez(self.k, self.fps) >= config.V_MOVIMIENTO:
            pista.impacto_reportado = False     # volvió a moverse
        if (pista.impacto_reportado or pista.estimada or pista.oculta_en is not None
                or pista.observaciones <= necesarios or len(pista.historial) <= necesarios
                or frame - pista.ultimo_borde <= necesarios):
            return None
        # La velocidad no es confiable si la caja cambió de golpe dentro de la
        # ventana en que se mide (antes del golpe o después de él)
        if frame - pista.salto <= self.k or m < frame - pista.salto <= necesarios:
            return None
        # Debe venir rápido de forma sostenida (tres mediciones seguidas): un
        # salto aislado de la caja también parece velocidad, pero dura un instante.
        antes = min(pista.rapidez(self.k, self.fps, saltar=m + i * self.k) for i in range(3))
        ahora = pista.rapidez(self.k, self.fps)
        if not (antes >= config.IMPACTO_V_MINIMA and ahora < config.V_MOVIMIENTO):
            return None
        # Dos manchas que se cruzan también "se detienen" un instante (el centro
        # de la mancha conjunta casi no se mueve), pero su tamaño sigue cambiando.
        areas = [(h[3] - h[1]) * (h[4] - h[2]) for h in list(pista.historial)[-(self.k + 1):]]
        if (max(areas) - min(areas)) / (sum(areas) / len(areas)) > config.VARIACION_ESTABLE:
            return None

        pista.impacto_reportado = True
        # Si el choque ya se reportó por la regla de dos objetos, no se repite
        if self._choque_reciente((pista,), tiempo):
            return None
        # Si estaba en contacto con otro objeto seguido, el choque fue con él
        for (id_a, id_b), estado in self.contactos.items():
            if pista.id in (id_a, id_b) and not estado["reportado"]:
                estado["reportado"] = True
                a, b = self.pistas[id_a], self.pistas[id_b]
                descripcion = (
                    f"Posible colisión entre {nombre(a)} y {nombre(b)} "
                    f"en el segundo {tiempo:.1f} ({nombre(pista)} se detuvo de golpe)"
                )
                return self._emitir("colision", "ALTO", frame, tiempo, (a, b), descripcion)
        descripcion = (
            f"Posible colisión de {nombre(pista)} en el segundo {tiempo:.1f} "
            f"(se detuvo de golpe)"
        )
        return self._emitir("colision", "ALTO", frame, tiempo, (pista,), descripcion)

    def _cambio_brusco(self, pista, rapidez_previa, velocidad_previa):
        """Compara el movimiento actual de un objeto con el que traía antes del contacto."""
        rapidez = pista.rapidez(self.k, self.fps)
        se_movia = rapidez_previa >= config.V_MOVIMIENTO
        se_mueve = rapidez >= config.V_MOVIMIENTO

        if se_movia and rapidez <= (1 - config.FRENADO_MINIMO) * rapidez_previa:
            return "frenó de golpe"
        if se_movia and se_mueve:
            giro = angulo_entre(velocidad_previa, pista.velocidad(self.k))
            if giro >= config.GIRO_MINIMO_GRADOS:
                return "cambió bruscamente de dirección"
        if not se_movia and rapidez >= config.V_CIERRE_MINIMA:
            return "salió impulsado"
        return None

    def _revisar_riesgo(self, a, b, frame, tiempo):
        """Riesgo = si siguen como van, se encuentran en menos de TTC_UMBRAL_S."""
        if not ({a.clase, b.clase} & config.VEHICULOS_MOTOR):
            return None                     # debe haber al menos un vehículo de motor
        if not (a.estable(frame, self.k) and b.estable(frame, self.k)):
            return None                     # todavía no hay velocidad confiable
        p, v, cierre = self._movimiento_relativo(a, b)
        if cierre < config.V_CIERRE_MINIMA:
            return None                     # no se acercan lo suficiente
        cierre_real, separacion = self._cierre_en_tamanos(a, b, completo=True)
        if cierre_real < config.V_CIERRE_MINIMA:
            return None                     # se juntan en la imagen solo por la perspectiva
        if separacion / cierre_real > config.TTC_UMBRAL_S:
            return None                     # descontada la perspectiva, falta más para el contacto

        v2 = v[0] ** 2 + v[1] ** 2
        t_frames = -(p[0] * v[0] + p[1] * v[1]) / v2    # momento de máximo acercamiento
        ttc = t_frames / self.fps
        if not 0 < ttc <= config.TTC_UMBRAL_S:
            return None

        # ¿Qué tan cerca pasarían?
        cruce = math.hypot(p[0] + v[0] * t_frames, p[1] + v[1] * t_frames)
        if cruce > config.DISTANCIA_CRUCE * (a.tam + b.tam) / 2:
            return None

        # ¿Estarían a la misma profundidad en ese momento?
        base_a = a.base + a.velocidad(self.k)[1] * t_frames
        base_b = b.base + b.velocidad(self.k)[1] * t_frames
        if not misma_profundidad(base_a, a.alto, base_b, b.alto):
            return None

        descripcion = (
            f"Riesgo de colisión entre {nombre(a)} y {nombre(b)} "
            f"en el segundo {tiempo:.1f} (contacto estimado en {ttc:.1f} s)"
        )
        evento = self._emitir("riesgo", "MEDIO", frame, tiempo, (a, b), descripcion)
        if evento:
            # Se anota cómo estaba cada uno, para comprobar después si el
            # choque anunciado ocurrió (ver _revisar_riesgo_cumplido)
            for pista, otro in ((a, b), (b, a)):
                self.anuncios[pista.id] = {
                    "tiempo": tiempo, "otro": otro.id, "area": pista.tam ** 2,
                    "rapidez": pista.rapidez(self.k, self.fps), "fusion": None,
                }
        return evento

    def _revisar_caida(self, persona, visibles, frame, tiempo):
        """Caída = la caja de la persona pasa de vertical a horizontal rápidamente."""
        ventana = config.VENTANA_CAIDA_S * self.fps
        recientes = [h for h in persona.historial if h[0] >= frame - ventana]
        if len(recientes) < 6 or recientes[-1][0] - recientes[0][0] < ventana / 2:
            return None

        def aspecto(h):                     # ancho / alto
            return (h[3] - h[1]) / max(h[4] - h[2], 1.0)

        tercio = max(2, len(recientes) // 3)
        antes, ahora = recientes[:tercio], recientes[-tercio:]
        aspecto_antes = median(aspecto(h) for h in antes)
        aspecto_ahora = median(aspecto(h) for h in ahora)
        alto_antes = median(h[4] - h[2] for h in antes)
        alto_ahora = median(h[4] - h[2] for h in ahora)
        ancho_antes = median(h[3] - h[1] for h in antes)
        ancho_ahora = median(h[3] - h[1] for h in ahora)

        if aspecto_ahora < config.ASPECTO_DE_PIE:
            persona.caida_reportada = False     # está de pie otra vez
        if persona.caida_reportada:
            return None
        if persona.en_borde or persona.estimada or persona.oculta_en is not None:
            return None                     # caja cortada, estimada o fundida con otra: no es confiable
        if not (
            aspecto_antes <= config.ASPECTO_DE_PIE
            and aspecto_ahora >= config.ASPECTO_CAIDA
            and alto_antes >= config.ALTURA_MINIMA_PERSONA
        ):
            return None
        # Al caer, la caja se acorta y se ensancha, y su área no cambia mucho.
        # Si solo se ensancha (otra mancha se le pegó) o solo se acorta (algo
        # la tapa por debajo), no es una caída.
        razon_area = (ancho_ahora * alto_ahora) / max(ancho_antes * alto_antes, 1.0)
        if not (
            alto_ahora <= config.CAIDA_ALTO_MAXIMO * alto_antes
            and ancho_ahora >= config.CAIDA_ANCHO_MINIMO * ancho_antes
            and 1 / config.CAIDA_CAMBIO_AREA <= razon_area <= config.CAIDA_CAMBIO_AREA
        ):
            return None

        persona.caida_reportada = True
        # Si hay una moto o bicicleta pegada a la persona, era su conductor
        vehiculo = next(
            (
                v
                for v in visibles
                if v.clase in config.VEHICULOS_LIVIANOS
                and math.dist(v.centro, persona.centro) <= 1.5 * max(v.tam, persona.tam)
            ),
            None,
        )
        if vehiculo is not None:
            quien = "motociclista" if vehiculo.clase == "motocicleta" else "ciclista"
            involucrados = (persona, vehiculo)
        else:
            quien = "peatón"
            involucrados = (persona,)
        descripcion = (
            f"Posible caída de {quien} (persona {persona.id}) en el segundo {tiempo:.1f}"
        )
        return self._emitir("caida", "ALTO", frame, tiempo, involucrados, descripcion)

    # ------------------------------------------------------------- utilidades
    def _choque_reciente(self, pistas, tiempo):
        """¿Alguno de estos objetos participó en una colisión reportada hace poco?"""
        return any(tiempo - self.ultima_colision.get(p.id, -1e9) <= config.ENFRIAMIENTO_S
                   for p in pistas)

    def _emitir(self, tipo, nivel, frame, tiempo, pistas, descripcion):
        """Crea el evento, salvo que la misma alerta se haya dado hace muy poco."""
        ids = tuple(p.id for p in pistas)
        clave = (tipo, ids)
        anterior = self.ultima_alerta.get(clave)
        if anterior is not None and tiempo - anterior < config.ENFRIAMIENTO_S:
            return None
        if tipo == "colision":
            # Un mismo choque se reporta una sola vez. Al golpearse, un vehículo
            # puede quedar en contacto con dos manchas a la vez (el otro vehículo
            # y un pedazo suyo que el seguimiento tomó por un objeto nuevo), y
            # las dos parejas cumplen la regla en el mismo fotograma.
            if self._choque_reciente(pistas, tiempo):
                return None
            for p in pistas:
                self.ultima_colision[p.id] = tiempo
        self.ultima_alerta[clave] = tiempo
        if tipo == "riesgo":
            for p in pistas:
                self.riesgo_reciente[p.id] = tiempo
        return Evento(
            tipo=tipo,
            nivel=nivel,
            frame=frame,
            tiempo_s=round(tiempo, 2),
            ids=ids,
            clases=tuple(p.clase for p in pistas),
            descripcion=descripcion,
        )
