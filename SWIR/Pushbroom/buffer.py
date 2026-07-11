# =============================================================================
# buffer.py — Rolling window con metadatos de sincronización temporal
# =============================================================================
# Responsabilidades:
#   1. Acumular líneas espectrales (Y_util, N_BANDS) en un buffer circular.
#   2. Exponer el frame 2D actual como (N_LINES, Y_util, N_BANDS).
#   3. Mantener un índice global de líneas para sincronización temporal.
#   4. Proveer la función de mapeo: columna_X_en_frame → índice_linea_global.
#
# CONCEPTO CLAVE — SINCRONIZACIÓN TEMPORAL:
#   En pushbroom, el eje X del frame reconstruido NO es espacio puro:
#   es tiempo convertido en espacio por el movimiento de la banda.
#
#   Cuando YOLO detecta un bbox con x1=150, x2=280, eso significa que
#   la mandarina cruzó la línea de captura entre las líneas globales:
#     linea_global_x1 = linea_global_actual - (BUFFER_N_LINES - x1)
#     linea_global_x2 = linea_global_actual - (BUFFER_N_LINES - x2)
#
#   El ROI espectral para el clasificador debe extraerse exactamente
#   de esas líneas, no del frame completo.
#
#   Esta clase mantiene ese mapeo de forma precisa y thread-safe.
# =============================================================================

import threading
import numpy as np
from typing import Tuple, Optional
import config


class RollingBuffer:
    """
    Buffer circular de líneas espectrales con sincronización temporal.

    Internamente mantiene:
      - _buffer: array (N_LINES, Y_util, N_BANDS) float32 — el frame actual
      - _linea_global: contador monotónico de líneas desde el inicio
      - _linea_inicio_buffer: línea global que corresponde a la columna X=0
        del frame actual

    El frame se construye en orientación (columnas=tiempo, filas=Y_espacial):
      frame[x, y, b] = píxel en posición temporal x, altura y, banda b

    NOTA DE ORIENTACIÓN:
      La columna X=0 es la línea MÁS ANTIGUA del buffer.
      La columna X=N_LINES-1 es la línea MÁS NUEVA (la última capturada).
      Las mandarinas avanzan de izquierda a derecha en la visualización.
    """

    def __init__(self):
        self._n_lineas = config.BUFFER_N_LINES
        self._y_util   = config.BUFFER_Y_END - config.BUFFER_Y_START
        self._n_bandas = config.N_BANDS

        # Buffer principal: (N_LINES, Y_util, N_BANDS) float32
        # Inicializado en cero — representa fondo vacío.
        self._buffer = np.zeros(
            (self._n_lineas, self._y_util, self._n_bandas),
            dtype=np.float32
        )

        # Índice circular interno (apunta a la próxima posición a escribir)
        self._idx_escritura: int = 0

        # Contador global monotónico de líneas procesadas
        self._linea_global: int = 0

        # Lock para acceso thread-safe al buffer
        self._lock = threading.RLock()

        # Flag para saber si el buffer está lleno por primera vez
        self._buffer_lleno: bool = False

        # Perfil de background (1280,) float32
        # Se establece una vez al inicio de la sesión con set_background()
        # Si no se ha calibrado, la corrección no se aplica.
        self._background: Optional[np.ndarray] = None
        self._y_start = config.BUFFER_Y_START
        self._y_end   = config.BUFFER_Y_END
        self._bandas  = config.SPECTRAL_BANDS

    def set_background(self, background: np.ndarray) -> None:
        """
        Establece el perfil de background para corrección en tiempo real.

        Args:
            background: array (1280,) float32 calculado por
                        camera.calibrar_background()

        La corrección se aplica en agregar_linea() sobre el frame completo
        antes de extraer las bandas — equivalente al procesamiento offline.
        """
        if background.shape != (config.CAMERA_N_BANDS,):
            raise ValueError(
                f"Shape de background incorrecto: {background.shape} "
                f"(esperado: ({config.CAMERA_N_BANDS},))"
            )
        self._background = background.astype(np.float32)
        print(f"[BUFFER] Background establecido — "
              f"mean={background.mean():.2f} max={background.max():.2f}")

    # ------------------------------------------------------------------
    # ESCRITURA
    # ------------------------------------------------------------------

    def agregar_linea(self, linea: np.ndarray) -> None:
        """
        Agrega una nueva línea al buffer circular.

        Args:
            linea: array (Y_util, N_BANDS) float32 normalizado

        La línea se inserta en la posición _idx_escritura y el puntero
        avanza. Cuando llega al final, vuelve a 0 (circular).
        """
        if linea.shape != (self._y_util, self._n_bandas):
            raise ValueError(
                f"Shape de línea incorrecto: {linea.shape} "
                f"(esperado: ({self._y_util}, {self._n_bandas}))"
            )

        with self._lock:
            # Background correction — se aplica si fue calibrado
            # La corrección usa solo los valores del perfil correspondientes
            # a las bandas de interés, en la región Y útil.
            # Equivalente offline: np.clip(banda - background[idx], 0, None)
            if self._background is not None:
                bg_bandas = np.array(
                    [self._background[idx] for idx in self._bandas],
                    dtype=np.float32
                )  # (N_BANDS,)
                linea = np.clip(linea - bg_bandas[np.newaxis, :], 0, None)

            self._buffer[self._idx_escritura] = linea
            self._idx_escritura = (self._idx_escritura + 1) % self._n_lineas
            self._linea_global += 1

            if not self._buffer_lleno and self._linea_global >= self._n_lineas:
                self._buffer_lleno = True

    # ------------------------------------------------------------------
    # LECTURA DEL FRAME ACTUAL
    # ------------------------------------------------------------------

    def get_frame(self) -> Tuple[np.ndarray, int]:
        """
        Retorna el frame 2D reconstruido y la línea global del inicio.

        Returns:
            frame: array (N_LINES, Y_util, N_BANDS) float32
                   Orientación: frame[0] = línea más antigua,
                                frame[N_LINES-1] = línea más nueva.
            linea_inicio: línea global que corresponde a frame[0].
                          Usada para mapear columnas X a líneas globales.

        IMPORTANTE: retorna una COPIA para evitar que el hilo de captura
        modifique el array mientras se procesa.
        """
        with self._lock:
            # Reconstruir en orden cronológico desde el buffer circular
            # idx_escritura apunta a la posición más antigua (próxima a sobrescribir)
            idx = self._idx_escritura

            if idx == 0:
                frame = self._buffer.copy()
            else:
                # Rotar para que quede [más_antigua, ..., más_nueva]
                frame = np.concatenate(
                    [self._buffer[idx:], self._buffer[:idx]],
                    axis=0
                )

            # Línea global de la primera columna del frame
            linea_inicio = max(0, self._linea_global - self._n_lineas)

            return frame, linea_inicio

    # ------------------------------------------------------------------
    # EXTRACCIÓN DE ROI ESPECTRAL CON SINCRONIZACIÓN TEMPORAL
    # ------------------------------------------------------------------

    def extraer_roi_espectral(
        self,
        bbox_x1: int,
        bbox_x2: int,
        bbox_y1: int,
        bbox_y2: int,
        linea_inicio_frame: int
    ) -> Optional[np.ndarray]:
        """
        Extrae el ROI espectral correspondiente a un bbox detectado por YOLO,
        usando las mismas líneas del buffer que generaron ese bbox.

        Args:
            bbox_x1, bbox_x2: columnas del bbox en el frame (eje temporal)
            bbox_y1, bbox_y2: filas del bbox en el frame (eje espacial Y)
            linea_inicio_frame: valor retornado por get_frame() en el mismo
                                instante en que YOLO procesó ese frame.
                                CRÍTICO: debe ser el valor del MISMO get_frame()
                                que generó el pseudo-RGB que YOLO vio.

        Returns:
            roi: array (bbox_h, bbox_w, N_BANDS) float32
                 En el mismo formato que los crops del entrenamiento.
                 Retorna None si el bbox está fuera del rango válido.

        SINCRONIZACIÓN TEMPORAL:
            Las columnas X del frame corresponden a líneas globales:
              linea_de_columna_x = linea_inicio_frame + x

            Entonces bbox_x1 → línea global (linea_inicio_frame + bbox_x1)
                      bbox_x2 → línea global (linea_inicio_frame + bbox_x2)

            Al momento de extraer el ROI, necesitamos esas líneas exactas
            del buffer. Si el buffer ya las sobreescribió (la mandarina
            entró hace más de N_LINES líneas), el ROI no es recuperable.
        """
        with self._lock:
            # Calcular líneas globales del bbox
            linea_global_x1 = linea_inicio_frame + bbox_x1
            linea_global_x2 = linea_inicio_frame + bbox_x2

            # Verificar que esas líneas aún están en el buffer
            linea_mas_antigua = max(0, self._linea_global - self._n_lineas)
            if linea_global_x1 < linea_mas_antigua:
                if config.LOG_LEVEL >= 1:
                    print(f"[BUFFER] ROI expirado: línea {linea_global_x1} "
                          f"ya fue sobreescrita (mínimo: {linea_mas_antigua})")
                return None

            # Mapear líneas globales a índices en el buffer circular
            # idx_en_buffer = linea_global % N_LINES
            # Pero necesitamos el rango [x1, x2] en el frame ordenado.
            # Como ya tenemos get_frame() que retorna ordenado, usamos
            # el offset relativo al inicio del frame.

            linea_inicio_actual = max(0, self._linea_global - self._n_lineas)
            offset_x1 = linea_global_x1 - linea_inicio_actual
            offset_x2 = linea_global_x2 - linea_inicio_actual

            # Clamping defensivo
            offset_x1 = max(0, min(offset_x1, self._n_lineas - 1))
            offset_x2 = max(0, min(offset_x2, self._n_lineas))

            if offset_x2 <= offset_x1:
                if config.LOG_LEVEL >= 1:
                    print(f"[BUFFER] ROI inválido: x1={offset_x1} >= x2={offset_x2}")
                return None

            # Clamping del eje Y
            y1 = max(0, min(bbox_y1, self._y_util - 1))
            y2 = max(0, min(bbox_y2, self._y_util))

            if y2 <= y1:
                return None

            # Reconstruir el frame ordenado para extraer el ROI
            # (igual que en get_frame pero sin copiar el frame completo)
            idx = self._idx_escritura
            if idx == 0:
                frame_ordenado = self._buffer
            else:
                frame_ordenado = np.concatenate(
                    [self._buffer[idx:], self._buffer[:idx]],
                    axis=0
                )

            # Extraer ROI: (temporal=X, espacial=Y, bandas)
            # frame_ordenado[x, y, b]
            roi_raw = frame_ordenado[offset_x1:offset_x2, y1:y2, :]
            # roi_raw.shape: (bbox_w, bbox_h, N_BANDS)

            # Transponer a (bbox_h, bbox_w, N_BANDS) para coincidir con
            # el shape del entrenamiento: (alto, ancho, N_BANDS)
            roi_raw_t = np.transpose(roi_raw, (1, 0, 2))
            # roi_raw_t.shape: (bbox_h, bbox_w, N_BANDS) float32 sin normalizar

        
            # Buffer entrega ROI crudo float32.
            # El preprocesamiento Vy (norm p1/p99.5 + gamma) se aplica
            # en main.py via preprocessing.preprocess_channels().
            roi_crudo = roi_raw_t.astype(np.float32)
        return roi_crudo

    # ------------------------------------------------------------------
    # UTILIDADES
    # ------------------------------------------------------------------

    def esta_listo(self) -> bool:
        """
        Retorna True cuando el buffer ha sido llenado al menos una vez.
        Antes de esto, el frame tiene ceros en las posiciones no escritas.
        Se recomienda esperar a que esté listo antes de iniciar YOLO.
        """
        return self._buffer_lleno

    def columna_x_a_linea_global(
        self,
        columna_x: int,
        linea_inicio_frame: int
    ) -> int:
        """
        Convierte una columna X del frame a su línea global correspondiente.

        Args:
            columna_x: posición X en el frame [0, N_LINES-1]
            linea_inicio_frame: retornado por get_frame() en el mismo instante

        Returns:
            Línea global (monotónica desde el inicio del sistema)
        """
        return linea_inicio_frame + columna_x

    @property
    def linea_global(self) -> int:
        """Número de líneas procesadas desde el inicio (thread-safe)."""
        with self._lock:
            return self._linea_global

    @property
    def n_lineas(self) -> int:
        return self._n_lineas

    @property
    def y_util(self) -> int:
        return self._y_util

    @property
    def n_bandas(self) -> int:
        return self._n_bandas
