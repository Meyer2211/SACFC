# =============================================================================
# pseudo_rgb.py — Construcción del pseudo-RGB Vy para YOLO y visualización
# =============================================================================
# Responsabilidades:
#   1. Tomar el frame (N_LINES, Y_util, N_BANDS) del buffer.
#   2. Seleccionar canales R, G, B según PSEUDO_RGB_BAND_INDICES.
#   3. Aplicar preprocesamiento Vy via preprocessing.preprocess_channels().
#   4. Retornar imagen (Y_util, N_LINES, 3) uint8 para YOLO y display.
#
# NORMALIZACIÓN:
#   Se usa preprocessing.preprocess_channels() — misma función que la CNN.
#   Garantiza que YOLO y la CNN reciben datos preprocesados identicamente.
#   Ver preprocessing.py para la especificación matemática completa.
#
# NOTA IMPORTANTE — DISPLAY Y YOLO RECIBEN LA MISMA IMAGEN:
#   El canvas de visualización usa exactamente el mismo pseudo_rgb uint8
#   que se entrega al detector YOLO. No existe una ruta de display separada.
#   Cualquier modificación a construir_pseudo_rgb() afecta simultáneamente
#   lo que ve el operador en pantalla y lo que procesa YOLO.
#   El buffer almacena (N_LINES, Y_util, N_BANDS) donde:
#     - eje 0 (N_LINES) = dirección temporal / avance de la banda
#     - eje 1 (Y_util)  = ancho espacial de la banda transportadora
#   Para YOLO necesitamos (Y_util, N_LINES, 3) → se transpone cada canal.
# =============================================================================

import numpy as np
import cv2
import config
from preprocessing import preprocess_channels


def construir_pseudo_rgb(frame: np.ndarray) -> np.ndarray:
    """
    Construye imagen pseudo-RGB Vy a partir del frame del buffer.

    Args:
        frame: array (N_LINES, Y_util, N_BANDS) float32 crudo del buffer.

    Returns:
        pseudo_rgb: array (Y_util, N_LINES, 3) uint8
                    Canal 0 (R) ← OFFSETS_STACK[PSEUDO_RGB_BAND_INDICES[0]] = pico+11
                    Canal 1 (G) ← OFFSETS_STACK[PSEUDO_RGB_BAND_INDICES[1]] = pico+2
                    Canal 2 (B) ← OFFSETS_STACK[PSEUDO_RGB_BAND_INDICES[2]] = pico-5

    PROCESO:
        1. Extraer canales R, G, B del frame y transponer a (Y_util, N_LINES).
        2. preprocess_channels(): norm por canal p1/p99.5 + gamma 1.440.
        3. Escalar a uint8.
    """
    idx_r = config.PSEUDO_RGB_BAND_INDICES[0]
    idx_g = config.PSEUDO_RGB_BAND_INDICES[1]
    idx_b = config.PSEUDO_RGB_BAND_INDICES[2]

    # Extraer y transponer: (N_LINES, Y_util) → (Y_util, N_LINES)
    canal_r = frame[:, :, idx_r].T.astype(np.float32)
    canal_g = frame[:, :, idx_g].T.astype(np.float32)
    canal_b = frame[:, :, idx_b].T.astype(np.float32)

    # Preprocesamiento Vy: norm por canal + gamma (misma función que CNN)
    rgb01 = preprocess_channels([canal_r, canal_g, canal_b])
    # rgb01: (Y_util, N_LINES, 3) float32 en [0,1]

    # Convertir a uint8 para YOLO y display
    return (rgb01 * 255.0).clip(0, 255).astype(np.uint8)


def pseudo_rgb_para_display(pseudo_rgb: np.ndarray,
                             target_h: int,
                             target_w: int) -> tuple:
    """
    Redimensiona el pseudo-RGB al espacio disponible para el panel izquierdo,
    conservando la relación de aspecto y rellenando con negro si es necesario.

    Args:
        pseudo_rgb: array (Y_util, N_LINES, 3) uint8
        target_h: alto disponible en píxeles
        target_w: ancho disponible en píxeles

    Returns:
        canvas: array (target_h, target_w, 3) uint8
    """
    h_src, w_src = pseudo_rgb.shape[:2]

    # Escalar conservando aspecto
    scale = min(target_w / w_src, target_h / h_src)
    new_w = int(w_src * scale)
    new_h = int(h_src * scale)

    resized = cv2.resize(pseudo_rgb, (new_w, new_h),
                         interpolation=cv2.INTER_LINEAR)

    # Canvas negro del tamaño exacto
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    # Centrar la imagen en el canvas
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    return canvas, scale, x_off, y_off


def banda_para_segmentacion(frame: np.ndarray, idx_banda: int = 0) -> np.ndarray:
    """
    Extrae una banda individual para diagnóstico. No se usa en producción.

    Args:
        frame: (N_LINES, Y_util, N_BANDS) float32
        idx_banda: índice dentro de N_BANDS

    Returns:
        imagen (Y_util, N_LINES) uint8
    """
    banda = frame[:, :, idx_banda].T.astype(np.float32)
    p1  = float(np.percentile(banda, 1))
    p99 = float(np.percentile(banda, 99))
    if p99 > p1:
        banda = np.clip((banda - p1) / (p99 - p1), 0, 1)
    return (banda * 255.0).astype(np.uint8)
