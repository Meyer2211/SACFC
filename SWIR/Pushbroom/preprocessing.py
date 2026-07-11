# =============================================================================
# preprocessing.py — Preprocesamiento Vy compartido
# =============================================================================
# ÚNICA fuente de la matemática de normalización + gamma.
# Usada por DOS rutas distintas:
#   1. pseudo_rgb.py  → salida uint8 para YOLO y visualización
#   2. worker_clasificador (main.py) → salida float32 para CNN
#
# ORDEN DE OPERACIONES (no modificar):
#   1. Normalización por canal: percentil P_LOW / P_HIGH → clip [0,1]
#   2. Gamma directa: canal ** GAMMA  (>1 oscurece, aplicada DESPUÉS de normalizar)
#   3. Clip [0,1]
#
# Esta función se aplica UNA SOLA VEZ por ruta.
# Ninguna etapa posterior (CNN, YOLO) debe repetir normalización o gamma.
# =============================================================================

import numpy as np
import config


def norm_por_canal(canal: np.ndarray) -> np.ndarray:
    """
    Normalización percentil P_LOW/P_HIGH sobre un canal 2D.

    Args:
        canal: array 2D float32 crudo del sensor

    Returns:
        array float32 en [0, 1]
    """
    v1 = float(np.percentile(canal, config.PREP_P_LOW))
    v2 = float(np.percentile(canal, config.PREP_P_HIGH))
    if v2 <= v1:
        return np.zeros_like(canal, dtype=np.float32)
    return np.clip((canal.astype(np.float32) - v1) / (v2 - v1), 0.0, 1.0)


def aplicar_gamma(arr01: np.ndarray) -> np.ndarray:
    """
    Gamma directa sobre array en [0,1].
    Gamma > 1 oscurece las zonas medias.
    Se aplica DESPUÉS de la normalización por canal.

    Args:
        arr01: array float32 en [0, 1]

    Returns:
        array float32 en [0, 1] con gamma aplicada
    """
    return np.clip(np.clip(arr01, 0.0, 1.0) ** config.PREP_GAMMA, 0.0, 1.0)


def preprocess_channels(canales: list) -> np.ndarray:
    """
    Aplica norm_por_canal + gamma a cada canal y apila el resultado.

    Esta función es la ÚNICA implementación de la transformación Vy.
    Debe llamarse identicamente en entrenamiento e inferencia.

    Args:
        canales: lista de arrays 2D float32 crudos del sensor.
                 Cada elemento tiene shape (H, W).
                 El orden de la lista determina el orden de los canales
                 en el stack resultante.

    Returns:
        stack: array float32 en [0,1] con shape (H, W, N_canales).
               N_canales = len(canales).
               Para CNN: float32 directo al modelo.
               Para YOLO: convertir a uint8 (*255) después.

    NOTA: La corrección de background está desactivada (BACKGROUND_CORRECTION_ON=False)
    porque bajo normalización por canal es matemáticamente invariante.
    El switch queda disponible en config.py para futuras comparaciones.
    """
    procesados = []
    for c in canales:
        canal_norm  = norm_por_canal(c)   # ya convierte a float32 internamente
        canal_gamma = aplicar_gamma(canal_norm)
        procesados.append(canal_gamma)

    return np.stack(procesados, axis=-1)  # (H, W, N_canales) float32
