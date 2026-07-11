# =============================================================================
# classifier.py — Clasificador intercambiable con preprocesamiento Vy
# =============================================================================
# Soporta 3 modos controlados por config.py:
#
#   Experimento A — 5 bandas espectrales Vy:
#     CNN_INPUT_CHANNELS = 5
#     CLASSIFIER_CHANNEL_INDICES = [0, 1, 2, 3, 4]
#     pkl → rf_glcm_5ch.pkl  |  h5 → cnn_5ch.h5
#
#   Experimento B — 3 canales pseudo-RGB Vy:
#     CNN_INPUT_CHANNELS = 3
#     CLASSIFIER_CHANNEL_INDICES = [0, 1, 2]  → offsets [+11, +2, -5]
#     pkl → rf_glcm_3ch.pkl  |  h5 → cnn_3ch.h5
#
# PREPROCESAMIENTO:
#   El ROI que llega a predecir() ya viene preprocesado con la
#   transformación Vy (norm por canal p1/p99.5 + gamma 1.440)
#   aplicada en main.py via preprocessing.preprocess_channels().
#   NO se aplica ninguna normalización adicional aquí.
# =============================================================================

import numpy as np
import cv2
import joblib
from dataclasses import dataclass
from typing import Optional
import config

try:
    from skimage.feature import graycomatrix, graycoprops
    SKIMAGE_DISPONIBLE = True
except ImportError:
    SKIMAGE_DISPONIBLE = False
    print("[CLASSIFIER] ADVERTENCIA: skimage no disponible.")

ROI_CENTRO_FRACCION: float = 0.85


@dataclass
class ResultadoClasificacion:
    etiqueta: int
    probabilidad: float
    es_buena: bool
    modelo_usado: str


class Clasificador:
    """
    Clasificador intercambiable.
    Recibe ROI (H, W, N_BANDS) float32 en [0,1] ya preprocesado con Vy.
    Retorna ResultadoClasificacion.
    """

    def __init__(self):
        self._modelo = None
        self._tipo   = config.CLASSIFIER_TYPE
        self._n_ch   = config.CNN_INPUT_CHANNELS
        self._ch_idx = config.CLASSIFIER_CHANNEL_INDICES
        self._inicializado = False

    # ------------------------------------------------------------------
    # INICIALIZACIÓN
    # ------------------------------------------------------------------

    def iniciar(self) -> None:
        print(f"[CLASSIFIER] Tipo: '{self._tipo}' | "
              f"Canales: {self._n_ch} (índices: {self._ch_idx})")
        print(f"[CLASSIFIER] Modelo: {config.CLASSIFIER_MODEL_PATH}")

        if self._tipo == "pkl_rf_glcm":
            self._cargar_pkl()
        elif self._tipo == "h5_cnn":
            self._cargar_h5()
        elif self._tipo == "pt_torch":
            self._cargar_pt()
        else:
            raise ValueError(f"CLASSIFIER_TYPE desconocido: {self._tipo}")

        self._inicializado = True
        print("[CLASSIFIER] Modelo cargado correctamente.")
        self._warm_up()

    def _cargar_pkl(self) -> None:
        if not SKIMAGE_DISPONIBLE:
            raise RuntimeError("skimage requerido. pip install scikit-image")
        self._modelo = joblib.load(config.CLASSIFIER_MODEL_PATH)
        print(f"[CLASSIFIER] Pipeline sklearn: "
              f"{[s[0] for s in self._modelo.steps]}")

    def _cargar_h5(self) -> None:
        try:
            from tensorflow.keras.models import load_model as keras_load
            import tensorflow as tf
        except ImportError:
            raise RuntimeError("TensorFlow no instalado. pip install tensorflow")

        # SeleccionCanales: capa custom usada por los modelos Dual-Branch.
        # Definida aquí para que keras_load pueda reconstruirla correctamente.
        class SeleccionCanales(tf.keras.layers.Layer):
            def __init__(self, indices, **kwargs):
                super().__init__(**kwargs)
                self.indices = indices

            def call(self, x):
                return tf.stack([x[:, :, :, i] for i in self.indices], axis=-1)

            def compute_output_shape(self, input_shape):
                return (input_shape[0], input_shape[1],
                        input_shape[2], len(self.indices))

            def get_config(self):
                cfg = super().get_config()
                cfg['indices'] = self.indices
                return cfg

        self._modelo = keras_load(
            config.CLASSIFIER_MODEL_PATH,
            custom_objects={'SeleccionCanales': SeleccionCanales}
        )
        print("[CLASSIFIER] Modelo Keras cargado.")

    def _cargar_pt(self) -> None:
        raise NotImplementedError(
            "Soporte PyTorch pendiente. Implementar _predecir_pt()."
        )

    def _warm_up(self) -> None:
        dummy = np.random.rand(50, 40, config.N_BANDS).astype(np.float32)
        try:
            self.predecir(dummy)
            print("[CLASSIFIER] Warm-up completado.")
        except Exception as e:
            print(f"[CLASSIFIER] Warm-up falló: {e}")

    # ------------------------------------------------------------------
    # INFERENCIA
    # ------------------------------------------------------------------

    def predecir(self, roi: np.ndarray) -> ResultadoClasificacion:
        """
        Args:
            roi: (H, W, N_BANDS) float32 en [0,1] — preprocesado con Vy.
                 El clasificador selecciona internamente los canales
                 definidos por CLASSIFIER_CHANNEL_INDICES.
        """
        if not self._inicializado:
            raise RuntimeError("Llamar iniciar() antes de predecir().")

        roi_exp = roi[:, :, self._ch_idx]
        roi_exp = self._recortar_centro(roi_exp)

        if roi_exp is None or roi_exp.size == 0:
            return ResultadoClasificacion(
                etiqueta=0, probabilidad=0.0,
                es_buena=False, modelo_usado=self._tipo
            )

        if self._tipo == "pkl_rf_glcm":
            return self._predecir_pkl(roi_exp)
        elif self._tipo == "h5_cnn":
            return self._predecir_h5(roi_exp)
        elif self._tipo == "pt_torch":
            return self._predecir_pt(roi_exp)

    # ------------------------------------------------------------------
    # PREDICCIÓN POR TIPO
    # ------------------------------------------------------------------

    def _predecir_pkl(self, roi: np.ndarray) -> ResultadoClasificacion:
        """
        RF + GLCM. El ROI ya viene preprocesado con Vy — no se normaliza aquí.
        """
        if not SKIMAGE_DISPONIBLE:
            raise RuntimeError("skimage requerido para pkl_rf_glcm")

        features = self._extraer_features_glcm(roi)
        prob = self._modelo.predict_proba(features.reshape(1, -1))[0]
        prob_buena = float(prob[1])
        etiqueta = 1 if prob_buena >= config.CLASSIFIER_DECISION_THRESHOLD else 0

        return ResultadoClasificacion(
            etiqueta=etiqueta,
            probabilidad=prob_buena,
            es_buena=(etiqueta == 1),
            modelo_usado=f"pkl_rf_glcm_{roi.shape[2]}ch"
        )

    def _extraer_features_glcm(self, roi: np.ndarray) -> np.ndarray:
        """
        21 features por canal: 7 estadísticas + 8 histograma + 1 energía
        + 1 entropía + 4 GLCM = 21.
        Total: 21 × n_canales (105 para Exp. A, 63 para Exp. B).
        CONGELADO — no modificar sin reentrenar el modelo pkl.
        """
        features = []

        for i in range(roi.shape[2]):
            banda = roi[:, :, i].astype(np.float32)
            b_flat = banda.flatten()

            hist, _ = np.histogram(b_flat, bins=8, range=(0, 1), density=True)
            hist = hist / (hist.sum() + 1e-8)

            features.extend([
                float(np.mean(b_flat)),
                float(np.std(b_flat)),
                float(np.percentile(b_flat, 10)),
                float(np.percentile(b_flat, 25)),
                float(np.median(b_flat)),
                float(np.percentile(b_flat, 75)),
                float(np.percentile(b_flat, 90)),
            ])
            features.extend(hist.tolist())
            features.append(float(np.sum(b_flat ** 2) / len(b_flat)))
            features.append(float(-np.sum(hist * np.log2(hist + 1e-8))))

            banda_16 = (banda * 15).clip(0, 15).astype(np.uint8)
            glcm = graycomatrix(
                banda_16,
                distances=[1],
                angles=[0, np.pi / 4, np.pi / 2],
                levels=16,
                symmetric=True,
                normed=True
            )
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy']:
                features.append(float(graycoprops(glcm, prop).mean()))

        return np.array(features, dtype=np.float32)

    def _predecir_h5(self, roi: np.ndarray) -> ResultadoClasificacion:
        """
        CNN Keras / Dual-Branch.
        El ROI ya viene preprocesado con Vy (norm p1/p99.5 + gamma).
        Solo se redimensiona al input shape esperado por el modelo.
        NO se aplica normalización adicional.
        """
        target_h = config.CNN_INPUT_H
        target_w = config.CNN_INPUT_W
        n_ch = roi.shape[2]

        # Redimensionar cada canal al input shape del modelo
        resultado = np.zeros((target_h, target_w, n_ch), dtype=np.float32)
        for i in range(n_ch):
            banda = roi[:, :, i].astype(np.float32)
            resultado[:, :, i] = cv2.resize(
                banda, (target_w, target_h),
                interpolation=cv2.INTER_AREA
            )

        input_cnn = np.expand_dims(resultado, axis=0)
        prob_buena = float(self._modelo.predict(input_cnn, verbose=0)[0][0])
        etiqueta = 1 if prob_buena >= config.CLASSIFIER_DECISION_THRESHOLD else 0

        return ResultadoClasificacion(
            etiqueta=etiqueta,
            probabilidad=prob_buena,
            es_buena=(etiqueta == 1),
            modelo_usado=f"h5_cnn_{n_ch}ch"
        )

    def _predecir_pt(self, roi: np.ndarray) -> ResultadoClasificacion:
        raise NotImplementedError(
            "Implementar _predecir_pt() con el preprocesamiento del modelo."
        )

    # ------------------------------------------------------------------
    # UTILIDADES
    # ------------------------------------------------------------------

    def _recortar_centro(self, roi: np.ndarray) -> Optional[np.ndarray]:
        H, W = roi.shape[:2]
        f = ROI_CENTRO_FRACCION
        if H < 10 or W < 10:
            return roi
        mh = int(H * (1.0 - f) / 2)
        mw = int(W * (1.0 - f) / 2)
        y1, y2 = mh, H - mh
        x1, x2 = mw, W - mw
        if y2 <= y1 or x2 <= x1:
            return roi
        return roi[y1:y2, x1:x2, :]

    @property
    def tipo(self) -> str:
        return self._tipo

    @property
    def inicializado(self) -> bool:
        return self._inicializado
