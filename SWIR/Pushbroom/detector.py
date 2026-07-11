# =============================================================================
# detector.py — Wrapper YOLOv5 con validación de bboxes
# =============================================================================
# Responsabilidades:
#   1. Cargar el modelo YOLOv5 finetuneado una sola vez.
#   2. Ejecutar inferencia sobre el pseudo-RGB.
#   3. Filtrar detecciones por confianza, área, clase y posición.
#   4. Retornar detecciones en formato estandarizado con metadatos
#      necesarios para el tracker y la sincronización temporal.
#
# DETALLE DE FILTRADO DE BBOXES EN BORDES DEL BUFFER:
#   En rolling window continuo, los bordes izquierdo y derecho del frame
#   pueden contener mandarinas parciales (entrando o saliendo).
#   El filtro YOLO_MIN_BBOX_INSIDE_RATIO descarta bboxes que están
#   mayoritariamente fuera del frame útil o pegados a los bordes temporales.
# =============================================================================

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import time
import config


@dataclass
class Deteccion:
    """
    Resultado de una detección YOLO con metadatos de sincronización.

    Atributos:
        x1, y1, x2, y2: coordenadas del bbox en el pseudo-RGB (píxeles)
        cx, cy        : centroide del bbox
        confianza     : score YOLO [0, 1]
        clase         : índice de clase detectado
        area          : área del bbox en píxeles cuadrados
        linea_inicio_frame: línea global del inicio del frame en que fue
                            detectada. CRÍTICO para extraer ROI espectral.
    """
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int
    cy: int
    confianza: float
    clase: int
    area: int
    linea_inicio_frame: int = 0  # se rellena en detector.detectar()


class DetectorYOLO:
    """
    Wrapper sobre YOLOv5 para detección de mandarinas en pseudo-RGB SWIR.

    Uso:
        detector = DetectorYOLO()
        detector.iniciar()
        detecciones = detector.detectar(pseudo_rgb, linea_inicio_frame)
    """

    def __init__(self):
        self._modelo = None
        self._frame_h: int = 0
        self._frame_w: int = 0
        self._inicializado: bool = False

    def iniciar(self) -> None:
        """
        Carga el modelo YOLOv5 y ejecuta warm-up.
        Llamar una sola vez al inicio del sistema.
        """
        
        print(f"[YOLO] Cargando modelo: {config.YOLO_WEIGHTS_PATH}")
        t0 = time.time()

        # Fix para modelos entrenados en Linux/Mac y cargados en Windows.
        # PosixPath no existe en Windows — se reemplaza por WindowsPath
        # antes de deserializar el checkpoint.
        import pathlib
        pathlib.PosixPath = pathlib.WindowsPath
        self._modelo = torch.hub.load(
            config.YOLO_SOURCE_PATH,
            'custom',
            path=config.YOLO_WEIGHTS_PATH,
            source='local',
            verbose=False
        )
        self._modelo.conf = config.YOLO_CONF_THRESHOLD
        self._modelo.eval()

        # Mover a GPU si está disponible
        if torch.cuda.is_available():
            self._modelo.cuda()
            print("[YOLO] GPU detectada — modelo en CUDA.")
        else:
            print("[YOLO] GPU no disponible — usando CPU.")

        print(f"[YOLO] Modelo cargado en {time.time()-t0:.2f}s")

        # Warm-up: inferencia con frame dummy para pre-compilar operaciones
        self._warm_up()
        self._inicializado = True

    def _warm_up(self) -> None:
        """Ejecuta 2 inferencias dummy para calentar el modelo."""
        print("[YOLO] Ejecutando warm-up...")
        # Shape dummy: alto=Y_util, ancho=N_LINES
        y_util = config.BUFFER_Y_END - config.BUFFER_Y_START
        dummy = np.zeros((y_util, config.BUFFER_N_LINES, 3), dtype=np.uint8)
        for _ in range(2):
            self._modelo(dummy, size=config.YOLO_INFERENCE_SIZE)
        print("[YOLO] Warm-up completado.")

    def detectar(
        self,
        pseudo_rgb: np.ndarray,
        linea_inicio_frame: int
    ) -> List[Deteccion]:
        """
        Ejecuta detección sobre el pseudo-RGB y retorna detecciones válidas.

        Args:
            pseudo_rgb: array (Y_util, N_LINES, 3) uint8 del pseudo-RGB
            linea_inicio_frame: línea global del inicio del frame.
                                Se almacena en cada Deteccion para
                                sincronización temporal posterior.

        Returns:
            Lista de Deteccion filtradas y validadas.
        """
        if not self._inicializado:
            raise RuntimeError("[YOLO] Llamar iniciar() antes de detectar().")

        self._frame_h, self._frame_w = pseudo_rgb.shape[:2]

        # Inferencia YOLO
        results = self._modelo(pseudo_rgb, size=config.YOLO_INFERENCE_SIZE)
        raw_dets = results.xyxy[0]  # tensor (N, 6): x1,y1,x2,y2,conf,cls

        detecciones: List[Deteccion] = []

        for det in raw_dets:
            x1, y1, x2, y2, conf, cls = det.tolist()
            cls = int(cls)

            # ── Filtro 1: clase válida ────────────────────────────────
            if cls not in config.YOLO_MANDARINA_CLASSES:
                continue

            # ── Filtro 2: convertir a int ─────────────────────────────
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Clamping al frame
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(self._frame_w, x2)
            y2 = min(self._frame_h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # ── Filtro 3: área mínima ─────────────────────────────────
            area = (x2 - x1) * (y2 - y1)
            if area < config.YOLO_MIN_BBOX_AREA:
                if config.LOG_LEVEL >= 2:
                    print(f"[YOLO] Bbox descartado por área: {area} px²")
                continue

            # ── Filtro 4: bbox dentro del frame útil ──────────────────
            # Evita clasificar mandarinas parcialmente visibles en los
            # bordes temporales del rolling window.
            if not self._bbox_dentro_frame(x1, y1, x2, y2):
                if config.LOG_LEVEL >= 2:
                    print(f"[YOLO] Bbox descartado por borde: "
                          f"({x1},{y1},{x2},{y2})")
                continue

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            detecciones.append(Deteccion(
                x1=x1, y1=y1, x2=x2, y2=y2,
                cx=cx, cy=cy,
                confianza=float(conf),
                clase=cls,
                area=area,
                linea_inicio_frame=linea_inicio_frame
            ))

        return detecciones

    def _bbox_dentro_frame(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> bool:
        """
        Verifica que el bbox está suficientemente dentro del frame útil.

        Un bbox pegado al borde izquierdo o derecho del rolling window
        indica una mandarina parcialmente capturada. Se descarta si
        la fracción dentro del frame útil es menor que el umbral.

        El margen de borde temporal se aplica solo en X (eje temporal).
        En Y no se aplica porque la región útil ya fue recortada en camera.py.
        """
        ancho_bbox = x2 - x1
        if ancho_bbox <= 0:
            return False

        # Margen temporal: excluir bboxes que tocan los bordes del buffer
        margen_temporal = int(self._frame_w * 0.05)  # 5% de los bordes

        # Calcular fracción del bbox dentro del área central
        x1_util = margen_temporal
        x2_util = self._frame_w - margen_temporal

        overlap_x1 = max(x1, x1_util)
        overlap_x2 = min(x2, x2_util)

        if overlap_x2 <= overlap_x1:
            return False

        fraccion_dentro = (overlap_x2 - overlap_x1) / ancho_bbox

        return fraccion_dentro >= config.YOLO_MIN_BBOX_INSIDE_RATIO

    @property
    def inicializado(self) -> bool:
        return self._inicializado
