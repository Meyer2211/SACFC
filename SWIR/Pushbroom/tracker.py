# =============================================================================
# tracker.py — Tracker multi-objeto con Y + área + IoU temporal
# =============================================================================
# Responsabilidades:
#   1. Asignar IDs únicos a cada mandarina detectada.
#   2. Mantener continuidad de IDs entre frames del rolling window.
#   3. Determinar el carril (1 o 2) según posición Y del centroide.
#   4. Controlar el estado de clasificación por objeto (no por carril).
#   5. Eliminar objetos no vistos por más de MAX_FRAMES_PERDIDO frames.
#
# CRITERIOS DE ASOCIACIÓN (en orden de prioridad):
#   1. IoU temporal entre bboxes consecutivos (más robusto)
#   2. Distancia en Y del centroide (posición espacial real)
#   3. Similaridad de área del bbox (mismo objeto ≈ mismo tamaño)
#
# CARRIL:
#   mitad_y = (BUFFER_Y_END - BUFFER_Y_START) / 2
#   cy < mitad_y  → carril 2 (mitad superior de la imagen)
#   cy >= mitad_y → carril 1 (mitad inferior de la imagen)
#
# POLÍTICA DE CLASIFICACIÓN ÚNICA:
#   Cada objeto se clasifica exactamente una vez, cuando su centroide X
#   entra en la zona de registro (alrededor de la línea virtual central
#   del buffer). Una vez marcado como 'clasificado', no se re-encola.
# =============================================================================

import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import config


@dataclass
class ObjetoTrackeado:
    """Estado completo de un objeto trackeado."""
    id_obj: int
    cx: int                   # centroide X (eje temporal del buffer)
    cy: int                   # centroide Y (eje espacial)
    x1: int                   # bbox actual
    y1: int
    x2: int
    y2: int
    area: int
    carril: int               # 1 (inferior) o 2 (superior)
    clasificado: bool = False # True si ya fue enviado a clasificación
    frames_perdido: int = 0   # frames consecutivos sin detección
    linea_inicio_frame: int = 0  # para sincronización temporal del ROI


class Tracker:
    """
    Tracker multi-objeto thread-safe para el sistema pushbroom.

    Diferencia con tracker del código RGB original:
    - Asocia por IoU + distancia_Y + área (no solo distancia_Y)
    - Guarda bbox completo por objeto (necesario para ROI espectral)
    - Guarda linea_inicio_frame por detección (sincronización temporal)
    - Determina carril por centroide_Y relativo a mitad del alto útil
    """

    def __init__(self):
        self._siguiente_id: int = 0
        self._objetos: Dict[int, ObjetoTrackeado] = {}
        self._lock = threading.RLock()

        # Mitad del alto útil para determinar carril
        self._mitad_y = (config.BUFFER_Y_END - config.BUFFER_Y_START) / 2.0

        # Línea virtual de clasificación: mitad horizontal del buffer
        self._linea_virtual_x = config.BUFFER_N_LINES // 2

    # ------------------------------------------------------------------
    # ACTUALIZACIÓN
    # ------------------------------------------------------------------

    def actualizar(
        self,
        detecciones: List,  # List[Deteccion] de detector.py
    ) -> Dict[int, ObjetoTrackeado]:
        """
        Actualiza el tracker con las detecciones del frame actual.

        Args:
            detecciones: lista de Deteccion del DetectorYOLO

        Returns:
            Diccionario {id_obj: ObjetoTrackeado} con todos los objetos
            activos (incluyendo los no vistos este frame con grace period).
        """
        with self._lock:
            ids_vistos = set()

            for det in detecciones:
                mejor_id = self._encontrar_mejor_match(det)

                if mejor_id is not None:
                    # Objeto conocido: actualizar posición
                    obj = self._objetos[mejor_id]
                    obj.cx = det.cx
                    obj.cy = det.cy
                    obj.x1 = det.x1
                    obj.y1 = det.y1
                    obj.x2 = det.x2
                    obj.y2 = det.y2
                    obj.area = det.area
                    obj.frames_perdido = 0
                    obj.linea_inicio_frame = det.linea_inicio_frame
                    # El carril NO se actualiza una vez asignado,
                    # para evitar cambios erróneos por deformación temporal
                    ids_vistos.add(mejor_id)
                else:
                    # Objeto nuevo: asignar ID
                    nuevo_id = self._siguiente_id
                    self._siguiente_id += 1

                    carril = self._determinar_carril(det.cy)

                    self._objetos[nuevo_id] = ObjetoTrackeado(
                        id_obj=nuevo_id,
                        cx=det.cx, cy=det.cy,
                        x1=det.x1, y1=det.y1,
                        x2=det.x2, y2=det.y2,
                        area=det.area,
                        carril=carril,
                        clasificado=False,
                        frames_perdido=0,
                        linea_inicio_frame=det.linea_inicio_frame
                    )
                    ids_vistos.add(nuevo_id)

                    if config.LOG_LEVEL >= 1:
                        print(f"[TRACKER] Nuevo objeto ID={nuevo_id} "
                              f"| carril={carril} | cy={det.cy:.0f} "
                              f"| área={det.area}")

            # Incrementar frames_perdido para objetos no vistos
            for id_ in list(self._objetos.keys()):
                if id_ not in ids_vistos:
                    self._objetos[id_].frames_perdido += 1
                    if (self._objetos[id_].frames_perdido >
                            config.TRACKER_MAX_FRAMES_PERDIDO):
                        if config.LOG_LEVEL >= 2:
                            print(f"[TRACKER] Eliminando ID={id_} "
                                  f"(perdido por "
                                  f"{self._objetos[id_].frames_perdido} frames)")
                        del self._objetos[id_]

            return dict(self._objetos)

    # ------------------------------------------------------------------
    # LÓGICA DE ASOCIACIÓN
    # ------------------------------------------------------------------

    def _encontrar_mejor_match(self, det) -> Optional[int]:
        """
        Encuentra el objeto activo más probable para una detección nueva.

        Criterios combinados:
          1. Carril compatible (misma mitad Y)
          2. Score = w_iou * IoU + w_dist * exp(-dist_y/sigma) + w_area * sim_area
          3. Score debe superar umbral mínimo

        Returns:
            ID del mejor candidato, o None si no hay match suficiente.
        """
        carril_det = self._determinar_carril(det.cy)

        mejor_id = None
        mejor_score = -1.0

        for id_, obj in self._objetos.items():
            # ── Filtro 1: mismo carril ────────────────────────────────
            if obj.carril != carril_det:
                continue

            # ── Filtro 2: objeto ya clasificado y alejado ─────────────
            # Si el objeto ya fue clasificado y su centroide X está muy
            # lejos del actual, probablemente salió del buffer.
            if obj.clasificado and abs(obj.cx - det.cx) > 150:
                continue

            # ── Score combinado ───────────────────────────────────────
            score = self._calcular_score(det, obj)

            if score > mejor_score:
                mejor_score = score
                mejor_id = id_

        # Umbral mínimo de score para aceptar la asociación
        SCORE_MIN = 0.25
        if mejor_score < SCORE_MIN:
            return None

        return mejor_id

    def _calcular_score(self, det, obj: ObjetoTrackeado) -> float:
        """
        Calcula score de asociación entre una detección y un objeto activo.

        Score ∈ [0, 1]. Mayor = mejor match.

        Componentes:
          - IoU temporal: overlap de bboxes (más robusto con deformación)
          - Distancia Y: proximidad espacial del centroide
          - Similitud de área: mismo objeto ≈ mismo tamaño
        """
        # ── Componente 1: IoU de bboxes ───────────────────────────────
        iou = self._calcular_iou(
            det.x1, det.y1, det.x2, det.y2,
            obj.x1, obj.y1, obj.x2, obj.y2
        )

        # Si IoU supera el umbral de asociación, este match es fuerte
        if iou >= config.TRACKER_MIN_IOU_ASOCIACION:
            return 0.5 + 0.5 * iou  # score alto garantizado

        # ── Componente 2: distancia Y ─────────────────────────────────
        dist_y = abs(det.cy - obj.cy)
        sigma_y = config.TRACKER_MAX_DIST_Y
        score_dist = np.exp(-dist_y / sigma_y) if dist_y < sigma_y * 3 else 0.0

        # ── Componente 3: similitud de área ───────────────────────────
        if obj.area > 0:
            ratio = det.area / obj.area
            # ratio cercano a 1.0 → mismo objeto
            diff = abs(ratio - 1.0)
            score_area = max(0.0, 1.0 - diff / config.TRACKER_MAX_AREA_RATIO_DIFF)
        else:
            score_area = 0.0

        # ── Score combinado ───────────────────────────────────────────
        # Pesos: distancia_Y es el criterio principal en pushbroom
        # porque el eje Y es espacial puro.
        score = 0.15 * iou + 0.55 * score_dist + 0.30 * score_area

        return float(score)

    @staticmethod
    def _calcular_iou(
        x1a: int, y1a: int, x2a: int, y2a: int,
        x1b: int, y1b: int, x2b: int, y2b: int
    ) -> float:
        """Calcula IoU entre dos bboxes."""
        ix1 = max(x1a, x1b); iy1 = max(y1a, y1b)
        ix2 = min(x2a, x2b); iy2 = min(y2a, y2b)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        area_inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (x2a - x1a) * (y2a - y1a)
        area_b = (x2b - x1b) * (y2b - y1b)
        area_union = area_a + area_b - area_inter

        return area_inter / area_union if area_union > 0 else 0.0

    # ------------------------------------------------------------------
    # LÓGICA DE CARRIL Y ZONA DE CLASIFICACIÓN
    # ------------------------------------------------------------------

    def _determinar_carril(self, cy: int) -> int:
        """
        Determina el carril físico según la posición Y del centroide.

        cy < mitad_y  → carril 2 (mitad superior, compuerta 2)
        cy >= mitad_y → carril 1 (mitad inferior, compuerta 1)
        """
        return 2 if cy < self._mitad_y else 1

    def objeto_en_zona_clasificacion(self, obj: ObjetoTrackeado) -> bool:
        """
        Determina si un objeto debe clasificarse ahora.

        Condición: centroide X dentro de la zona de registro alrededor
        de la línea virtual central del buffer.

        La línea virtual está en BUFFER_N_LINES // 2. Se clasifican solo
        los objetos cuyo centroide la cruce, no los que están en bordes.
        """
        zona_inicio = (self._linea_virtual_x -
                       config.YOLO_ZONA_REGISTRO_ANCHO)
        zona_fin    = (self._linea_virtual_x +
                       config.YOLO_ZONA_REGISTRO_ANCHO)

        return zona_inicio <= obj.cx <= zona_fin

    # ------------------------------------------------------------------
    # CONTROL DE ESTADO
    # ------------------------------------------------------------------

    def marcar_clasificado(self, id_obj: int) -> None:
        """Marca un objeto como clasificado. Thread-safe."""
        with self._lock:
            if id_obj in self._objetos:
                self._objetos[id_obj].clasificado = True

    def obtener_objeto(self, id_obj: int) -> Optional[ObjetoTrackeado]:
        """Retorna el estado de un objeto por ID. Thread-safe."""
        with self._lock:
            return self._objetos.get(id_obj, None)

    def objetos_activos(self) -> Dict[int, ObjetoTrackeado]:
        """Retorna copia del dict de objetos activos. Thread-safe."""
        with self._lock:
            return dict(self._objetos)

    @property
    def linea_virtual_x(self) -> int:
        return self._linea_virtual_x
