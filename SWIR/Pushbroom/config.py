# =============================================================================
# config.py — Parámetros globales del sistema de clasificación SWIR
# =============================================================================
# INSTRUCCIÓN: Este archivo es el único lugar donde se modifican parámetros
# operativos. No hay "números mágicos" dispersos en el resto del código.
# Cada sección está comentada con la unidad, el rango típico y el efecto
# de cambiar el valor.
# =============================================================================

# -----------------------------------------------------------------------------
# SECCIÓN 1 — CÁMARA SWIR (Daheng MER3-138-136G3M-P-SWIR)
# -----------------------------------------------------------------------------

CAMERA_INDEX: int = 1
CAMERA_EXPOSURE_US: float = 4000.0
CAMERA_N_BANDS: int = 1280
CAMERA_LAMBDA_MIN_NM: float = 400.0
CAMERA_LAMBDA_MAX_NM: float = 1100.0
CAMERA_LINE_HEIGHT: int = 1024

# -----------------------------------------------------------------------------
# SECCIÓN 2 — BANDAS ESPECTRALES DE INTERÉS
# -----------------------------------------------------------------------------
# ⚙️  ÚNICO PARÁMETRO A CAMBIAR POR SESIÓN DE CAPTURA
#
# Índice del pico espectral principal de la mandarina para la sesión actual.
# Obtenerlo corriendo detectar_pico.py sobre un cubo capturado al inicio.
#
# Valor actual: última sesión de laboratorio (16-jun-2026, iluminación controlada)
# Historial reciente:
#   16-jun-2026 (halógena)  → PICO ≈ 576
#   12-jun-2026             → PICO = 682
#   10-jun-2026             → PICO = 549
#
SPECTRAL_PEAK_BAND: int = 579   # ← CAMBIAR por sesión (correr detectar_pico.py)

# Offsets congelados — NO modificar sin reentrenar los clasificadores.
# Representan la estructura espectral relativa al pico:
#   offset -150 → soporte izquierdo lejano   (canal B del pseudo-RGB)
#   offset -100 → soporte izquierdo cercano
#   offset  -80 → soporte izquierdo próximo  (canal G del pseudo-RGB)
#   offset    0 → pico principal             (canal R del pseudo-RGB)
#   offset  +50 → soporte derecho
OFFSETS_STACK: list = [+11, +2, -5, -10, -3]   # ← NO cambiar sin reentrenar
# Canal 0 (+11): R del pseudo-RGB | Canal 1 (+2): G | Canal 2 (-5): B
# Canal 3 (-10): soporte 1        | Canal 4 (-3): soporte 2
SPECTRAL_OFFSETS: list = OFFSETS_STACK  # alias para compatibilidad
SPECTRAL_BANDS: list = [SPECTRAL_PEAK_BAND + o for o in OFFSETS_STACK]

# Número de bandas — derivado automáticamente.
N_BANDS: int = len(SPECTRAL_BANDS)  # = 5

# -----------------------------------------------------------------------------
# SECCIÓN 3 — PSEUDO-RGB PARA YOLO
# -----------------------------------------------------------------------------
# Índices DENTRO de SPECTRAL_BANDS asignados a los canales R, G, B.
# Con los offsets actuales:
#   R = SPECTRAL_BANDS[3] = pico          (offset  0)
#   G = SPECTRAL_BANDS[2] = pico-80       (offset -80)
#   B = SPECTRAL_BANDS[0] = pico-150      (offset -150)
#
# IMPORTANTE: estos mismos 3 canales son los usados en el Experimento B
# del clasificador (CNN_INPUT_CHANNELS = 3).
# Cambiar esto requiere reentrenar YOLO y los clasificadores del Exp. B.
PSEUDO_RGB_BAND_INDICES: list = [0, 1, 2]  # R=canal0(+11), G=canal1(+2), B=canal2(-5)

# -----------------------------------------------------------------------------
# SECCIÓN 3b — PREPROCESAMIENTO Vy (normalización + gamma)
# -----------------------------------------------------------------------------
# Parámetros de la transformación Vy compartida entre YOLO y CNN.
# Modificar aquí afecta ambas rutas — coordinrar con reentrenamiento.
PREP_P_LOW:  float = 1.0    # percentil inferior para norm por canal
PREP_P_HIGH: float = 99.5   # percentil superior para norm por canal
PREP_GAMMA:  float = 1.440  # gamma directa aplicada DESPUÉS de normalizar

# Background correction switch.
# OFF por validación experimental con Vy: bajo normalización por canal
# la corrección de fondo es matemáticamente invariante.
# Mantener disponible para futuras comparaciones multi-lote/multi-día.
BACKGROUND_CORRECTION_ON: bool = False

# -----------------------------------------------------------------------------
# SECCIÓN 4 — BUFFER ROLLING WINDOW
# -----------------------------------------------------------------------------
BUFFER_N_LINES: int = 500
BUFFER_Y_START: int = 100
BUFFER_Y_END:   int = 924
# Alto útil = BUFFER_Y_END - BUFFER_Y_START = 824 px

# -----------------------------------------------------------------------------
# SECCIÓN 5 — PARÁMETROS MECÁNICOS DE LA BANDA
# -----------------------------------------------------------------------------

# Retardo experimental desde cruce del centroide hasta activación del mecanismo.
# Medido desde que el centroide del objeto cruza la línea central del buffer
# hasta que debe iniciarse el movimiento del mecanismo.
# Recalibrar si cambia velocidad de banda, relación de transmisión o posición
# de compuerta. Validar con mínimo 5 mediciones antes de congelar el valor.
CENTER_CROSSING_DELAY_S: float = 40.0                                                       #tiempo

# ------------------------------------------------------------------
# Referencia mecánica (NO utilizada para el cálculo del disparo)
# ------------------------------------------------------------------
RPM_BANDA: float = 30.0                      # referencia — medir con tacómetro
DIAMETRO_RODILLO_CM: float = 6.0             # rodillo motriz
DISTANCIA_CAMARA_COMPUERTA_CM: float = 40.0  # referencia — medir con cinta

# -----------------------------------------------------------------------------
# SECCIÓN 6 — DETECCIÓN YOLO
# -----------------------------------------------------------------------------
YOLO_WEIGHTS_PATH: str = "best.pt"
YOLO_SOURCE_PATH: str = "yolov5"
YOLO_INFERENCE_SIZE: int = 320
YOLO_CONF_THRESHOLD: float = 0.60
YOLO_MANDARINA_CLASSES: list = [0, 1]
YOLO_MIN_BBOX_AREA: int = 3000
#YOLO_MIN_BBOX_AREA: int = 1000
YOLO_MIN_BBOX_INSIDE_RATIO: float = 0.70
#YOLO_MIN_BBOX_INSIDE_RATIO: float = 0.30 # prueba par adeterminar por que la mandarina sale recortada en el pseudo rgb
YOLO_ZONA_REGISTRO_ANCHO: int = 25

# -----------------------------------------------------------------------------
# SECCIÓN 7 — TRACKER MULTI-OBJETO
# -----------------------------------------------------------------------------
TRACKER_MAX_DIST_Y: int = 60
TRACKER_MAX_AREA_RATIO_DIFF: float = 0.50
TRACKER_MAX_FRAMES_PERDIDO: int = 8
TRACKER_MIN_IOU_ASOCIACION: float = 0.30

# -----------------------------------------------------------------------------
# SECCIÓN 8 — CLASIFICADOR INTERCAMBIABLE
# -----------------------------------------------------------------------------
# ⚙️  DOS PARÁMETROS A CAMBIAR PARA ALTERNAR ENTRE EXPERIMENTOS:
#     CNN_INPUT_CHANNELS y CLASSIFIER_TYPE + CLASSIFIER_MODEL_PATH
#
# EXPERIMENTO A — 5 bandas espectrales Vy:
#   CNN_INPUT_CHANNELS = 5
#   CLASSIFIER_CHANNEL_INDICES = [0, 1, 2, 3, 4]
#   Canales: pico+11, pico+2, pico-5, pico-10, pico-3
#
# EXPERIMENTO B — 3 canales pseudo-RGB Vy (mismo que YOLO):
#   CNN_INPUT_CHANNELS = 3
#   CLASSIFIER_CHANNEL_INDICES = [0, 1, 2]  → offsets [+11, +2, -5]

# ── Configuración activa ──────────────────────────────────────────────────────
CLASSIFIER_TYPE: str = "h5_cnn"           # ← "pkl_rf_glcm" | "h5_cnn" | "pt_torch"
CLASSIFIER_MODEL_PATH: str = "experimento5.h5"  # ← ruta al modelo activo
#CLASSIFIER_DECISION_THRESHOLD: float = 0.45
CLASSIFIER_DECISION_THRESHOLD: float = 0.0001  #pruebas (todas son buenas)

#
# Número de canales de entrada del clasificador.
# 5 → Experimento A (todas las bandas espectrales)
# 3 → Experimento B (pseudo-RGB: pico, pico-80, pico-150)
CNN_INPUT_CHANNELS: int = 5                    # ← 5 (Exp. A) | 3 (Exp. B)

# Índices dentro de SPECTRAL_BANDS que se pasan al clasificador.
# Exp. A: [0, 1, 2, 3, 4]
# Exp. B: [0, 2, 3]  → PSEUDO_RGB_BAND_INDICES en orden B, G, R
CLASSIFIER_CHANNEL_INDICES: list = [0, 1, 2, 3, 4]  # ← Exp. A por defecto

# Shape de entrada para CNN Keras
CNN_INPUT_H: int = 100
CNN_INPUT_W: int = 100

# -----------------------------------------------------------------------------
# SECCIÓN 9 — COMUNICACIÓN ARDUINO
# -----------------------------------------------------------------------------
ARDUINO_PORT: str = "COM7"
ARDUINO_BAUDRATE: int = 115200
ARDUINO_INIT_WAIT_S: float = 2.0
ARDUINO_TIMEOUT_S: float = 1.0
ARDUINO_MAX_DELAY_MS: int = 120000
# Hilo ACK deshabilitado temporalmente para reducir acceso concurrente al
# puerto serie. Los mensajes READY/FIRE/CLOSE son informativos únicamente.
# Reactivar cuando se confirme que los disparos simultáneos funcionan.
ARDUINO_ACK_ENABLED: bool = False

# -----------------------------------------------------------------------------
# SECCIÓN 10 — GEOMETRÍA DE CARRILES
# -----------------------------------------------------------------------------
NUM_CARRILES: int = 2

# -----------------------------------------------------------------------------
# SECCIÓN 11 — VISUALIZACIÓN Y DEBUG
# -----------------------------------------------------------------------------
DISPLAY_ENABLED: bool = True
DISPLAY_SCALE: float = 0.7         # solo para compatibilidad legacy
DISPLAY_SHOW_FPS: bool = True

# Ventana de tamaño fijo — modificar aquí para adaptar a monitor/proyector
WINDOW_WIDTH:       int = 1600     # ancho total de la ventana en píxeles
WINDOW_HEIGHT:      int = 900      # alto total de la ventana en píxeles
RIGHT_PANEL_WIDTH:  int = 320      # ancho fijo del panel derecho

LOG_LEVEL: int = 1
LOG_TO_CSV: bool = False
LOG_CSV_PATH: str = "clasificaciones.csv"

# -----------------------------------------------------------------------------
# SECCIÓN 12 — SIMULADOR (CamaraSimulada con Holdout real)
# -----------------------------------------------------------------------------
# Ruta al dataset y archivo de splits del holdout.
SIM_DATASET_PATH:   str   = r"C:\Users\ASUS\Documents\UIS\Trabajo_de_grado\Datasets\Datasets_refinadosv4\CNN_Vy"
SIM_SPLITS_JSON:    str   = r"C:\Users\ASUS\Documents\UIS\Trabajo_de_grado\Datasets\Datasets_refinadosv4\CNN_Vy\splits\splits_actual.json"

# Velocidad de emisión de líneas — debe coincidir con la cámara real.
SIM_LINES_PER_SECOND: int = 136

# Separación variable entre mandarinas (líneas de fondo entre frutas).
SIM_LINES_MIN: int = 150
SIM_LINES_MAX: int = 250

# Percentil para identificar columnas de fondo en cada stack.
# Columnas con energía < percentil N se consideran fondo.
SIM_FONDO_PERCENTIL: float = 10.0

# Margen mínimo (px) respecto a los bordes superior e inferior del buffer
# donde puede colocarse una mandarina simulada.
# Cada muestra recibe una posición vertical aleatoria, constante durante
# todo su recorrido. Si H + 2*SIM_Y_MARGIN > Y_util, el margen se reduce
# automáticamente para evitar errores.
SIM_Y_MARGIN: int = 30

# Reproducibilidad: False → orden exacto del splits_actual.json.
#                   True  → shuffle con SIM_RANDOM_SEED.
SIM_RANDOM_ORDER: bool  = True
SIM_RANDOM_SEED:  int   = 42

# Modo debug: None → recorrer todo el holdout.
# str  → repetir indefinidamente una sola muestra (nombre del archivo .npy).
# Útil para depurar YOLO, tracker, CNN o Arduino con una fruta conocida.
# Ejemplo: SIM_REPEAT_SAMPLE = "good_0017_stack.npy"
SIM_REPEAT_SAMPLE: object = None

# Imprimir en consola información de cada muestra emitida.
SIM_OVERLAY_INFO: bool = True

# ── Modo dual — dos mandarinas simultáneas (una por carril) ──
# Permite probar que el scheduler y Arduino manejan dos disparos
# con t_disparo prácticamente igual.
SIM_DUAL_MODE:  bool = False    # habilitar/deshabilitar modo dual
SIM_DUAL_EVERY: int  = 5       # cada N mandarinas, generar un par simultáneo
SIM_DUAL_OFFSET_LINES: int = 0 # desfase temporal entre ambas frutas (en líneas)
                                # 0 = centroides cruzan al mismo tiempo
                                # 5 = una cruza 5 líneas después (~37ms a 136 FPS)

# -----------------------------------------------------------------------------
# VALIDACIÓN EN CARGA
# -----------------------------------------------------------------------------
assert len(PSEUDO_RGB_BAND_INDICES) == 3, \
    "PSEUDO_RGB_BAND_INDICES debe tener exactamente 3 elementos [R, G, B]"

assert all(0 <= i < N_BANDS for i in PSEUDO_RGB_BAND_INDICES), \
    f"PSEUDO_RGB_BAND_INDICES fuera de rango (0..{N_BANDS-1})"

assert all(0 <= b < CAMERA_N_BANDS for b in SPECTRAL_BANDS), \
    f"Banda fuera de rango del sensor. Verifica SPECTRAL_PEAK_BAND={SPECTRAL_PEAK_BAND}"

assert CNN_INPUT_CHANNELS in (3, 5), \
    "CNN_INPUT_CHANNELS debe ser 3 (Exp. B) o 5 (Exp. A)"

assert len(CLASSIFIER_CHANNEL_INDICES) == CNN_INPUT_CHANNELS, \
    f"CLASSIFIER_CHANNEL_INDICES tiene {len(CLASSIFIER_CHANNEL_INDICES)} elementos " \
    f"pero CNN_INPUT_CHANNELS={CNN_INPUT_CHANNELS}"

assert BUFFER_Y_END > BUFFER_Y_START, \
    "BUFFER_Y_END debe ser mayor que BUFFER_Y_START"

assert BUFFER_Y_END <= CAMERA_LINE_HEIGHT, \
    f"BUFFER_Y_END no puede superar CAMERA_LINE_HEIGHT ({CAMERA_LINE_HEIGHT})"

assert CLASSIFIER_TYPE in ("pkl_rf_glcm", "h5_cnn", "pt_torch"), \
    f"CLASSIFIER_TYPE desconocido: {CLASSIFIER_TYPE}"

assert CENTER_CROSSING_DELAY_S > 0, \
    "CENTER_CROSSING_DELAY_S debe ser positivo"
