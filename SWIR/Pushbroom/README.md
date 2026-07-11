# Sistema de Clasificación SWIR — Mandarinas

## Estructura del proyecto

```
mandarina_swir/
├── config.py           ← TODOS los parámetros modificables (leer primero)
├── camera.py           ← Adquisición gxipy + extracción de bandas
├── buffer.py           ← Rolling window con sincronización temporal
├── preprocessing.py    ← Preprocesamiento Vy: norm por canal + gamma (compartido)
├── pseudo_rgb.py       ← Construcción pseudo-RGB para YOLO y display
├── detector.py         ← Wrapper YOLOv5
├── classifier.py       ← Clasificador intercambiable (pkl / h5 / pt)
├── tracker.py          ← Tracker multi-objeto (Y + área + IoU)
├── actuator.py         ← Cola FIFO + serial Arduino
├── main.py             ← Orquestador + hilos + visualización
├── requirements.txt
└── arduino/
    └── servo_control.ino
```

## Instalación

```bash
pip install -r requirements.txt
# Instalar gxipy manualmente desde el SDK de Daheng Imaging
```

## Uso

```bash
# Con cámara real
python main.py

# Modo simulado (sin hardware)
python main.py --sim
```

## Parámetro crítico por sesión

**Antes de cada sesión de laboratorio**, detectar el pico espectral del lote actual:

```bash
python detectar_pico.py --cubo cubos_prueba/<nombre>.npy
```

Luego actualizar en `config.py`:

```python
SPECTRAL_PEAK_BAND: int = <valor_detectado>  # único parámetro que cambia por sesión
```

Las bandas absolutas y el pseudo-RGB se recalculan automáticamente a partir de este valor.

## Calibración de parámetros mecánicos

Editar en `config.py`:

| Parámetro | Descripción |
|---|---|
| `RPM_BANDA` | Medir con tacómetro en el motor |
| `DIAMETRO_RODILLO_CM` | Medir con calibrador en el rodillo |
| `DISTANCIA_CAMARA_COMPUERTA_CM` | Medir con cinta métrica |

## Representación espectral Vy (NO modificar sin reentrenar)

Los offsets y parámetros de preprocesamiento están congelados en `config.py`:

| Parámetro | Valor | Descripción |
|---|---|---|
| `OFFSETS_STACK` | `[+11, +2, -5, -10, -3]` | Offsets relativos al pico para los 5 canales |
| `PSEUDO_RGB_BAND_INDICES` | `[0, 1, 2]` | Canales R, G, B dentro del stack |
| `PREP_P_LOW` | `1.0` | Percentil inferior de normalización por canal |
| `PREP_P_HIGH` | `99.5` | Percentil superior de normalización por canal |
| `PREP_GAMMA` | `1.440` | Corrección gamma aplicada después de normalizar |
| `BACKGROUND_CORRECTION_ON` | `False` | OFF validado experimentalmente con Vy |

## Visualización

La ventana de tamaño fijo se configura en `config.py`:

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `WINDOW_WIDTH` | `1600` | Ancho total de la ventana en píxeles |
| `WINDOW_HEIGHT` | `900` | Alto total de la ventana en píxeles |
| `RIGHT_PANEL_WIDTH` | `320` | Ancho fijo del panel de estado |

El panel izquierdo muestra el pseudo-RGB en tiempo real con bboxes y etiquetas.
El panel derecho muestra última clasificación, contadores de sesión, estado de módulos
y configuración activa. El diseño no se deforma si cambia el tamaño del buffer.

## Cambiar clasificador

En `config.py`:

```python
# Para RF+GLCM (.pkl sklearn Pipeline)
CLASSIFIER_TYPE = "pkl_rf_glcm"
CLASSIFIER_MODEL_PATH = "rf_glcm_5ch.pkl"

# Para CNN Keras (.h5)
CLASSIFIER_TYPE = "h5_cnn"
CLASSIFIER_MODEL_PATH = "cnn_5ch.h5"
```

El preprocesamiento Vy (normalización por canal + gamma) se aplica en
`preprocessing.py` antes de entregar los datos al modelo — no dentro del
clasificador. Cambiar el modelo no requiere modificar el preprocesamiento.

## Arquitectura de hilos

```
Hilo Cámara      → cola_camara     → Loop Principal
Loop Principal   → buffer          → get_frame() → pseudo_rgb.py → YOLO → Tracker
Tracker          → cola_clasif.    → Hilo Clasificador
Hilo Clasif.     → preprocessing.py → clasificador → cola_resultados
Loop Principal   → cola_resultados → Actuador → Serial Arduino
Loop Principal   → canvas fijo     → cv2.imshow
```

## Flujo de preprocesamiento Vy

```
Frame crudo del buffer
      ↓
preprocessing.preprocess_channels()
      ├── norm por canal (percentil 1 / 99.5)
      ├── clip [0, 1]
      └── gamma 1.440
      ↓
┌─────────────────────────────────┐
│ Ruta YOLO/display               │  → uint8 (*255) → YOLO + visualización
│ canales [0, 1, 2] (R, G, B)    │
└─────────────────────────────────┘
┌─────────────────────────────────┐
│ Ruta CNN                        │  → float32 en [0,1] → clasificador
│ canales [0, 1, 2, 3, 4]        │
└─────────────────────────────────┘
```

La misma función matemática se aplica en ambas rutas.
YOLO y la CNN siempre reciben datos preprocesados idénticamente.

## Supuesto operativo crítico

La velocidad de la banda se asume **constante**. Variaciones de velocidad
deforman el pseudo-RGB y afectan el timing del servo. Si la banda
experimenta variaciones de velocidad, recalibrar `RPM_BANDA` o implementar
un encoder de velocidad en futuras versiones.
