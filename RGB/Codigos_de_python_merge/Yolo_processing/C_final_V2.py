import cv2
import torch
import numpy as np
from tensorflow.keras.models import load_model

# ================= CONFIGURACIÓN =================
CONF_THRESHOLD = 0.75
MIN_AREA = 4000
MANDARINA_CLASSES = [0, 1]

INPUT_SIZE = (100, 100)
UMBRAL_MALA = 0.5

NUM_CARRILES = 4
ZONA_REGISTRO_ANCHO = 20   # ancho en píxeles alrededor de la línea

# ================= CARGAR MODELOS =================
yolo = torch.hub.load(
    'yolov5',
    'custom',
    path='best.pt',
    source='local'
)
yolo.conf = CONF_THRESHOLD

calidad_model = load_model("BuenasMalas.h5")

# ================= CÁMARA =================
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret, frame = cap.read()
if not ret:
    print("Error iniciando cámara")
    exit()

HEIGHT, WIDTH, _ = frame.shape

# ================= DEFINICIÓN DE CARRILES =================
# 🔥 AHORA LOS CARRILES SON HORIZONTALES
ALTO_CARRIL = HEIGHT // NUM_CARRILES

CARRILES = []
for i in range(NUM_CARRILES):
    y_min = i * ALTO_CARRIL
    y_max = (i + 1) * ALTO_CARRIL
    CARRILES.append((y_min, y_max))

# Línea virtual vertical (60% del ancho)
LINEA_X = int(WIDTH * 0.6)

# ================= ESTRUCTURAS =================
colas = {i: [] for i in range(NUM_CARRILES)}
procesadas_frame = set()

print("Sistema iniciado. Presiona 'q' para salir.")

# ================= LOOP PRINCIPAL =================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame)
    detections = results.xyxy[0]

    # Dibujar línea virtual (vertical)
    cv2.line(frame, (LINEA_X, 0), (LINEA_X, HEIGHT), (255, 0, 0), 2)

    # Dibujar divisiones horizontales de carril
    for i in range(1, NUM_CARRILES):
        y = i * ALTO_CARRIL
        cv2.line(frame, (0, y), (WIDTH, y), (200, 200, 200), 1)

    procesadas_frame.clear()

    for det in detections:

        x1, y1, x2, y2, conf, cls = det.tolist()
        cls = int(cls)

        if cls not in MANDARINA_CLASSES:
            continue

        area = (x2 - x1) * (y2 - y1)
        if area < MIN_AREA:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Dibujar bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)

        # 🔥 Determinar carril por coordenada Y
        carril_detectado = None
        for i, (ymin, ymax) in enumerate(CARRILES):
            if ymin <= cy < ymax:
                carril_detectado = i
                break

        if carril_detectado is None:
            continue

        # 🔥 REGISTRO AL CRUZAR LA LÍNEA VERTICAL
        if LINEA_X - ZONA_REGISTRO_ANCHO <= cx <= LINEA_X + ZONA_REGISTRO_ANCHO:

            # Evitar doble registro en mismo frame
            if carril_detectado in procesadas_frame:
                continue

            # ===== CLASIFICACIÓN =====
            roi = frame[y1:y2, x1:x2].copy()
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = cv2.resize(roi, INPUT_SIZE)
            roi = roi / 255.0
            roi = np.expand_dims(roi, axis=0)

            score = calidad_model.predict(roi, verbose=0)[0][0]

            if score >= UMBRAL_MALA:
                estado = 1
                etiqueta = "MALA"
            else:
                estado = 0
                etiqueta = "BUENA"

            colas[carril_detectado].append(estado)
            procesadas_frame.add(carril_detectado)

            print(f"[Carril {carril_detectado}] → {etiqueta} | Cola: {colas[carril_detectado]}")

            # Visual
            cv2.putText(
                frame,
                etiqueta,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,255),
                2
            )

    cv2.imshow("Sistema Multi-Carril Horizontal", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()