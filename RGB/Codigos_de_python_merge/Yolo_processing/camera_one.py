
import cv2
import torch
#import time 
import numpy as np
from collections import Counter

# ================= CONFIGURACIÓN =================
CONF_THRESHOLD = 0.75        # confianza mínima YOLO
STABLE_FRAMES = 10           # frames para decisión estable
MIN_AREA = 4000              # área mínima del bounding box
MANDARINA_CLASSES = [0, 1]   # clases válidas

# ================= VARIABLES =================
frame_count = 0
class_votes = []
processed = False
last_bbox = None
contador = 0
# ================= CARGAR MODELO YOLOv5 =================
model = torch.hub.load(
    'yolov5',        # carpeta local yolov5
    'custom',
    path='best.pt',
    source='local'
)
model.conf = CONF_THRESHOLD
print("Clases del modelo:")
print(model.names)


# ================= CÁMARA =================
cap = cv2.VideoCapture(0)

# Opcional: bajar resolución para estabilidad
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("📷 Cámara iniciada. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0]

    valid_detections = []

    # -------- FILTRADO DE DETECCIONES --------
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        cls = int(cls)

        if cls not in MANDARINA_CLASSES:
            continue

        area = (x2 - x1) * (y2 - y1)
        if area < MIN_AREA:
            continue

        valid_detections.append(det)

    if len(valid_detections) > 0:
        # Tomar la detección más confiable
        det = max(valid_detections, key=lambda x: x[4])

        x1, y1, x2, y2, conf, cls = det.tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls = int(cls)

        frame_count += 1
        class_votes.append(cls)
        last_bbox = (x1, y1, x2, y2)

        # Dibujar bounding box
        color = (0, 255, 0) if cls == 0 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"Clase {cls} | {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        # -------- DECISIÓN ESTABLE --------
        if frame_count >= STABLE_FRAMES and not processed:
            dominant_class = Counter(class_votes).most_common(1)[0][0]

            print("✅ Mandarina estable detectada")
            print(f"👉 Clase dominante: {dominant_class}")

            # Recorte final
            roi = frame[y1:y2, x1:x2].copy()
            #cv2.imwrite("mandarina_recortada.jpg", roi)
            #print("📸 Recorte guardado")


            #solucion recorte eliminado

            contador += 1
            filename = f"mandarina_{contador}.jpg"
            cv2.imwrite(filename, roi)


            # Aquí luego va la comunicación con Arduino
            # Ejemplo:
            # if dominant_class == 0: mandar señal BUENA
            # else: mandar señal MALA

            processed = True

    else:
        # No hay detecciones → reset
        frame_count = 0
        class_votes = []
        processed = False
        last_bbox = None

    cv2.imshow("YOLO Mandarina - Estable", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()