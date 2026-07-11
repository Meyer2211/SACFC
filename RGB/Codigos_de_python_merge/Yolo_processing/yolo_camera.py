
import cv2
import torch
import numpy as np
from collections import Counter


# ---------------- CONFIGURACIÓN ----------------
STABLE_FRAMES = 10        # Frames necesarios para estabilidad
CONF_THRESHOLD = 0.75      # Confianza mínima YOLO

MANDARINA_CLASS = 2       #  

# ---------------- VARIABLES DE CONTROL ----------------
frame_count = 0
class_votes = []
processed = False

last_bbox = None

# ---------------- CARGAR MODELO YOLOv5 ----------------
model = torch.hub.load(
    'yolov5',        # ← carpeta local
    'custom',
    path='best.pt',
    source='local'
)
model.conf = CONF_THRESHOLD

# ---------------- CÁMARA ----------------
cap = cv2.VideoCapture(0)

print("📷 Cámara iniciada. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0]

    if len(detections) > 0:

        mandarinas = []

        for det in detections:
            x1, y1, x2, y2, conf, cls = det.tolist()
            cls = int(cls)

            # 🔹 Filtro por confianza
            if conf < CONF_THRESHOLD:
                continue

            # 🔹 Filtro por clase (AQUÍ VA)
            if cls != MANDARINA_CLASS:
                continue

            mandarinas.append(det)

        if len(mandarinas) == 0:
            frame_count = 0
            class_votes = []
            processed = False
            last_bbox = None
            continue

        # 👉 de las mandarinas, tomamos la más confiable
        mandarinas = torch.stack(mandarinas)
        det = mandarinas[mandarinas[:, 4].argmax()]


        x1, y1, x2, y2, conf, cls = det.tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls = int(cls)

        frame_count += 1
        class_votes.append(cls)
        last_bbox = (x1, y1, x2, y2)

        # Dibujar bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Clase {cls} | {conf:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        # --------- DECISIÓN ESTABLE ---------
        if frame_count >= STABLE_FRAMES and not processed:
            dominant_class = Counter(class_votes).most_common(1)[0][0]

            print("✅ Mandarina estable detectada")
            print(f"👉 Clase dominante YOLO: {dominant_class}")

            # Recorte
            roi = frame[y1:y2, x1:x2].copy()
            cv2.imwrite("mandarina_recortada.jpg", roi)
            print("📸 Recorte guardado")

            processed = True  # Bloquea nuevos recortes

    else:
        # No hay detección → reset
        frame_count = 0
        class_votes = []
        processed = False
        last_bbox = None

    cv2.imshow("YOLO Mandarina", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



#limitar resolucion 
#cap = cv2.VideoCapture(0)

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#limitar fps
