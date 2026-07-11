
import cv2
import torch
#import time 
import numpy as np
from collections import Counter
from tensorflow.keras.models import load_model

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
calidad = None

# ================= CARGAR MODELO YOLOv5 =================
yolo = torch.hub.load(
    'yolov5',        # carpeta local yolov5
    'custom',
    path='best.pt',
    source='local'
)
yolo.conf = CONF_THRESHOLD
print("Clases del modelo:")
print(yolo.names)


#================= modelo de desicion ====================
# ================= CARGAR RED DE CALIDAD =================
calidad_model = load_model("BuenasMalas.h5")

INPUT_SIZE = (100, 100)
UMBRAL_MALA = 0.5



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

    results = yolo(frame)
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
            #print(f"👉 Clase dominante: {dominant_class}")

            # Recorte final
            roi = frame[y1:y2, x1:x2].copy()
            #cv2.imwrite("mandarina_recortada.jpg", roi)
            #print("📸 Recorte guardado")



####
        #================= Pre procesamiento ===============

            roi_resized = cv2.resize(roi, INPUT_SIZE)
            roi_resized = roi_resized / 255.0
            roi_resized = np.expand_dims(roi_resized, axis=0)

        #================== Inferencia de la red ==============
            

            score = calidad_model.predict(roi_resized, verbose=0)[0][0]

            if score >= UMBRAL_MALA:
                calidad = "MALA"
                estado = 1
            else:
                calidad = "BUENA"
                estado = 0

                        #solucion recorte eliminado (se eliminaban esto es solo pruebas en masa no es necesario)
            if calidad is not None:

                contador += 1
                filename = f"mandarina_{contador}_{calidad}.jpg"
                cv2.imwrite(filename, roi)

            print(f"📊 Score red: {score:.3f} → {calidad}")

            # ---------- VISUAL ----------
            cv2.putText(
                frame,
                f"{calidad} ({score:.2f})",
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

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