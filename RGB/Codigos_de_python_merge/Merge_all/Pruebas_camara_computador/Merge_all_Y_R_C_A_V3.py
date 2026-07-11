import cv2
import torch
import numpy as np
from tensorflow.keras.models import load_model
import serial
import time

# ================= ARDUINO =================
arduino = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2)
print("Arduino listo")

# ================= CONFIGURACION GENERAL =================
CONF_THRESHOLD = 0.75
MIN_AREA = 4000
MANDARINA_CLASSES = [0, 1]

INPUT_SIZE = (100, 100)
UMBRAL_MALA = 0.5

NUM_CARRILES = 4
ZONA_REGISTRO_ANCHO = 20

# =========================================================
# =============== PARAMETROS MECANICOS ====================
# =========================================================

RPM_BANDA = 300              # <-- 🔧 CAMBIAR RPM AQUI
DIAMETRO_RODILLO_CM = 5      # <-- 🔧 CAMBIAR DIAMETRO SI CAMBIA
DISTANCIA_CM = 40            # <-- 🔧 CAMBIAR DISTANCIA LINEA→SERVO

# ================= CALCULO AUTOMATICO ====================
VELOCIDAD_CM_S = (RPM_BANDA / 60.0) * (np.pi * DIAMETRO_RODILLO_CM)
TIEMPO_DISPARO = DISTANCIA_CM / VELOCIDAD_CM_S

print(f"Velocidad banda: {VELOCIDAD_CM_S:.2f} cm/s")
print(f"Tiempo de disparo calculado: {TIEMPO_DISPARO:.3f} segundos")

# ================= CARGAR MODELOS =================
yolo = torch.hub.load(
    'yolov5',
    'custom',
    path='best.pt',
    source='local'
)
yolo.conf = CONF_THRESHOLD

calidad_model = load_model("BuenasMalas.h5")

# ================= CAMARA =================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ret, frame = cap.read()
if not ret:
    print("Error iniciando cámara")
    exit()

HEIGHT, WIDTH, _ = frame.shape

# ================= DEFINICION DE CARRILES =================
ALTO_CARRIL = HEIGHT // NUM_CARRILES
CARRILES = []

for i in range(NUM_CARRILES):
    y_min = i * ALTO_CARRIL
    y_max = (i + 1) * ALTO_CARRIL
    CARRILES.append((y_min, y_max))

LINEA_X = int(WIDTH * 0.6)

# ================= ESTRUCTURAS =================
posicion_anterior = {}
bloqueo_envio = {i: False for i in range(NUM_CARRILES)}

# 🔥 NUEVO: Cola de eventos temporizados
eventos_pendientes = []

print("Sistema iniciado. Presiona 'q' para salir.")

# ================= LOOP PRINCIPAL =================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = yolo(frame)
    detections = results.xyxy[0]

    # Dibujar línea vertical
    cv2.line(frame, (LINEA_X, 0), (LINEA_X, HEIGHT), (255, 0, 0), 2)

    # Dibujar divisiones horizontales
    for i in range(1, NUM_CARRILES):
        y = i * ALTO_CARRIL
        cv2.line(frame, (0, y), (WIDTH, y), (200, 200, 200), 1)

    # ================= DETECCIONES =================
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

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)

        # Determinar carril
        carril_detectado = None
        for i, (ymin, ymax) in enumerate(CARRILES):
            if ymin <= cy < ymax:
                carril_detectado = i
                break

        if carril_detectado is None:
            continue

        id_obj = carril_detectado

        if id_obj not in posicion_anterior:
            posicion_anterior[id_obj] = cx

        # ================= DETECCIÓN DE CRUCE =================
        if (not bloqueo_envio[carril_detectado] and
            posicion_anterior[id_obj] < LINEA_X - ZONA_REGISTRO_ANCHO and
            LINEA_X - ZONA_REGISTRO_ANCHO <= cx <= LINEA_X + ZONA_REGISTRO_ANCHO):

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

            # 🔥 En vez de enviar inmediatamente, programamos evento
            evento = {
                "carril": carril_detectado,
                "estado": estado,
                "tiempo_envio": time.time() + TIEMPO_DISPARO
            }

            eventos_pendientes.append(evento)

            print(f"[C😀😀😀😀😀arril {carril_detectado}] → {etiqueta} programado")

            bloqueo_envio[carril_detectado] = True

        posicion_anterior[id_obj] = cx

    # ================= LIBERAR BLOQUEO =================
    for i in range(NUM_CARRILES):

        objeto_en_zona = False

        for det in detections:
            x1, y1, x2, y2, conf, cls = det.tolist()
            cx_temp = int((x1 + x2) / 2)
            cy_temp = int((y1 + y2) / 2)

            ymin, ymax = CARRILES[i]

            if ymin <= cy_temp < ymax:
                if LINEA_X - ZONA_REGISTRO_ANCHO <= cx_temp <= LINEA_X + ZONA_REGISTRO_ANCHO:
                    objeto_en_zona = True
                    break

        if not objeto_en_zona:
            bloqueo_envio[i] = False

    # ==========================================================
    # =========== GESTION DE EVENTOS TEMPORIZADOS =============
    # ==========================================================

    tiempo_actual = time.time()

    for evento in eventos_pendientes[:]:
        if tiempo_actual >= evento["tiempo_envio"]:

            mensaje = f'{evento["carril"]},{evento["estado"]}\n'
            arduino.write(mensaje.encode())

            print("DISPARO:", mensaje.strip())

            eventos_pendientes.remove(evento)

    cv2.imshow("Sistema Multi-Carril Horizontal", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()