# =============================================================================
# main.py — Orquestador principal del sistema de clasificación SWIR
# =============================================================================
import threading
import time
import queue
import cv2
import numpy as np
from collections import deque
from typing import Optional

import config
from camera import crear_camara, CamaraBase
from buffer import RollingBuffer
from pseudo_rgb import construir_pseudo_rgb, pseudo_rgb_para_display
from detector import DetectorYOLO
from tracker import Tracker
from classifier import Clasificador
from actuator import GestorActuadores
from preprocessing import preprocess_channels

# =============================================================================
YOLO_CADA_N_LINEAS: int = 15

ids_en_clasificacion: set = set()
lock_ids_cnn = threading.Lock()


# =============================================================================
# HILO CLASIFICADOR
# =============================================================================

def worker_clasificador(
    cola_entrada: queue.Queue,
    cola_salida: queue.Queue,
    buffer: RollingBuffer,
    clasificador: Clasificador
) -> None:
    while True:
        try:
            trabajo = cola_entrada.get(timeout=0.5)
        except queue.Empty:
            continue

        if trabajo is None:
            break

        id_obj, obj, linea_inicio_snap = trabajo

        try:
            # ── Extraer ROI crudo del buffer ──────────────────────────
            roi_crudo = buffer.extraer_roi_espectral(
                bbox_x1=obj.x1,
                bbox_x2=obj.x2,
                bbox_y1=obj.y1,
                bbox_y2=obj.y2,
                linea_inicio_frame=linea_inicio_snap
            )

            if roi_crudo is None or roi_crudo.size == 0:
                if config.LOG_LEVEL >= 1:
                    print(f"[CNN] ID={id_obj}: ROI inválido o expirado.")
                with lock_ids_cnn:
                    ids_en_clasificacion.discard(id_obj)
                continue

            # ── Preprocesamiento Vy (norm por canal + gamma) ──────────
            # Aplicado UNA SOLA VEZ — identical al entrenamiento
            canales = [roi_crudo[:, :, b] for b in range(roi_crudo.shape[2])]
            roi_procesado = preprocess_channels(canales)
            # roi_procesado: (H, W, N_BANDS) float32 en [0,1]

            # ── Clasificar ────────────────────────────────────────────
            resultado = clasificador.predecir(roi_procesado)

            # ── Encolar resultado ─────────────────────────────────────
            cola_salida.put({
                "id_obj"      : id_obj,
                "carril"      : obj.carril,
                "etiqueta"    : "BUENA" if resultado.es_buena else "MALA",
                "score"       : resultado.probabilidad,
                "tiempo_cruce": time.time()
            })

            if config.LOG_LEVEL >= 1:
                print(f"[CNN] ID={id_obj} carril={obj.carril} → "
                      f"{'BUENA' if resultado.es_buena else 'MALA'} "
                      f"score={resultado.probabilidad:.3f}")

        except Exception as e:
            print(f"[CNN] Error clasificando ID={id_obj}: {e}")
        finally:
            with lock_ids_cnn:
                ids_en_clasificacion.discard(id_obj)


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

# Colores BGR para cv2
COLOR_VERDE    = (0, 200, 0)
COLOR_ROJO     = (0, 60, 220)
COLOR_AMARILLO = (0, 180, 255)
COLOR_CYAN     = (255, 200, 0)
COLOR_GRIS     = (160, 160, 160)
COLOR_BLANCO   = (255, 255, 255)
COLOR_NEGRO    = (0, 0, 0)
COLOR_PANEL    = (30, 30, 30)
COLOR_LINEA_SEP= (70, 70, 70)


def construir_canvas(
    pseudo_rgb:      np.ndarray,
    objetos_activos: dict,
    ultima_clasif:   Optional[dict],
    contadores:      dict,
    fps_yolo:        float,
    fps_camara:      float,
    estado_modulos:  dict,
    linea_virtual_x: int,
    escala_x:        float,
    x_off:           int,
    y_off:           int,
    img_panel_h:     int
) -> np.ndarray:
    """
    Construye el canvas completo de la ventana fija.

    Panel izquierdo: pseudo-RGB con overlays.
    Panel derecho: estado, contadores, módulos, config.
    """
    W  = config.WINDOW_WIDTH
    H  = config.WINDOW_HEIGHT
    RP = config.RIGHT_PANEL_WIDTH
    LP = W - RP  # ancho panel izquierdo

    canvas = np.full((H, W, 3), COLOR_PANEL, dtype=np.uint8)

    # ── Panel izquierdo — pseudo-RGB ──────────────────────────────────
    # pseudo_rgb ya viene en BGR (convertido antes de llamar aquí)
    panel_img = pseudo_rgb  # (img_panel_h, LP, 3)
    if panel_img.shape[0] <= H and panel_img.shape[1] <= LP:
        canvas[:panel_img.shape[0], :panel_img.shape[1]] = panel_img

    # Línea de clasificación virtual
    lx = x_off + int(linea_virtual_x * escala_x)
    cv2.line(canvas, (lx, 0), (lx, H), (255, 120, 0), 2)

    # Línea separación de carriles
    mitad_y = y_off + img_panel_h // 2
    cv2.line(canvas, (0, mitad_y), (LP, mitad_y), COLOR_GRIS, 1)
    cv2.putText(canvas, "Carril 2", (8, mitad_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 255), 1)
    cv2.putText(canvas, "Carril 1", (8, mitad_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 255), 1)

    # Objetos trackeados con bboxes
    for id_obj, obj in objetos_activos.items():
        bx1 = x_off + int(obj.x1 * escala_x)
        bx2 = x_off + int(obj.x2 * escala_x)
        by1 = y_off + obj.y1
        by2 = y_off + obj.y2
        bcx = x_off + int(obj.cx * escala_x)
        bcy = y_off + obj.cy

        if obj.clasificado:
            es_buena = getattr(obj, 'es_buena', None)
            color = COLOR_VERDE if es_buena else COLOR_ROJO
            label_estado = "GOOD" if es_buena else "BAD"
            score = getattr(obj, 'score', None)
            label = (f"ID:{id_obj} C:{obj.carril} {label_estado}"
                     + (f" {score:.2f}" if score is not None else ""))
        else:
            color = COLOR_AMARILLO
            label = f"ID:{id_obj} C:{obj.carril}"

        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, 2)
        cv2.circle(canvas, (bcx, bcy), 4, (0, 0, 255), -1)
        cv2.putText(canvas, label, (bx1, max(by1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    # FPS en panel izquierdo
    cv2.putText(canvas, f"FPS Cam:{fps_camara:.1f}  YOLO:{fps_yolo:.1f}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # ── Separador vertical ─────────────────────────────────────────────
    cv2.line(canvas, (LP, 0), (LP, H), COLOR_LINEA_SEP, 2)

    # ── Panel derecho ─────────────────────────────────────────────────
    px = LP + 10   # x de inicio del texto
    py = 0         # y acumulador

    def sep_linea(y, grosor=1):
        cv2.line(canvas, (LP + 2, y), (W - 2, y), COLOR_LINEA_SEP, grosor)
        return y + 8

    def texto(txt, y, escala=0.42, color=COLOR_BLANCO, negrita=False):
        grosor = 2 if negrita else 1
        cv2.putText(canvas, txt, (px, y),
                    cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor)
        return y + int(escala * 38)

    # -- BLOQUE 1: Última clasificación --
    py += 18
    py = texto("ULTIMA CLASIFICACION", py, 0.45, COLOR_GRIS)
    py += 4

    if ultima_clasif is not None:
        es_buena = ultima_clasif["etiqueta"] == "BUENA"
        color_uc = COLOR_VERDE if es_buena else COLOR_ROJO
        label_uc = "BUENA" if es_buena else "MALA"
        signo    = "+" if es_buena else "-"

        # Rectángulo de fondo grande
        rect_y1 = py - 4
        rect_y2 = py + 54
        cv2.rectangle(canvas, (LP + 4, rect_y1), (W - 4, rect_y2),
                       color_uc, -1)
        cv2.putText(canvas, f"{signo} {label_uc}",
                    (px, py + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, COLOR_BLANCO, 3)
        py = rect_y2 + 10
    else:
        py = texto("  Sin datos aun", py, 0.42, COLOR_GRIS)
        py += 10

    py = sep_linea(py)

    # -- BLOQUE 2: Detalle técnico --
    if ultima_clasif is not None:
        py = texto(f"Tracker : {ultima_clasif['id_obj']}", py)
        py = texto(f"Carril  : {ultima_clasif['carril']}", py)
        score_pct = ultima_clasif['score'] * 100
        py = texto(f"Confianza CNN : {score_pct:.1f} %", py)
        hora = time.strftime('%H:%M:%S',
               time.localtime(ultima_clasif['tiempo_cruce']))
        py = texto(f"Hora    : {hora}", py)
    else:
        py = texto("  ---", py, 0.42, COLOR_GRIS)
    py += 4
    py = sep_linea(py)

    # -- BLOQUE 3: Contadores --
    py += 4
    py = texto("CONTADORES DE SESION", py, 0.42, COLOR_GRIS)
    py += 4
    py = texto(f"Buenas  : {contadores['buenas']}", py, 0.48, COLOR_VERDE)
    py = texto(f"Malas   : {contadores['malas']}", py, 0.48, COLOR_ROJO)
    total = contadores['buenas'] + contadores['malas']
    py = texto(f"Total   : {total}", py, 0.48, COLOR_BLANCO)
    py += 4
    py = sep_linea(py)

    # -- BLOQUE 4: Estado de módulos --
    py += 4
    py = texto("ESTADO DEL SISTEMA", py, 0.42, COLOR_GRIS)
    py += 4
    for nombre, activo in estado_modulos.items():
        icono = "OK" if activo else "XX"
        color_m = COLOR_VERDE if activo else COLOR_ROJO
        py = texto(f"{nombre:<10} {icono}", py, 0.42, color_m)
    py += 4
    py = sep_linea(py)

    # -- BLOQUE 5: Config activa --
    py += 4
    py = texto("CONFIG ACTIVA", py, 0.42, COLOR_GRIS)
    py += 2
    py = texto(f"Pico    : {config.SPECTRAL_PEAK_BAND}", py, 0.40)
    py = texto(f"Offsets : {config.OFFSETS_STACK}", py, 0.36)
    py = texto(f"Gamma   : {config.PREP_GAMMA}", py, 0.40)
    py = texto(f"p1/p99.5: {config.PREP_P_LOW}/{config.PREP_P_HIGH}", py, 0.40)

    return canvas


# =============================================================================
# LOOP PRINCIPAL
# =============================================================================

def main(usar_camara_simulada: bool = False) -> None:
    print("=" * 60)
    print("SISTEMA DE CLASIFICACIÓN SWIR — MANDARINAS")
    print("=" * 60)
    print(f"  Offsets Vy         : {config.OFFSETS_STACK}")
    print(f"  Pseudo-RGB índices : {config.PSEUDO_RGB_BAND_INDICES}")
    print(f"  Gamma              : {config.PREP_GAMMA}")
    print(f"  Buffer             : {config.BUFFER_N_LINES} líneas")
    print(f"  Clasificador       : {config.CLASSIFIER_TYPE}")
    print(f"  Arduino            : {config.ARDUINO_PORT}")
    print(f"  Ventana            : {config.WINDOW_WIDTH}×{config.WINDOW_HEIGHT}")
    print("=" * 60)

    # ── Inicializar componentes ────────────────────────────────────────
    camara:      CamaraBase    = crear_camara(forzar_simulada=usar_camara_simulada)
    buffer:      RollingBuffer = RollingBuffer()
    detector:    DetectorYOLO  = DetectorYOLO()
    tracker:     Tracker       = Tracker()
    clasificador: Clasificador = Clasificador()
    actuador:    GestorActuadores = GestorActuadores()

    detector.iniciar()
    clasificador.iniciar()
    actuador.iniciar()

    # ── Colas ─────────────────────────────────────────────────────────
    cola_clasificacion: queue.Queue = queue.Queue(maxsize=50)
    cola_resultados:    queue.Queue = queue.Queue(maxsize=50)

    # ── Hilo clasificador ──────────────────────────────────────────────
    hilo_cnn = threading.Thread(
        target=worker_clasificador,
        args=(cola_clasificacion, cola_resultados, buffer, clasificador),
        name="hilo_clasificador",
        daemon=True
    )
    hilo_cnn.start()
    print("[MAIN] Hilo clasificador iniciado.")

    # ── Conectar cámara ────────────────────────────────────────────────
    camara.conectar()

    # ── Calibración background ─────────────────────────────────────────
    UMBRAL_BG_MAXIMO = getattr(config, 'BACKGROUND_UMBRAL_MAXIMO', 30.0)
    if not config.BACKGROUND_CORRECTION_ON or usar_camara_simulada:
        # Background OFF: pasar vector de ceros sin invocar calibración
        background = np.zeros(config.CAMERA_N_BANDS, dtype=np.float32)
        buffer.set_background(background)
        print("[CAL] Background correction OFF — ceros aplicados.")
    else:
        background_valido = False
        while not background_valido:
            print("\n[CAL] ── CALIBRACIÓN DE BACKGROUND ─────────────────────")
            print("[CAL] Banda VACÍA y SIN mandarinas.")
            input("[CAL] Presiona Enter para iniciar...")
            try:
                background = camara.calibrar_background(n_frames=100)
            except Exception as e:
                print(f"[CAL] ERROR: {e}")
                resp = input("[CAL] [R]eintentar / [Q]salir: ").strip().lower()
                if resp == 'q':
                    return
                continue
            bg_mean = float(np.mean([background[i] for i in config.SPECTRAL_BANDS]))
            if bg_mean > UMBRAL_BG_MAXIMO:
                print(f"[CAL] ⚠️  Perfil alto (mean={bg_mean:.2f}). "
                      f"[R]epetir/[C]ontinuar/[Q]salir: ", end="")
                resp = input().strip().lower()
                if resp == 'q':
                    return
                elif resp == 'c':
                    background_valido = True
            else:
                print(f"[CAL] ✅ Válido (mean={bg_mean:.2f})")
                background_valido = True
        buffer.set_background(background)

    # ── Iniciar hilo de captura ────────────────────────────────────────
    camara.iniciar()

    # ── Llenado inicial del buffer ─────────────────────────────────────
    print("[MAIN] Llenando buffer inicial...")
    while not buffer.esta_listo():
        linea = camara.get_linea(timeout=0.05)
        if linea is not None:
            buffer.agregar_linea(linea)
    print("[MAIN] Buffer listo. Iniciando detección.")

    # ── Estado de visualización ────────────────────────────────────────
    fps_yolo_counter  = deque(maxlen=30)
    fps_cam_counter   = deque(maxlen=60)
    t_ultimo_yolo     = time.time()
    t_ultimo_fps_cam  = time.time()
    lineas_desde_ultimo_yolo:  int = 0
    lineas_procesadas_total:   int = 0

    ultima_clasif:  Optional[dict] = None
    contadores = {"buenas": 0, "malas": 0}
    estado_modulos = {
        "Camara"  : True,
        "YOLO"    : True,
        "CNN"     : True,
        "Arduino" : actuador.arduino_conectado
               if hasattr(actuador, 'arduino_conectado') else True,
        "Tracker" : True,
    }

    # Crear ventana fija
    if config.DISPLAY_ENABLED:
        cv2.namedWindow("Clasificador SWIR", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Clasificador SWIR",
                         config.WINDOW_WIDTH, config.WINDOW_HEIGHT)

    print("[MAIN] Sistema activo. Presiona 'q' para salir.")

    LP = config.WINDOW_WIDTH - config.RIGHT_PANEL_WIDTH  # ancho panel izq

    try:
        while True:
            # ── 1. Consumir líneas de cámara ───────────────────────────
            lineas_nuevas = 0
            while True:
                linea = camara.get_linea(timeout=0.001)
                if linea is None:
                    break
                buffer.agregar_linea(linea)
                lineas_nuevas += 1
                lineas_procesadas_total += 1
                if lineas_nuevas >= YOLO_CADA_N_LINEAS * 2:
                    break

            lineas_desde_ultimo_yolo += lineas_nuevas

            # FPS cámara
            if lineas_nuevas > 0:
                ahora_cam = time.time()
                dt_cam = ahora_cam - t_ultimo_fps_cam
                t_ultimo_fps_cam = ahora_cam
                if dt_cam > 0:
                    fps_cam_counter.append(lineas_nuevas / dt_cam)
            fps_camara = (sum(fps_cam_counter) / len(fps_cam_counter)
                          if fps_cam_counter else 0.0)

            # ── 2. Ejecutar YOLO ───────────────────────────────────────
            if lineas_desde_ultimo_yolo < YOLO_CADA_N_LINEAS:
                time.sleep(0.001)
                continue
            lineas_desde_ultimo_yolo = 0

            # ── 3. Frame del buffer ────────────────────────────────────
            frame, linea_inicio_frame = buffer.get_frame()

            # ── 4. Pseudo-RGB Vy ───────────────────────────────────────
            pseudo_rgb = construir_pseudo_rgb(frame)

            # ── 5. Detección YOLO ──────────────────────────────────────
            detecciones = detector.detectar(pseudo_rgb, linea_inicio_frame)

            # ── 6. Tracker ─────────────────────────────────────────────
            objetos_activos = tracker.actualizar(detecciones)

            # ── 7. Zona de clasificación ───────────────────────────────
            for id_obj, obj in objetos_activos.items():
                if obj.clasificado:
                    continue
                with lock_ids_cnn:
                    if id_obj in ids_en_clasificacion:
                        continue
                if not tracker.objeto_en_zona_clasificacion(obj):
                    continue
                if obj.x2 <= obj.x1 or obj.y2 <= obj.y1:
                    continue
                tracker.marcar_clasificado(id_obj)
                with lock_ids_cnn:
                    ids_en_clasificacion.add(id_obj)
                try:
                    cola_clasificacion.put_nowait(
                        (id_obj, obj, obj.linea_inicio_frame))
                    if config.LOG_LEVEL >= 1:
                        print(f"[MAIN] ID={id_obj} → clasificación (cx={obj.cx})")
                except queue.Full:
                    tracker._objetos[id_obj].clasificado = False
                    with lock_ids_cnn:
                        ids_en_clasificacion.discard(id_obj)

            # ── 8. Resultados del clasificador ─────────────────────────
            while not cola_resultados.empty():
                try:
                    res = cola_resultados.get_nowait()
                    ultima_clasif = res

                    if res["etiqueta"] == "BUENA":
                        contadores["buenas"] += 1
                        # Fail-safe: solo se dispara el mecanismo para BUENA.
                        # MALA, objetos no detectados y fallos del sistema
                        # siguen de largo hacia la salida de rechazo.
                        print(f"[MAIN] Clasificación BUENA → "
                              f"programando disparo carril {res['carril']}")
                        #print(f"[DEBUG-1] Encolando evento id={res['id_obj']} carril={res['carril']}") #debug 09_07_26
                        actuador.encolar_evento(
                            id_obj      =res["id_obj"],
                            carril      =res["carril"],
                            etiqueta    =res["etiqueta"],
                            score       =res["score"],
                            tiempo_cruce=res["tiempo_cruce"]
                        )
                    else:
                        contadores["malas"] += 1

                    # Actualizar color del objeto en tracker
                    if res["id_obj"] in tracker._objetos:
                        tracker._objetos[res["id_obj"]].es_buena = \
                            res["etiqueta"] == "BUENA"
                        tracker._objetos[res["id_obj"]].score = res["score"]
                except queue.Empty:
                    break

            # ── 9. FPS YOLO ────────────────────────────────────────────
            ahora = time.time()
            dt = ahora - t_ultimo_yolo
            t_ultimo_yolo = ahora
            if dt > 0:
                fps_yolo_counter.append(dt)
            fps_yolo = (1.0 / (sum(fps_yolo_counter) / len(fps_yolo_counter))
                        if fps_yolo_counter else 0.0)

            # ── 10. Visualización ──────────────────────────────────────
            if config.DISPLAY_ENABLED:
                # Convertir pseudo-RGB a BGR para cv2 y escalar al panel
                pseudo_bgr = cv2.cvtColor(pseudo_rgb, cv2.COLOR_RGB2BGR)
                panel_img, escala, x_off, y_off = pseudo_rgb_para_display(
                    pseudo_bgr,
                    target_h=config.WINDOW_HEIGHT,
                    target_w=LP
                )
                escala_x = escala

                canvas = construir_canvas(
                    pseudo_rgb      = panel_img,
                    objetos_activos = objetos_activos,
                    ultima_clasif   = ultima_clasif,
                    contadores      = contadores,
                    fps_yolo        = fps_yolo,
                    fps_camara      = fps_camara,
                    estado_modulos  = estado_modulos,
                    linea_virtual_x = tracker.linea_virtual_x,
                    escala_x        = escala_x,
                    x_off           = x_off,
                    y_off           = y_off,
                    img_panel_h     = panel_img.shape[0]
                )

                cv2.imshow("Clasificador SWIR", canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[MAIN] Salida por usuario.")
                    break

    except KeyboardInterrupt:
        print("\n[MAIN] Interrupción por teclado.")

    finally:
        print("[MAIN] Cerrando sistema...")
        cola_clasificacion.put(None)
        camara.detener()
        actuador.detener()
        if config.DISPLAY_ENABLED:
            cv2.destroyAllWindows()
        print(f"[MAIN] Líneas procesadas: {lineas_procesadas_total}")
        print(f"[MAIN] Sesión — Buenas: {contadores['buenas']} | "
              f"Malas: {contadores['malas']}")
        print("[MAIN] Sistema detenido correctamente.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Sistema de clasificación SWIR de mandarinas"
    )
    parser.add_argument("--sim", action="store_true",
                        help="Usar cámara simulada (sin hardware)")
    args = parser.parse_args()
    main(usar_camara_simulada=args.sim)
