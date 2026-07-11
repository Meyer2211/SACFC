# =============================================================================
# actuator.py — Scheduler en Python + ejecutor serial para Arduino
# =============================================================================
# Arquitectura Vy:
#   Python es el único responsable del timing.
#   Arduino únicamente ejecuta órdenes inmediatas (L\n o R\n).
#
# Flujo:
#   encolar_evento() → cola Python → hilo scheduler espera CENTER_CROSSING_DELAY_S
#   → envía L\n o R\n al Arduino → Arduino mueve servo inmediatamente.
#
# Protocolo serial:
#   Python → Arduino: "L\n" (carril izquierdo) | "R\n" (carril derecho)
#   Arduino → Python: "READY\n" al iniciar | "FIRE:L\n" | "FIRE:R\n" | "CLOSE:L\n"
#
# Contrato de tiempo:
#   tiempo_cruce : obtenido con time.time() — timestamp absoluto para logs/CSV.
#   t_disparo    : obtenido con time.monotonic() — usado internamente por el
#                  scheduler para esperar con precisión, nunca para logs.
#   tiempo_programado : time.time() + delay_s — guardado al crear el evento
#                       para poder comparar programado vs real en CSV.
#
# Fail-safe:
#   Solo se encolan eventos BUENA (decisión en main.py).
#   Arduino sin cola ni timers → si pierde conexión, servos quedan en reposo.
# =============================================================================

import serial
import threading
import time
import queue
import csv
from dataclasses import dataclass, field
from typing import Optional, List
import config


@dataclass
class EventoActuador:
    """Evento de accionamiento pendiente en la cola del scheduler Python."""
    id_obj           : int
    carril           : int       # 1=izquierdo, 2=derecho
    etiqueta         : str       # "BUENA" — solo para log/CSV
    score            : float
    tiempo_cruce     : float     # time.time() del cruce — para logs y CSV
    t_disparo        : float     # time.monotonic() del disparo programado
    tiempo_programado: float     # time.time() + delay_s — para CSV


# Mapeo carril → comando serial
_CMD_CARRIL = {1: "L", 2: "R"}


class GestorActuadores:
    """
    Scheduler de disparos en Python + comunicación serial con Arduino.
    Arduino es un ejecutor inmediato: solo recibe L o R y mueve el servo.
    """

    def __init__(self):
        self._serial       : Optional[serial.Serial] = None
        self._cola         : queue.PriorityQueue = queue.PriorityQueue()
        self._running      : bool = False
        self._hilo_sched   : Optional[threading.Thread] = None
        self._hilo_ack     : Optional[threading.Thread] = None
        self._evento_sched : threading.Event = threading.Event()

        # Estadísticas
        self._disparos_enviados: int = 0
        self._errores_serial   : int = 0

        # Log CSV
        self._csv_file   = None
        self._csv_writer = None

    # ------------------------------------------------------------------
    # INICIALIZACIÓN
    # ------------------------------------------------------------------

    def iniciar(self) -> None:
        print(f"[ACTUATOR] Conectando Arduino en {config.ARDUINO_PORT} "
              f"@ {config.ARDUINO_BAUDRATE} bps...")

        try:
            self._serial = serial.Serial(
                port          = config.ARDUINO_PORT,
                baudrate      = config.ARDUINO_BAUDRATE,
                timeout       = config.ARDUINO_TIMEOUT_S,
                write_timeout = 1.0   # evita bloqueo infinito en write()
            )
            # Limpiar buffer antes de esperar READY
            self._serial.reset_input_buffer()
            self._esperar_ready(timeout_s=5.0)
            print("[ACTUATOR] Arduino conectado y listo.")
        except serial.SerialException as e:
            print(f"[ACTUATOR] ERROR al conectar Arduino: {e}")
            print("[ACTUATOR] El sistema continuará SIN control de servos.")
            self._serial = None

        # Log CSV
        if config.LOG_TO_CSV:
            try:
                self._csv_file = open(config.LOG_CSV_PATH, 'w',
                                      newline='', encoding='utf-8')
                self._csv_writer = csv.writer(self._csv_file)
                self._csv_writer.writerow([
                    'timestamp_real', 'id_obj', 'carril', 'etiqueta',
                    'score', 'tiempo_cruce', 'tiempo_programado',
                    'retardo_real_s'
                ])
                print(f"[ACTUATOR] Log CSV: {config.LOG_CSV_PATH}")
            except Exception as e:
                print(f"[ACTUATOR] No se pudo crear CSV: {e}")

        self._running = True

        # Hilo scheduler
        self._hilo_sched = threading.Thread(
            target=self._hilo_scheduler,
            name="hilo_actuador_sched",
            daemon=True
        )
        self._hilo_sched.start()

        # Hilo ACK — solo si hay conexión serial y ACK habilitado
        if self._serial is not None and config.ARDUINO_ACK_ENABLED:
            self._hilo_ack = threading.Thread(
                target=self._hilo_leer_ack,
                name="hilo_actuador_ack",
                daemon=True
            )
            self._hilo_ack.start()
        elif self._serial is not None:
            print("[ACTUATOR] Hilo ACK deshabilitado "
                  "(ARDUINO_ACK_ENABLED=False en config.py).")

        print("[ACTUATOR] Scheduler iniciado.")

    def _esperar_ready(self, timeout_s: float = 5.0) -> None:
        """Espera el mensaje READY del Arduino con timeout."""
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                linea = self._serial.readline().decode('ascii', errors='ignore').strip()
                if linea == "READY":
                    print("[ACTUATOR] Arduino READY.")
                    return
            except serial.SerialException:
                break
        print("[ACTUATOR] ADVERTENCIA: no se recibió READY del Arduino "
              f"en {timeout_s}s — continuando de todas formas.")

    # ------------------------------------------------------------------
    # ENCOLAR EVENTO
    # ------------------------------------------------------------------

    def encolar_evento(
        self,
        id_obj      : int,
        carril      : int,
        etiqueta    : str,
        score       : float,
        tiempo_cruce: float
    ) -> None:
        """
        Encola un evento de disparo.

        Args:
            id_obj       : ID del objeto clasificado
            carril       : 1 (izquierdo) o 2 (derecho)
            etiqueta     : "BUENA" — informativo para logs
            score        : probabilidad del clasificador
            tiempo_cruce : time.time() del cruce de línea virtual

        El scheduler esperará CENTER_CROSSING_DELAY_S desde este instante
        antes de enviar el comando al Arduino.
        """
        delay_s           = max(0.0, config.CENTER_CROSSING_DELAY_S)
        t_disparo         = time.monotonic() + delay_s
        tiempo_programado = time.time() + delay_s

        evento = EventoActuador(
            id_obj           = id_obj,
            carril           = carril,
            etiqueta         = etiqueta,
            score            = score,
            tiempo_cruce     = tiempo_cruce,
            t_disparo        = t_disparo,
            tiempo_programado= tiempo_programado
        )

        # PriorityQueue ordena por t_disparo (primer elemento de la tupla)
        self._cola.put((t_disparo, evento))
        self._evento_sched.set()  # despertar scheduler si está dormido

        if config.LOG_LEVEL >= 1:
            print(f"[ACTUATOR] Evento encolado | Carril {carril} | "
                  f"Delay={delay_s:.2f}s | "
                  f"Programado={tiempo_programado:.2f}")

    # ------------------------------------------------------------------
    # HILO SCHEDULER
    # ------------------------------------------------------------------

    def _hilo_scheduler(self) -> None:
        """
        Hilo dedicado que espera el momento de disparo de cada evento.
        Usa time.monotonic() para esperar con precisión.

        Una vez alcanzado el instante de disparo, vacía todos los eventos
        cuya t_disparo <= ahora antes de volver a esperar. Esto garantiza
        que eventos con tiempos iguales o muy próximos se transmitan
        consecutivamente (L, R) con separación de microsegundos.

        El sleep es interrumpible via _evento_sched para reevaluar la cola
        si llega un evento más urgente.
        """
        while self._running:
            # ── Esperar el próximo evento ──────────────────────────────
            try:
                t_disparo, evento = self._cola.get(timeout=0.05)
            except queue.Empty:
                continue

            # ── Esperar hasta el momento de disparo (interrumpible) ────
            while self._running:
                ahora    = time.monotonic()
                restante = t_disparo - ahora
                if restante <= 0:
                    break
                self._evento_sched.clear()
                despertado = not self._evento_sched.wait(timeout=restante)

                if not despertado:
                    # Nuevo evento llegó — verificar si es más urgente
                    try:
                        t_nuevo, ev_nuevo = self._cola.get_nowait()
                        if t_nuevo < t_disparo:
                            self._cola.put((t_disparo, evento))
                            t_disparo, evento = t_nuevo, ev_nuevo
                        else:
                           self._cola.put((t_nuevo, ev_nuevo))  # ← devolver si no es más urgente
                    except queue.Empty:
                        pass

            # ── Despachar el evento actual ─────────────────────────────
            self._enviar_comando(evento)

            # ── Vaciar todos los eventos vencidos antes de volver a esperar
            while self._running:
                try:
                    t_sig, ev_sig = self._cola.get_nowait()
                    if t_sig <= time.monotonic():
                        self._enviar_comando(ev_sig)
                    else:
                        self._cola.put((t_sig, ev_sig))
                        break
                except queue.Empty:
                    break

    def _enviar_comando(self, evento: EventoActuador) -> None:
        """Envía el comando L o R al Arduino inmediatamente."""
        cmd = _CMD_CARRIL.get(evento.carril, "L")
        mensaje = f"{cmd}\n"
        retardo_real = time.time() - evento.tiempo_cruce

        if self._serial is not None and self._serial.is_open:
            try:
                if not self._serial.is_open:
                    raise serial.SerialException("Puerto serial cerrado")
                n = self._serial.write(mensaje.encode('ascii'))
                if n != len(mensaje):
                    raise serial.SerialTimeoutException(
                        f"Solo se enviaron {n} de {len(mensaje)} bytes"
                    )
                self._disparos_enviados += 1

                if config.LOG_LEVEL >= 1:
                    print(f"[ACTUATOR] DISPARO | Carril {evento.carril} | "
                          f"Retardo={retardo_real:.2f}s")
                    print(f"[ARDUINO ←] {mensaje.strip()} (ID={evento.id_obj})")

            except (serial.SerialException, serial.SerialTimeoutException) as e:
                self._errores_serial += 1
                print(f"[ACTUATOR] Error serial al enviar: {e}")
        else:
            self._disparos_enviados += 1
            if config.LOG_LEVEL >= 1:
                print(f"[ACTUATOR] DISPARO SIM | Carril {evento.carril} | "
                      f"Retardo={retardo_real:.2f}s")
                print(f"[ACTUATOR SIM] {mensaje.strip()} (ID={evento.id_obj})")

        # Log CSV
        if self._csv_writer is not None:
            try:
                self._csv_writer.writerow([
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                    evento.id_obj,
                    evento.carril,
                    evento.etiqueta,
                    f"{evento.score:.4f}",
                    f"{evento.tiempo_cruce:.3f}",
                    f"{evento.tiempo_programado:.3f}",
                    f"{retardo_real:.3f}"
                ])
                self._csv_file.flush()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # HILO LECTURA ACK
    # ------------------------------------------------------------------

    def _hilo_leer_ack(self) -> None:
        """
        Lee mensajes de confirmación del Arduino (FIRE, CLOSE, etc.).
        Solo se inicia si hay conexión serial válida.
        Ignora silenciosamente bytes corruptos pero registra si el puerto cierra.
        """
        while self._running:
            try:
                if self._serial is None or not self._serial.is_open:
                    time.sleep(0.1)
                    continue
                linea = self._serial.readline()
                if linea:
                    msg = linea.decode('ascii', errors='ignore').strip()
                    if msg and config.LOG_LEVEL >= 1:
                        print(f"[ARDUINO →] {msg}")
            except serial.SerialException as e:
                print(f"[ACTUATOR] ADVERTENCIA: puerto serial cerrado — {e}")
                break
            except Exception:
                pass

    # ------------------------------------------------------------------
    # CIERRE
    # ------------------------------------------------------------------

    def detener(self) -> None:
        self._running = False
        self._evento_sched.set()  # despertar scheduler para que termine

        if self._hilo_sched and self._hilo_sched.is_alive():
            self._hilo_sched.join(timeout=2.0)

        if self._hilo_ack and self._hilo_ack.is_alive():
            self._hilo_ack.join(timeout=1.0)

        if self._serial and self._serial.is_open:
            try:
                self._serial.close()
                print("[ACTUATOR] Puerto serial cerrado.")
            except Exception:
                pass

        if self._csv_file:
            self._csv_file.close()

        print(f"[ACTUATOR] Estadísticas:")
        print(f"  Disparos enviados : {self._disparos_enviados}")
        print(f"  Errores serial    : {self._errores_serial}")

    @property
    def disparos_enviados(self) -> int:
        return self._disparos_enviados

    @property
    def arduino_conectado(self) -> bool:
        return self._serial is not None and self._serial.is_open
