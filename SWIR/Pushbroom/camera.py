# =============================================================================
# camera.py — Adquisición SWIR con gxipy + extracción de bandas en hilo
# =============================================================================
# Responsabilidades:
#   1. Abrir la cámara Daheng con gxipy y configurar exposición.
#   2. Capturar líneas espectrales crudas (1024, 1280) en hilo dedicado.
#   3. Extraer las 5 bandas Vy de interés → (Y_util, 5) float32 crudo por línea.
#   4. Exponer las líneas reducidas a través de una cola thread-safe.
#      El preprocesamiento Vy (norm por canal + gamma) vive en preprocessing.py.
#
# MODO SIMULADO:
#   CamaraSimulada reproduce muestras reales del Holdout del dataset,
#   emitiendo líneas (Y_util, 5) con el mismo contrato de interfaz
#   que CamaraGXIpy. El resto del pipeline no cambia.
# =============================================================================

import threading
import time
import queue
import numpy as np
from typing import Optional
import config

# Intentar importar gxipy. Si no está disponible, el modo simulado
# permite desarrollar y probar sin hardware.
try:
    import gxipy as gx
    GXIPY_DISPONIBLE = True
except ImportError:
    GXIPY_DISPONIBLE = False
    print("[CAMERA] ADVERTENCIA: gxipy no encontrado. "
          "Solo disponible modo simulado.")


# =============================================================================
# CLASE BASE — interfaz común para cámara real y simulada
# =============================================================================

class CamaraBase:
    """
    Interfaz que toda implementación de cámara debe respetar.
    Permite intercambiar cámara real por simulada sin cambiar main.py.

    Secuencia de uso:
        1. conectar()            — abre hardware y activa stream
        2. calibrar_background() — calibración sin hilo activo
        3. iniciar()             — lanza hilo de captura
    """
    def conectar(self) -> None:
        raise NotImplementedError

    def iniciar(self) -> None:
        raise NotImplementedError

    def detener(self) -> None:
        raise NotImplementedError

    def get_linea(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Retorna una línea reducida (Y_util, N_BANDS) float32,
        o None si no hay línea disponible en el timeout dado.
        Y_util = BUFFER_Y_END - BUFFER_Y_START
        """
        raise NotImplementedError

    def calibrar_background(self, n_frames: int = 100) -> np.ndarray:
        raise NotImplementedError


# =============================================================================
# IMPLEMENTACIÓN REAL — gxipy
# =============================================================================

class CamaraGXIpy(CamaraBase):
    """
    Captura continua con gxipy en hilo dedicado.

    Flujo interno:
      hilo _capturar:
        raw = data_stream[0].get_image()          # (1024, 1280) uint8
        linea_cruda = raw.get_numpy_array()
        linea_bandas = _extraer_bandas(linea_cruda)  # (Y_util, 8) float32
        cola.put(linea_bandas)

    El loop principal consume de la cola sin bloquear el hilo de captura.
    """

    def __init__(self):
        self._cola: queue.Queue = queue.Queue(maxsize=200)
        # maxsize=200 actúa como backpressure: si el procesamiento no consume
        # rápido, el hilo de captura bloquea en put() en lugar de consumir
        # memoria ilimitada.

        self._running: bool = False
        self._hilo: Optional[threading.Thread] = None
        self._cam = None
        self._dm = None

        # Región útil precalculada
        self._y_start = config.BUFFER_Y_START
        self._y_end   = config.BUFFER_Y_END
        self._bandas  = config.SPECTRAL_BANDS  # lista de índices espectrales

        # Contador de líneas para diagnóstico
        self._lineas_capturadas: int = 0
        self._lineas_perdidas: int = 0
        self._t_inicio: float = 0.0

    # ------------------------------------------------------------------
    # INICIALIZACIÓN — separada en dos pasos
    # ------------------------------------------------------------------
    # Secuencia correcta de uso:
    #   1. conectar()          — abre hardware y activa stream
    #   2. calibrar_background() — calibración sin hilo activo
    #   3. iniciar()           — lanza hilo de captura
    # ------------------------------------------------------------------

    def conectar(self) -> None:
        """
        Abre el dispositivo, configura exposición y activa el stream.
        NO lanza el hilo de captura.

        Debe llamarse antes de calibrar_background() e iniciar().
        """
        if not GXIPY_DISPONIBLE:
            raise RuntimeError(
                "gxipy no está instalado. Instalar Galaxy SDK de Daheng "
                "o usar CamaraSimulada para pruebas."
            )

        print("[CAMERA] Inicializando DeviceManager...")
        self._dm = gx.DeviceManager()
        num_camaras, info_list = self._dm.update_all_device_list()

        if num_camaras == 0:
            raise RuntimeError(
                "[CAMERA] No se detectó ninguna cámara. "
                "Verificar conexión GigE, firewall y Galaxy Viewer."
            )

        print(f"[CAMERA] Cámaras detectadas: {num_camaras}")
        for i, info in enumerate(info_list):
            print(f"  [{i+1}] {info}")

        self._cam = self._dm.open_device_by_index(config.CAMERA_INDEX)
        print(f"[CAMERA] Cámara {config.CAMERA_INDEX} abierta.")

        self._configurar_exposicion()
        print("[CAMERA] Stream activo. Listo para calibración.")

    def iniciar(self) -> None:
        """
        Lanza el hilo de captura continua.

        Debe llamarse DESPUÉS de conectar() y calibrar_background().
        El stream ya debe estar activo (conectar lo activa).
        """
        if self._cam is None:
            raise RuntimeError(
                "[CAMERA] Llamar conectar() antes de iniciar()."
            )

        self._running = True
        self._t_inicio = time.time()
        self._hilo = threading.Thread(
            target=self._capturar,
            name="hilo_camara",
            daemon=True
        )
        self._hilo.start()
        print("[CAMERA] Hilo de captura iniciado.")

        # Warm-up: esperar hasta tener al menos 1 línea capturada
        tiempo_espera = 0.0
        while self._lineas_capturadas == 0 and tiempo_espera < 3.0:
            time.sleep(0.1)
            tiempo_espera += 0.1

        if self._lineas_capturadas == 0:
            raise RuntimeError(
                "[CAMERA] No se recibieron líneas en 3 segundos. "
                "Revisar exposición y conexión."
            )
        print("[CAMERA] Primera línea recibida. Sistema listo.")

    def _configurar_exposicion(self) -> None:
        """
        Desactiva AutoExposure y fija el tiempo de exposición.

        ADAPTACIÓN SDK: si el feature 'ExposureTime' tiene nombre diferente
        en tu versión del SDK, ajustar aquí. Usar Galaxy Viewer para
        verificar los nombres exactos de features de tu cámara.
        """
        remote = self._cam.get_remote_device_feature_control()

        if (remote.is_implemented("ExposureAuto") and
                remote.is_writable("ExposureAuto")):
            remote.get_enum_feature("ExposureAuto").set("Off")
            print("[CAMERA] ExposureAuto desactivado.")

        if (remote.is_implemented("ExposureTime") and
                remote.is_writable("ExposureTime")):
            remote.get_float_feature("ExposureTime").set(
                float(config.CAMERA_EXPOSURE_US)
            )
            print(f"[CAMERA] ExposureTime = {config.CAMERA_EXPOSURE_US} µs")
        else:
            print("[CAMERA] ADVERTENCIA: ExposureTime no es escribible.")

        # Activar stream antes de capturar
        self._cam.stream_on()
        time.sleep(0.3)  # Estabilización del sensor

    # ------------------------------------------------------------------
    # HILO DE CAPTURA
    # ------------------------------------------------------------------

    def _capturar(self) -> None:
        """
        Loop principal del hilo de captura.
        Captura líneas crudas y las reduce a (Y_util, N_BANDS).

        NOTA SOBRE EL SHAPE:
        La cámara entrega cada llamada a get_image() como una imagen
        (Y=1024, X=1280) donde X es el eje espectral (bandas).
        Esto es un frame pushbroom de una sola línea espacial.
        El eje temporal (movimiento de banda) se construye en buffer.py
        apilando estas líneas.
        """
        while self._running:
            try:
                # ── 1. Captura cruda ──────────────────────────────────
                # ADAPTACIÓN SDK: get_image() puede llamarse de formas
                # distintas según versión. Si retorna None frecuentemente,
                # revisar timeout del stream o modo de trigger.
                raw = self._cam.data_stream[0].get_image()

                if raw is None:
                    self._lineas_perdidas += 1
                    if config.LOG_LEVEL >= 2:
                        print(f"[CAMERA] Frame None (perdidas: "
                              f"{self._lineas_perdidas})")
                    continue

                linea_cruda = raw.get_numpy_array()
                if linea_cruda is None:
                    self._lineas_perdidas += 1
                    continue

                # linea_cruda.shape esperado: (1024, 1280) uint8
                # Si el shape es distinto, la cámara puede estar en
                # modo diferente. Verificar con Galaxy Viewer.
                if linea_cruda.shape != (config.CAMERA_LINE_HEIGHT,
                                         config.CAMERA_N_BANDS):
                    if config.LOG_LEVEL >= 1:
                        print(f"[CAMERA] Shape inesperado: {linea_cruda.shape} "
                              f"(esperado: ({config.CAMERA_LINE_HEIGHT}, "
                              f"{config.CAMERA_N_BANDS}))")
                    self._lineas_perdidas += 1
                    continue

                # ── 2. Extraer bandas de interés y región útil ────────
                linea_reducida = self._extraer_bandas(linea_cruda)

                # ── 3. Encolar ────────────────────────────────────────
                try:
                    self._cola.put_nowait(linea_reducida)
                    self._lineas_capturadas += 1
                except queue.Full:
                    # El procesamiento no consume suficientemente rápido.
                    # Descartar la línea más antigua para no acumular latencia.
                    try:
                        self._cola.get_nowait()
                    except queue.Empty:
                        pass
                    self._cola.put_nowait(linea_reducida)
                    self._lineas_capturadas += 1
                    if config.LOG_LEVEL >= 2:
                        print("[CAMERA] Cola llena — línea antigua descartada.")

            except Exception as e:
                if self._running:
                    print(f"[CAMERA] Error en captura: {e}")
                    time.sleep(0.01)

    def _extraer_bandas(self, linea_cruda: np.ndarray) -> np.ndarray:
        """
        Extrae las bandas espectrales de interés y la región útil Y.

        Args:
            linea_cruda: array (1024, 1280) uint8

        Returns:
            array (Y_util, N_BANDS) float32 SIN NORMALIZAR

        DECISIÓN DE DISEÑO — por qué NO se normaliza aquí:
        El preprocesamiento Vy (normalización por canal p1/p99.5 + gamma 1.440)
        vive exclusivamente en preprocessing.py y se aplica una sola vez:
          - Para YOLO/display: en pseudo_rgb.py via preprocess_channels()
          - Para CNN: en main.py (worker_clasificador) via preprocess_channels()

        Aplicar normalización aquí (sobre 824 píxeles de una sola columna
        con rango típico de 4-6 DN) produciría percentiles inestables y
        reescalado dinámico cuando aparece la mandarina.
        """
        # Recortar región útil Y (eliminar bordes con ruido óptico)
        linea_util = linea_cruda[self._y_start:self._y_end, :]
        # linea_util.shape: (Y_util, 1280)

        Y_util   = linea_util.shape[0]
        n_bandas = len(self._bandas)

        # Extraer solo las bandas de interés — SIN normalización
        # El buffer acumula valores uint8 crudos del sensor
        resultado = np.empty((Y_util, n_bandas), dtype=np.float32)
        for i, idx_banda in enumerate(self._bandas):
            resultado[:, i] = linea_util[:, idx_banda].astype(np.float32)

        return resultado

    # ------------------------------------------------------------------
    # CALIBRACIÓN DE BACKGROUND
    # ------------------------------------------------------------------

    def calibrar_background(self, n_frames: int = 100) -> np.ndarray:
        """
        Captura N frames crudos con la banda vacía y calcula el perfil
        de background espectral (1280,) promediando sobre la región
        Y[:50] de todos los frames.

        Debe llamarse ANTES de iniciar el buffer, con la banda vacía
        y sin mandarinas.

        Args:
            n_frames: número de frames a promediar (default: 100)

        Returns:
            background: array (1280,) float32 — perfil espectral del fondo

        EQUIVALENCIA OFFLINE/ONLINE:
            Offline:  cube[:50, :, -100:].mean(axis=(0, 2)) → (1280,)
            Online:   stack(N, 50, 1280).mean(axis=(0, 1)) → (1280,)
            Ambos promedian sobre región Y[:50] y múltiples instantes T.
        """
        print(f"[CAL] Capturando {n_frames} frames para calibración de background...")
        esquinas = []

        capturados = 0
        t0 = time.time()
        while capturados < n_frames:
            if time.time() - t0 > 10.0:
                print(f"[CAL] Timeout — solo se capturaron {capturados} frames.")
                break
            try:
                raw = self._cam.data_stream[0].get_image()
                if raw is None:
                    continue
                img = raw.get_numpy_array()
                if img is None or img.shape != (config.CAMERA_LINE_HEIGHT,
                                                config.CAMERA_N_BANDS):
                    continue
                # Extraer región Y[:50] — esquina superior
                esquina = img[:50, :].astype(np.float32)  # (50, 1280)
                esquinas.append(esquina)
                capturados += 1
            except Exception as e:
                print(f"[CAL] Error capturando frame: {e}")
                continue

        if not esquinas:
            raise RuntimeError("[CAL] No se pudieron capturar frames para background.")

        # Promediar sobre frames y filas Y → (1280,)
        stack = np.stack(esquinas, axis=0)      # (N, 50, 1280)
        background = stack.mean(axis=(0, 1))    # (1280,)

        # Estadísticas del perfil calculado
        bg_bandas = background[config.SPECTRAL_BANDS]
        print(f"[CAL] Background calibrado con {len(esquinas)} frames:")
        print(f"[CAL]   mean={background.mean():.2f} "
              f"std={background.std():.2f} "
              f"max={background.max():.2f}")
        print(f"[CAL]   Bandas de interés — "
              f"mean={bg_bandas.mean():.2f} "
              f"max={bg_bandas.max():.2f}")

        return background

    # ------------------------------------------------------------------
    # API PÚBLICA
    # ------------------------------------------------------------------

    def get_linea(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """
        Retorna la siguiente línea reducida (Y_util, N_BANDS) float32,
        o None si no hay datos en el timeout dado.
        """
        try:
            return self._cola.get(timeout=timeout)
        except queue.Empty:
            return None

    def detener(self) -> None:
        """Detiene el hilo de captura y cierra la cámara correctamente."""
        self._running = False

        if self._hilo and self._hilo.is_alive():
            self._hilo.join(timeout=2.0)

        if self._cam is not None:
            try:
                self._cam.stream_off()
                self._cam.close_device()
                print("[CAMERA] Cámara cerrada correctamente.")
            except Exception as e:
                print(f"[CAMERA] Error al cerrar: {e}")

        elapsed = time.time() - self._t_inicio if self._t_inicio > 0 else 0
        fps_real = self._lineas_capturadas / elapsed if elapsed > 0 else 0
        print(f"[CAMERA] Estadísticas:")
        print(f"  Líneas capturadas : {self._lineas_capturadas}")
        print(f"  Líneas perdidas   : {self._lineas_perdidas}")
        print(f"  FPS promedio      : {fps_real:.1f} líneas/s")

    @property
    def lineas_capturadas(self) -> int:
        return self._lineas_capturadas


# =============================================================================
# IMPLEMENTACIÓN SIMULADA — reproduce muestras reales del Holdout
# =============================================================================

class CamaraSimulada(CamaraBase):
    """
    Simula la cámara pushbroom emitiendo líneas reales del Holdout del dataset.

    Cada muestra .npy (H, W, 5) float32 se emite columna por columna,
    replicando el comportamiento temporal de la cámara física.
    Entre mandarinas se intercalan líneas de fondo extraídas del mismo dataset.

    Contrato de interfaz:
        get_linea() → (Y_util, N_BANDS) = (824, 5) float32 crudo
        Idéntico en shape, dtype y semántica a CamaraGXIpy.get_linea().
    """

    def __init__(self):
        self._y_util  = config.BUFFER_Y_END - config.BUFFER_Y_START  # 824
        self._n_bandas = config.N_BANDS                               # 5
        self._fps     = config.SIM_LINES_PER_SECOND
        self._periodo = 1.0 / self._fps

        self._cola: queue.Queue = queue.Queue(maxsize=500)
        self._running: bool = False
        self._hilo: Optional[threading.Thread] = None
        self._lineas_emitidas: int = 0

        # Muestras y banco de fondo se cargan en _cargar_holdout()
        self._muestras: list = []        # list of dict {stack, clase, nombre}
        self._fondo_pool: list = []      # list of (824, 5) float32
        self._rng = None

    # ------------------------------------------------------------------
    # CARGA DEL HOLDOUT
    # ------------------------------------------------------------------

    def _cargar_holdout(self) -> None:
        """
        Carga todas las muestras del holdout en memoria.
        Lee splits_actual.json, filtra holdout, carga .npy en RAM.
        Construye el banco de fondo mediante percentil relativo.
        """
        import json
        from pathlib import Path

        dataset_path = Path(config.SIM_DATASET_PATH)
        splits_path  = dataset_path / config.SIM_SPLITS_JSON

        if not splits_path.exists():
            raise FileNotFoundError(
                f"[SIM] No se encontró: {splits_path}\n"
                f"Verifica SIM_DATASET_PATH y SIM_SPLITS_JSON en config.py"
            )

        with open(splits_path, encoding='utf-8') as f:
            splits = json.load(f)

        holdout_ids = set(splits["holdout"]["samples"])
        if not holdout_ids:
            raise ValueError("[SIM] El holdout en splits_actual.json está vacío.")

        print(f"[SIM] Cargando {len(holdout_ids)} muestras del holdout...")

        muestras_cargadas = []
        fondo_columnas   = []

        for clase in ("good", "bad"):
            clase_path = dataset_path / "originales" / clase
            if not clase_path.exists():
                continue
            for npy_path in sorted(clase_path.glob("*_stack.npy")):
                sample_id = npy_path.stem.replace("_stack", "")
                if sample_id not in holdout_ids:
                    continue

                # Modo debug: repetir una sola muestra
                if (config.SIM_REPEAT_SAMPLE is not None and
                        npy_path.name != config.SIM_REPEAT_SAMPLE):
                    continue

                stack = np.load(str(npy_path)).astype(np.float32)
                # stack.shape: (H, W, 5) — solo lectura, sin modificar

                muestras_cargadas.append({
                    "stack" : stack,
                    "clase" : clase,
                    "nombre": npy_path.name,
                    "H"     : stack.shape[0],
                    "W"     : stack.shape[1],
                })

                # Extraer columnas de fondo por percentil relativo
                energia = stack.mean(axis=(0, 2))  # (W,) energía por columna
                umbral  = np.percentile(energia, config.SIM_FONDO_PERCENTIL)
                for w in range(stack.shape[1]):
                    if energia[w] < umbral:
                        fondo_columnas.append(stack[:, w, :])  # (H, 5)

        print("\n================ DEBUG HOLDOUT ================")#debug 09_07_26
        print(f"IDs en holdout: {len(holdout_ids)}")
        print(f"Muestras cargadas: {len(muestras_cargadas)}")

        print("\nPrimeros IDs del JSON:")
        for x in sorted(holdout_ids)[:10]:
            print("  ", repr(x))

        print("\nPrimeros archivos GOOD:")
        clase_path = dataset_path / "originales" / "good"
        for p in sorted(clase_path.glob("*_stack.npy"))[:10]:
            sid = p.stem.replace("_stack", "")
            print("  ", repr(sid))

        print("\nPrimeros archivos BAD:")
        clase_path = dataset_path / "originales" / "bad"
        for p in sorted(clase_path.glob("*_stack.npy"))[:10]:
            sid = p.stem.replace("_stack", "")
            print("  ", repr(sid))

        print("==============================================\n")
                
        
        
        
        
        if not muestras_cargadas:
            raise ValueError(
                "[SIM] No se encontraron muestras del holdout en el dataset. "
                "Verifica SIM_DATASET_PATH y que splits_actual.json tenga holdout."
            )

        if not fondo_columnas:
            # Fallback: usar ruido bajo como fondo
            print("[SIM] ADVERTENCIA: no se encontraron columnas de fondo. "
                  "Usando ruido bajo como fondo.")
            fondo_columnas = [
                np.random.uniform(0, 3, (self._y_util, self._n_bandas)
                                  ).astype(np.float32)
                for _ in range(50)
            ]

        self._muestras   = muestras_cargadas
        self._fondo_pool = fondo_columnas

        if config.SIM_RANDOM_ORDER:
            self._rng = np.random.default_rng(config.SIM_RANDOM_SEED)
            self._rng.shuffle(self._muestras)

        print(f"[SIM] {len(self._muestras)} muestras cargadas | "
              f"{len(self._fondo_pool)} columnas de fondo en pool")

    # ------------------------------------------------------------------
    # CONSTRUCCIÓN DE LÍNEA
    # ------------------------------------------------------------------

    def _construir_linea_fondo(self) -> np.ndarray:
        """
        Construye una línea (824, 5) de fondo mediante muestreo aleatorio
        del pool. Los 824 píxeles se samplear independientemente del pool
        para evitar patrones periódicos.
        """
        pool_size   = len(self._fondo_pool)
        linea       = np.zeros((self._y_util, self._n_bandas), dtype=np.float32)
        indices_col = np.random.randint(0, pool_size, size=self._y_util)

        for y_dst in range(self._y_util):
            col = self._fondo_pool[indices_col[y_dst]]  # (H_src, 5)
            # Tomar un pixel aleatorio de esa columna de fondo
            y_src = np.random.randint(0, col.shape[0])
            linea[y_dst, :] = col[y_src, :]

        return linea

    def _construir_linea_con_frutas(
        self,
        frutas: list    # list of (col_fruta, y_pos) tuples
    ) -> np.ndarray:
        """
        Construye una línea (824, 5) insertando una o más frutas sobre fondo.
        Cada fruta se inserta en su y_pos correspondiente.
        """
        linea = self._construir_linea_fondo()
        for col_fruta, y_pos in frutas:
            H = col_fruta.shape[0]
            y_end = min(y_pos + H, self._y_util)
            h_real = y_end - y_pos
            linea[y_pos:y_end, :] = col_fruta[:h_real, :]
        return linea

    # ------------------------------------------------------------------
    # HILO DE GENERACIÓN
    # ------------------------------------------------------------------

    def _generar(self) -> None:
        """
        Emite líneas del holdout a la cadencia real.
        Soporta modo dual: dos frutas activas simultáneamente, cada una
        en su propia mitad vertical del buffer (carril 1 y carril 2).
        """
        idx_muestra  = 0
        n_muestras   = len(self._muestras)
        contador_dual = 0  # cuenta mandarinas para decidir cuándo generar par

        while self._running:
            # ── Decidir si esta iteración es dual ─────────────────────
            es_dual = (config.SIM_DUAL_MODE and
                       contador_dual % config.SIM_DUAL_EVERY == 0 and
                       n_muestras >= 2)

            if es_dual:
                # Seleccionar dos muestras consecutivas del holdout
                m_a = self._muestras[idx_muestra % n_muestras]
                m_b = self._muestras[(idx_muestra + 1) % n_muestras]
                idx_muestra += 2

                stack_a = m_a["stack"]  # (H_a, W_a, 5)
                stack_b = m_b["stack"]  # (H_b, W_b, 5)
                H_a, W_a, _ = stack_a.shape
                H_b, W_b, _ = stack_b.shape

                # Posiciones verticales: A en mitad superior, B en mitad inferior
                mitad = self._y_util // 2
                margin = config.SIM_Y_MARGIN

                # Fruta A — carril superior (0 a mitad)
                y_max_a = mitad - H_a - margin
                if y_max_a < margin:
                    y_pos_a = max(0, (mitad - H_a) // 2)
                else:
                    y_pos_a = int(np.random.randint(margin, y_max_a + 1))

                # Fruta B — carril inferior (mitad a 824)
                y_min_b = mitad + margin
                y_max_b = self._y_util - H_b - margin
                if y_max_b < y_min_b:
                    y_pos_b = max(mitad, self._y_util - H_b)
                else:
                    y_pos_b = int(np.random.randint(y_min_b, y_max_b + 1))

                # Ancho máximo (la más ancha determina cuántas columnas emitir)
                offset = config.SIM_DUAL_OFFSET_LINES
                W_max = max(W_a + offset, W_b)

                # Separación de fondo antes del par
                n_sep = np.random.randint(config.SIM_LINES_MIN,
                                          config.SIM_LINES_MAX + 1)

                if config.SIM_OVERLAY_INFO:
                    print(f"[SIM] ═══ PAR DUAL ═══")
                    print(f"[SIM]   A: {m_a['clase'].upper()} | "
                          f"{m_a['nombre']} | W={W_a} | Y={y_pos_a}")
                    print(f"[SIM]   B: {m_b['clase'].upper()} | "
                          f"{m_b['nombre']} | W={W_b} | Y={y_pos_b}")
                    print(f"[SIM]   Offset: {offset} líneas | Sep: {n_sep}")

                # Emitir fondo
                for _ in range(n_sep):
                    if not self._running:
                        return
                    self._emitir(self._construir_linea_fondo())

                # Emitir ambas frutas columna por columna
                for w in range(W_max):
                    if not self._running:
                        return
                    frutas = []
                    # Fruta A (con offset aplicado)
                    w_a = w - offset
                    if 0 <= w_a < W_a:
                        frutas.append((stack_a[:, w_a, :], y_pos_a))
                    # Fruta B (sin offset)
                    if 0 <= w < W_b:
                        frutas.append((stack_b[:, w, :], y_pos_b))

                    if frutas:
                        linea = self._construir_linea_con_frutas(frutas)
                    else:
                        linea = self._construir_linea_fondo()
                    self._emitir(linea)

                contador_dual += 1

            else:
                # ── Modo normal: una sola fruta ───────────────────────
                muestra = self._muestras[idx_muestra % n_muestras]
                stack   = muestra["stack"]
                H, W, _ = stack.shape

                # Posición vertical aleatoria
                y_min = config.SIM_Y_MARGIN
                y_max = self._y_util - H - config.SIM_Y_MARGIN
                if y_max < y_min:
                    y_min = 0
                    y_max = max(0, self._y_util - H)
                y_pos = int(np.random.randint(y_min, max(y_min + 1, y_max + 1)))

                # Separación de fondo
                n_sep = np.random.randint(config.SIM_LINES_MIN,
                                          config.SIM_LINES_MAX + 1)

                if config.SIM_OVERLAY_INFO:
                    print(f"[SIM] Muestra {idx_muestra % n_muestras + 1}/{n_muestras} | "
                          f"Clase: {muestra['clase'].upper()} | "
                          f"Archivo: {muestra['nombre']} | "
                          f"Columnas: {W} | Separación: {n_sep} líneas | "
                          f"Y_pos: {y_pos}")

                # Emitir fondo
                for _ in range(n_sep):
                    if not self._running:
                        return
                    self._emitir(self._construir_linea_fondo())

                # Emitir muestra columna por columna
                for w in range(W):
                    if not self._running:
                        return
                    col = stack[:, w, :]
                    linea = self._construir_linea_con_frutas([(col, y_pos)])
                    self._emitir(linea)

                idx_muestra += 1
                contador_dual += 1

            # Reinicio con shuffle al agotar el holdout
            if idx_muestra >= n_muestras:
                idx_muestra = 0
                if config.SIM_RANDOM_ORDER and self._rng is not None:
                    self._rng.shuffle(self._muestras)
                    print("[SIM] Holdout agotado — reiniciando con nuevo orden.")

    def _emitir(self, linea: np.ndarray) -> None:
        """Pone la línea en la cola respetando la cadencia."""
        t0 = time.time()
        try:
            self._cola.put_nowait(linea)
        except queue.Full:
            try:
                self._cola.get_nowait()
            except queue.Empty:
                pass
            self._cola.put_nowait(linea)

        self._lineas_emitidas += 1
        elapsed  = time.time() - t0
        sleep_t  = self._periodo - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    # ------------------------------------------------------------------
    # INTERFAZ PÚBLICA
    # ------------------------------------------------------------------

    def conectar(self) -> None:
        print("[CAMERA SIM] conectar() — cargando holdout en memoria...")
        self._cargar_holdout()
        print("[CAMERA SIM] Holdout cargado. Listo para iniciar.")

    def iniciar(self) -> None:
        self._running = True
        self._hilo = threading.Thread(
            target=self._generar,
            name="hilo_camara_sim",
            daemon=True
        )
        self._hilo.start()
        time.sleep(0.2)
        print(f"[CAMERA SIM] Iniciada | {self._fps} líneas/s | "
              f"Shape: ({self._y_util}, {self._n_bandas}) | "
              f"Muestras holdout: {len(self._muestras)}")

    def calibrar_background(self, n_frames: int = 100) -> np.ndarray:
        """Modo simulado — background = zeros, corrección sin efecto."""
        print("[CAL SIM] Modo simulado — background = zeros.")
        return np.zeros(config.CAMERA_N_BANDS, dtype=np.float32)

    def get_linea(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        try:
            return self._cola.get(timeout=timeout)
        except queue.Empty:
            return None

    def detener(self) -> None:
        self._running = False
        if self._hilo and self._hilo.is_alive():
            self._hilo.join(timeout=2.0)
        print(f"[CAMERA SIM] Detenida. {self._lineas_emitidas} líneas emitidas.")


# =============================================================================
# FACTORY — selección automática de implementación
# =============================================================================

def crear_camara(forzar_simulada: bool = False) -> CamaraBase:
    """
    Retorna la implementación de cámara apropiada.

    Args:
        forzar_simulada: si True, siempre usa la simulada aunque haya
                         hardware disponible. Útil para CI/CD y pruebas.

    Returns:
        Instancia de CamaraGXIpy o CamaraSimulada lista para llamar .iniciar()
    """
    if forzar_simulada or not GXIPY_DISPONIBLE:
        print("[CAMERA] Usando CamaraSimulada.")
        return CamaraSimulada()
    else:
        print("[CAMERA] Usando CamaraGXIpy (hardware real).")
        return CamaraGXIpy()
