# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This is an undergraduate thesis ("Trabajo de grado") project: an automated real-time mandarin/tangerine
quality classification system for a conveyor-belt sorting line. It combines computer vision (YOLO
detection), a spectral/CNN quality classifier, and Arduino-driven servo actuators that divert fruit
into "good"/"bad" lanes.

The repository is **not a git repository** at the top level (`SWIR/Pushbroom/yolov5/` is a vendored
clone that does have its own `.git`). Don't assume `git status`/`git log` work from the repo root.

The repo contains two parallel hardware approaches developed at different stages of the thesis:

- **`SWIR/`** — the current, actively developed system. Uses a SWIR (short-wave infrared) hyperspectral
  **pushbroom** line-scan camera. This is where almost all active work happens.
- **`RGB/`** — an earlier prototype using a standard RGB camera. Mostly exploratory/archived scripts,
  bibliography, and posters from the initial phase of the project. Treat scripts under
  `RGB/Codigos_de_python_merge/Yolo_processing/` (`merge.py`, `merge_V2.py`, `merge_Y_R_C.py`,
  `C_final.py`, `C_final_V2.py`, ...) as sequential experimental snapshots, not a maintained module —
  don't assume the latest-named file is authoritative without checking its README.
- **`Impresiones_3D/`** — STL files and vendor zips for 3D-printed mechanical parts (door/actuator
  mechanisms). Not code.

## The active system: `SWIR/Pushbroom/`

This is the only subsystem with a real dependency list and a runnable entry point.

### Running it

```bash
cd SWIR/Pushbroom
pip install -r requirements.txt
# gxipy (Daheng Imaging camera SDK) is NOT on PyPI — install manually from the Galaxy SDK.

python main.py          # real SWIR camera (requires gxipy + hardware + Arduino on ARDUINO_PORT)
python main.py --sim    # simulated camera — replays real holdout-set samples, no hardware needed
```

There is no test suite, linter, or build step in this repo. Verify changes by running
`python main.py --sim` and observing console logs plus the `cv2` visualization window (press `q` to
exit). `--sim` is the practical way to exercise the full pipeline (camera → buffer → YOLO → tracker →
classifier → actuator scheduler) without hardware.

Two things to be aware of before poking around this directory:
- `venv_mandarina/` is a **committed virtualenv** — never edit files inside it.
- `yolov5/` is a **vendored copy of the Ultralytics YOLOv5 repo** (has its own `.git`) — treat as
  third-party; only touch it if you're intentionally updating the detector dependency.

### Configuration: `config.py` is the single source of truth

Every tunable parameter (camera settings, spectral bands, mechanical timing, thresholds, classifier
selection, window layout) lives in `config.py`, grouped into numbered sections, with an `assert`-based
self-validation block at the bottom. There are no magic numbers scattered elsewhere in the codebase —
when changing behavior, look here first.

Two parameters matter most operationally:
- **`SPECTRAL_PEAK_BAND`** — must be re-measured and updated at the start of every lab session (the
  spectral peak shifts with lighting/batch). The README documents the intended workflow via a
  `detectar_pico.py` peak-detection script; that script does not currently exist in this checkout, so
  don't assume it's runnable without checking first.
- **`OFFSETS_STACK`** (`[+11, +2, -5, -10, -3]`) — the 5 spectral channel offsets relative to the peak.
  These are frozen and informed by the offline analysis in `SWIR/Analisis_Bandas_de_soporte/`; changing
  them requires retraining YOLO and the classifiers.

`config.py` also hardcodes absolute Windows paths for the simulator dataset
(`SIM_DATASET_PATH`, `SIM_SPLITS_JSON`) that point outside this repo — these are machine-specific and
will need adjusting on another machine.

Model files selected via `CLASSIFIER_TYPE`/`CLASSIFIER_MODEL_PATH` in `config.py`:
`best.pt` (YOLOv5 weights), `experimento5.h5` / `expE_final_best.h5` / `fusion_C.h5` (Keras CNN
classifiers, "Experimento A" = 5 spectral channels vs "Experimento B" = 3 pseudo-RGB channels), and
`rf_glcm_5ch.pkl` (sklearn RandomForest + GLCM texture-feature pipeline).

### Pipeline architecture

Multi-threaded producer/consumer, orchestrated by `main.py`:

```
Camera thread    → cola_camara     → Main loop
Main loop        → RollingBuffer   → get_frame() → pseudo_rgb.py → YOLO → Tracker
Tracker          → cola_clasif.    → Classifier thread
Classifier thread→ preprocessing.py → classifier → cola_resultados
Main loop        → cola_resultados → Actuator scheduler → Arduino (serial)
Main loop        → fixed canvas    → cv2.imshow
```

Module responsibilities:

- **`camera.py`** — `CamaraBase` interface with two implementations sharing one contract
  (`get_linea()` returns `(Y_util, N_BANDS)` float32 raw, unnormalized): `CamaraGXIpy` (real Daheng
  camera, capture thread + background calibration) and `CamaraSimulada` (replays real holdout `.npy`
  samples at the real line rate, with configurable dual-lane/simultaneous-fruit test mode). Factory:
  `crear_camara()`.
- **`buffer.py`** — `RollingBuffer`: circular buffer of spectral lines forming a 2D pushbroom frame
  where the X axis is *time* (belt motion), not space. It tracks a monotonic global line counter so
  that a YOLO bbox's X-column can be mapped back to the exact raw spectral lines it came from —
  `extraer_roi_espectral()` uses this to pull the un-preprocessed ROI for classification. This
  temporal-sync mapping is the trickiest invariant in the codebase; a bbox's `linea_inicio_frame`
  metadata must come from the *same* `get_frame()` call that produced the pseudo-RGB YOLO saw.
- **`preprocessing.py`** — the single implementation of the "Vy" transform (per-channel percentile
  normalization, then gamma). Called identically from both the YOLO/display path and the CNN path so
  they always see identically-processed data. Do not duplicate or re-apply normalization anywhere else.
- **`pseudo_rgb.py`** — builds the 3-channel pseudo-RGB image from the 5-band frame for YOLO input.
  The exact same image is also what's shown on screen — there is no separate display rendering path.
- **`detector.py`** — YOLOv5 wrapper (loaded via `torch.hub` from the local vendored `yolov5/`), filters
  detections by confidence/area/class and discards boxes too close to the rolling-window's temporal
  edges (partially-captured fruit).
- **`tracker.py`** — multi-object tracker keyed by combined IoU + Y-distance + area-similarity score.
  Assigns a fixed lane (`carril` 1/2) by Y-centroid on first sighting (never reassigned). Each object is
  sent for classification exactly once, when its centroid crosses the buffer's virtual center line.
- **`classifier.py`** — pluggable classifier (`pkl_rf_glcm` / `h5_cnn` / `pt_torch`, the last one is a
  stub) selected entirely via `config.py`. Always receives an already-Vy-preprocessed ROI; never
  re-normalizes.
- **`actuator.py`** — Python owns all timing. `encolar_evento()` schedules a servo fire
  `CENTER_CROSSING_DELAY_S` after the object's line-crossing timestamp via a `PriorityQueue` +
  `time.monotonic()` scheduler thread; only a single-character command (`"L\n"`/`"R\n"`) is sent over
  serial. **Fail-safe design: only "BUENA" (good) classifications are ever enqueued** — bad fruit and
  anything unclassified simply continues to the reject lane by default, and if Arduino loses
  connection, servos just stay at rest.
- **`main.py`** — the orchestrator: runs the capture-drain / YOLO-every-N-lines / tracker / dispatch-to-
  classifier loop, and builds the fixed-size `cv2` window (left panel = live pseudo-RGB with bbox
  overlays, right panel = last classification, session counters, module health, active config dump).
- **`servo_control/servo_control.ino`** — Arduino sketch. Deliberately dumb: a non-blocking per-servo
  state machine that opens on `L`/`R` and closes after a fixed `TIEMPO_ABIERTO_MS`. It holds no queue
  and no cross-command timing — all scheduling intelligence is in `actuator.py`.

### `SWIR/Analisis_Bandas_de_soporte/`

Offline scripts used to choose the spectral channel offsets (`OFFSETS_STACK`) that `config.py` freezes.
`extraer_features_espectrales.py` computes per-cube spectral statistics restricted to segmented
fruit-mask pixels (not a rectangular ROI, not whole-cube) across a sweep of offsets;
`analizar_separabilidad.py` ranks offsets by class separability. Two JSON files act as sources of truth
that must stay consistent: `reporte_picos.json` (primary — cube_id/peak/path) and `cube_index.json`
(secondary — cube_id→lot_id).
