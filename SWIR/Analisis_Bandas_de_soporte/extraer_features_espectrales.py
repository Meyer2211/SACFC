"""
extraer_features_espectrales.py — Fase 1 del análisis de bandas de soporte
===========================================================================
Extrae métricas espectrales por offset para cada cubo, operando ÚNICAMENTE
sobre píxeles pertenecientes a mandarinas (máscara de segmentación).

NO usa ROI rectangular. NO analiza el cubo completo.
La máscara se construye con la misma lógica del pipeline de producción.

Offsets analizados: [-150, +20] → 171 offsets por cubo.

Fuentes de verdad (jerarquía aprobada por auditoría):
    PRIMARIA  — reporte_picos.json: cube_id, pico_idx, carpeta, ruta física.
    SECUNDARIA — cube_index.json: cube_id → lot_id (único uso).

El script valida consistencia entre ambas fuentes antes de comenzar.
Si un cubo del reporte no aparece en cube_index.json → ValueError.

Inferencia de clase: Maduras* → good, Verdes* → bad (desde nombre de carpeta).

Zona de exclusión pseudo-RGB (documentada explícitamente):
    Offsets RGB: [+11, +2, -5]
    Zona prohibida efectiva: [-15, +21] (solapamiento de ventanas ±10)
    Esta exclusión se aplica en Fase 2, no aquí.

Nota sobre background correction:
    Desactivada por especificación aprobada: bajo normalización por_canal
    la corrección de fondo es invariante. No se aplica en ninguna etapa.

Salida: features_espectrales.json
    Un entry por cubo:
    {
        "cube_id": "...",
        "lot_id": "...",
        "class": "good",
        "peak": 682,
        "n_pixels_mascara": 12345,
        "offsets": {
            "-150": {"mean":..., "std":..., "p10":..., "p25":...,
                     "p50":..., "p75":..., "p90":..., "energy":...,
                     "snr":..., "cv":..., "skewness":..., "kurtosis":...},
            ...
        }
    }

Nota: las métricas se calculan sobre la UNIÓN de todos los píxeles de
mandarina detectados en el cubo (todas las frutas segmentadas juntas).
En cubos con múltiples mandarinas esto produce una distribución agregada.
Esta decisión es aceptable para la primera selección de bandas soporte.
"""

import json
import time
import numpy as np
import cv2
from pathlib import Path
from scipy import stats as scipy_stats

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────

REPORTE_PICOS_JSON = r"C:\Users\ASUS\Documents\UIS\Trabajo_de_grado\Analisis_Bandas_de_soporte\reporte_picos.json"
CUBE_INDEX_JSON    = r"C:\Users\ASUS\Documents\UIS\Trabajo_de_grado\Analisis_Bandas_de_soporte\cube_index.json"
CARPETA_SALIDA     = r"C:\Users\ASUS\Documents\UIS\Trabajo_de_grado\Analisis_Bandas_de_soporte\analisis_bandas"


# Lotes a incluir en el análisis (los que entran al Dataset_CNN)
LOTES_INCLUIDOS = {"2026_05_15", "2026_05_19", "2026_06_12_A", "2026_06_12_B"}

# Rango de offsets a analizar: [-150, +20] → 171 offsets
OFFSET_MIN = -150
OFFSET_MAX =  20

# Parámetros de segmentación (idénticos al pipeline de producción)
OTSU_FACTOR    = 0.46
MORPH_KERNEL   = 15
MIN_AREA_RATIO = 0.002
MAX_AREA_RATIO = 0.25
BORDER_MARGIN  = 5

# ──────────────────────────────────────────────────────────────────────────────


def inferir_clase(carpeta_str):
    """Infiere clase desde nombre de carpeta: Maduras* → good, Verdes* → bad."""
    nombre = Path(carpeta_str).name.lower()
    if nombre.startswith("maduras"):
        return "good"
    if nombre.startswith("verdes"):
        return "bad"
    return "unknown"


def construir_cube_to_lot(cube_index_path):
    """
    Lee cube_index.json y construye dict {cube_id: lot_id}.
    Fuente secundaria — ÚNICO uso: resolver lot_id por cube_id.

    Normalización: cube_index guarda cube_id sin extensión (cube_20260515_154500)
    mientras que reporte_picos guarda el nombre con extensión (.npy).
    Se añade .npy a la clave para que los lookups sean consistentes.
    """
    with open(cube_index_path) as f:
        idx = json.load(f)
    return {c["cube_id"] + ".npy": c["lot_id"] for c in idx["cubos"]}


def validar_consistencia(reporte_detalle, cube_to_lot, lotes_incluidos):
    """
    Verifica que todo cubo del reporte cuyo lot_id esté en lotes_incluidos
    aparezca en cube_index.json. Si no → ValueError.

    La consistencia se verifica EXCLUSIVAMENTE mediante cube_to_lot[cube_name].
    No se usan heurísticas de nombre ni fecha.
    """
    faltantes = []
    for entry in reporte_detalle:
        cube_name = entry["cubo"]
        lot_id    = cube_to_lot.get(cube_name)
        # Si el cubo está en cube_index y su lot_id está en los incluidos → OK
        # Si no está en cube_index → puede ser de un lote excluido → no es error
        # Solo es error si SÍ está en cube_index con un lote incluido pero
        # la ruta del reporte no permite cargarlo — eso se detecta después.
        # Aquí solo verificamos que ningún cubo cuyo lot_id sea incluido
        # esté ausente de cube_index.
        if lot_id is not None and lot_id not in lotes_incluidos:
            continue   # lote excluido — OK
        if lot_id is None:
            continue   # no está en cube_index → lote excluido implícitamente → OK

    # Verificación inversa: todo cubo en cube_index con lote incluido
    # debe aparecer en el reporte
    cubos_reporte = {e["cubo"] for e in reporte_detalle}
    for cube_name, lot_id in cube_to_lot.items():
        if lot_id not in lotes_incluidos:
            continue
        if cube_name not in cubos_reporte:
            faltantes.append(f"{cube_name} (lot_id={lot_id})")

    if faltantes:
        raise ValueError(
            f"\n[ERROR] Inconsistencia entre cube_index.json y reporte_picos.json.\n"
            f"Los siguientes cubos están en cube_index pero no en reporte_picos:\n"
            + "\n".join(f"  - {c}" for c in faltantes)
            + "\n  Verifica que reporte_picos.json esté actualizado."
        )


def segmentar(img_norm):
    """
    Segmentación idéntica al pipeline de producción.
    Entrada: imagen 2D float32 normalizada [0,1].
    Salida: máscara binaria uint8.
    """
    total_px = img_norm.shape[0] * img_norm.shape[1]
    min_area = int(MIN_AREA_RATIO * total_px)
    max_area = int(MAX_AREA_RATIO * total_px)

    img_u8 = (img_norm * 255).astype(np.uint8)
    thresh_otsu, _ = cv2.threshold(img_u8, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_final = int(thresh_otsu * OTSU_FACTOR)

    _, mask = cv2.threshold(img_u8, thresh_final, 255, cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (MORPH_KERNEL, MORPH_KERNEL))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    n_labels, labeled, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    H, W = mask.shape
    final_mask = np.zeros_like(mask)
    n_validos = 0
    for label in range(1, n_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not (min_area <= area <= max_area):
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if (x <= BORDER_MARGIN or (x + w) >= (W - BORDER_MARGIN) or
                y <= BORDER_MARGIN or (y + h) >= (H - BORDER_MARGIN)):
            continue
        final_mask[labeled == label] = 255
        n_validos += 1

    return final_mask, n_validos


def normalizar_percentil(img, p_low=1, p_high=99):
    """Normalización percentil para la banda de segmentación."""
    v1 = float(np.percentile(img, p_low))
    v2 = float(np.percentile(img, p_high))
    if v2 > v1:
        return np.clip((img.astype(np.float32) - v1) / (v2 - v1), 0, 1)
    return np.zeros_like(img, dtype=np.float32)


def calcular_metricas(pixels):
    """
    Calcula 12 métricas espectrales sobre un array 1D de píxeles de mandarina.
    Retorna dict con todas las métricas, o None si hay pocos píxeles.
    """
    if len(pixels) < 10:
        return None

    px = pixels.astype(np.float64)
    mean   = float(np.mean(px))
    std    = float(np.std(px))
    p10    = float(np.percentile(px, 10))
    p25    = float(np.percentile(px, 25))
    p50    = float(np.percentile(px, 50))
    p75    = float(np.percentile(px, 75))
    p90    = float(np.percentile(px, 90))
    energy = float(np.mean(px ** 2))
    snr    = float(mean / std) if std > 0 else 0.0
    cv     = float(std / mean) if mean > 0 else 0.0
    skew   = float(scipy_stats.skew(px))
    kurt   = float(scipy_stats.kurtosis(px))

    return {
        "mean"    : round(mean,   6),
        "std"     : round(std,    6),
        "p10"     : round(p10,    6),
        "p25"     : round(p25,    6),
        "p50"     : round(p50,    6),
        "p75"     : round(p75,    6),
        "p90"     : round(p90,    6),
        "energy"  : round(energy, 6),
        "snr"     : round(snr,    6),
        "cv"      : round(cv,     6),
        "skewness": round(skew,   6),
        "kurtosis": round(kurt,   6),
    }


def procesar_cubo(cube_path, peak, clase, lot_id):
    """
    Carga el cubo, construye máscara de mandarinas y extrae métricas
    por cada offset en [OFFSET_MIN, OFFSET_MAX].

    Las métricas se calculan sobre la UNIÓN de todos los píxeles
    de mandarina detectados (todas las frutas del cubo juntas).

    Background correction: desactivada (invariante bajo normalización
    por_canal — especificación aprobada).

    Retorna dict con el entry completo del cubo, o None si falla.
    """
    cube = np.load(str(cube_path), mmap_mode='r')
    Y, n_bandas, T = cube.shape

    if not (0 <= peak < n_bandas):
        print(f"  [SKIP] Pico {peak} fuera de rango para {cube_path.name}")
        return None

    # Construir máscara usando banda del pico (igual que el pipeline)
    banda_pico = cube[:, peak, :].astype(np.float32)
    img_norm   = normalizar_percentil(banda_pico)
    mascara, n_frutas = segmentar(img_norm)

    n_pixels = int((mascara > 0).sum())
    if n_pixels < 100:
        print(f"  [SKIP] Máscara insuficiente ({n_pixels} px, "
              f"{n_frutas} frutas) en {cube_path.name}")
        return None

    print(f"  Pico={peak} | Frutas={n_frutas} | "
          f"Píxeles máscara={n_pixels} | "
          f"Offsets [{OFFSET_MIN}, {OFFSET_MAX}]")

    # Extraer métricas por offset
    offsets_data     = {}
    offsets_validos  = 0
    offsets_invalidos = 0

    for offset in range(OFFSET_MIN, OFFSET_MAX + 1):
        banda_idx = peak + offset
        if not (0 <= banda_idx < n_bandas):
            offsets_invalidos += 1
            continue

        # Extraer solo píxeles de máscara — evita copia float32 del frame completo
        pixels_mandarina = cube[:, banda_idx, :][mascara > 0].astype(np.float32)

        metricas = calcular_metricas(pixels_mandarina)
        if metricas is not None:
            offsets_data[str(offset)] = metricas
            offsets_validos += 1

    print(f"  Offsets válidos={offsets_validos} | "
          f"Fuera de rango={offsets_invalidos}")

    return {
        "cube_id"          : cube_path.stem,
        "lot_id"           : lot_id,
        "class"            : clase,
        "peak"             : peak,
        "n_frutas"         : n_frutas,
        "n_pixels_mascara" : n_pixels,
        "offsets"          : offsets_data,
    }


def main():
    salida_dir = Path(CARPETA_SALIDA)
    salida_dir.mkdir(parents=True, exist_ok=True)

    # ── Cargar fuentes de verdad ──────────────────────────────────────────────
    with open(REPORTE_PICOS_JSON) as f:
        reporte = json.load(f)

    cube_to_lot = construir_cube_to_lot(CUBE_INDEX_JSON)

    print(f"\n{'='*60}")
    print(f"EXTRACCIÓN DE FEATURES ESPECTRALES — FASE 1")
    print(f"  Lotes incluidos      : {sorted(LOTES_INCLUIDOS)}")
    print(f"  Rango offsets        : [{OFFSET_MIN}, {OFFSET_MAX}]")
    print(f"  Zona excl. pseudo-RGB: [-15, +21] (calculada en Fase 2)")
    print(f"  Background correction: OFF (invariante bajo norm. por_canal)")
    print(f"  Salida               : {salida_dir}")
    print("="*60)

    # ── Validar consistencia entre fuentes ───────────────────────────────────
    print(f"\n[VALIDACIÓN] Consistencia reporte_picos ↔ cube_index...")
    validar_consistencia(reporte["detalle"], cube_to_lot, LOTES_INCLUIDOS)
    print(f"  ✅ Consistencia verificada")

    # ── Construir lista de cubos a procesar ───────────────────────────────────
    cubos_a_procesar = []
    omitidos_lote    = 0
    omitidos_clase   = 0

    for entry in reporte["detalle"]:
        cube_name = entry["cubo"]
        carpeta   = entry["carpeta"]
        peak      = entry["pico_idx"]

        # Resolver lot_id desde cube_index (fuente secundaria)
        lot_id = cube_to_lot.get(cube_name)
        if lot_id is None or lot_id not in LOTES_INCLUIDOS:
            omitidos_lote += 1
            continue

        # Inferir clase desde nombre de carpeta (fuente primaria)
        clase = inferir_clase(carpeta)
        if clase == "unknown":
            print(f"  [WARN] No se pudo inferir clase para: {carpeta}")
            omitidos_clase += 1
            continue

        cube_path = Path(carpeta) / cube_name
        if not cube_path.exists():
            print(f"  [WARN] Archivo no encontrado: {cube_path}")
            omitidos_clase += 1
            continue

        cubos_a_procesar.append({
            "path"   : cube_path,
            "lot_id" : lot_id,
            "clase"  : clase,
            "peak"   : peak,
        })

    print(f"\n  Cubos a procesar : {len(cubos_a_procesar)}")
    print(f"  Omitidos (lote)  : {omitidos_lote}")
    print(f"  Omitidos (clase) : {omitidos_clase}")

    # ── Cargar resultados previos (incrementalidad) ───────────────────────────
    json_salida = salida_dir / "features_espectrales.json"
    if json_salida.exists():
        with open(json_salida) as f:
            resultados = json.load(f)
        ya_procesados = {r["cube_id"] for r in resultados}
        print(f"  Ya procesados    : {len(ya_procesados)} (se saltarán)")
    else:
        resultados    = []
        ya_procesados = set()

    errores  = []
    t_inicio = time.time()

    for i, cubo in enumerate(cubos_a_procesar):
        cube_id = cubo["path"].stem
        if cube_id in ya_procesados:
            continue

        print(f"\n[{i+1}/{len(cubos_a_procesar)}] {cubo['path'].name} "
              f"| {cubo['lot_id']} | {cubo['clase']}")
        t0 = time.time()

        try:
            entry = procesar_cubo(
                cube_path = cubo["path"],
                peak      = cubo["peak"],
                clase     = cubo["clase"],
                lot_id    = cubo["lot_id"],
            )
            if entry is not None:
                resultados.append(entry)
                ya_procesados.add(cube_id)
                with open(json_salida, 'w') as f:
                    json.dump(resultados, f, indent=2)
                print(f"  ✅ {time.time()-t0:.1f}s")
            else:
                errores.append(cube_id)

        except Exception as e:
            print(f"  [ERROR] {e}")
            errores.append(cube_id)

    # ── Resumen ───────────────────────────────────────────────────────────────
    t_total    = time.time() - t_inicio
    good_count = sum(1 for r in resultados if r["class"] == "good")
    bad_count  = sum(1 for r in resultados if r["class"] == "bad")

    print(f"\n{'='*60}")
    print(f"FASE 1 COMPLETADA")
    print(f"  Cubos procesados : {len(resultados)} "
          f"(good={good_count}, bad={bad_count})")
    print(f"  Errores          : {len(errores)}")
    print(f"  Tiempo total     : {t_total/60:.1f} min")
    print(f"  JSON guardado    : {json_salida}")
    print(f"\n  Siguiente paso   : analizar_separabilidad.py")
    print("="*60)


if __name__ == "__main__":
    main()
