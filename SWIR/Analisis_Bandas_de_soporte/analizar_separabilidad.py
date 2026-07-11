"""
analizar_separabilidad.py — Fase 2 del análisis de bandas de soporte
======================================================================
Lee features_espectrales.json (Fase 1) y selecciona los dos offsets
espectrales óptimos como bandas de soporte del stack CNN.

NO abre cubos. Opera exclusivamente sobre el JSON de Fase 1.

Criterios de selección (todos obligatorios, en orden):
    1. Separabilidad good/bad (Cohen's d, Fisher score, AUC univariado).
    2. Estabilidad entre lotes (Cohen's d consistente en cada lote).
    3. No redundancia con pseudo-RGB [+11, +2, -5]:
       zona de exclusión ±10 bandas alrededor de cada uno.
    4. No redundancia entre los dos candidatos finales:
       selección secuencial — el segundo maximiza separabilidad
       sujeto a baja correlación con el primero.

Proceso en tres etapas:
    Etapa 1 — Ranking global por métricas estadísticas baratas.
    Etapa 2 — Análisis de dispersión basado en percentiles sobre Top-20
              finalistas (entropía de distribución, contraste IQR)
              usando el JSON de Fase 1. No es GLCM — no hay matriz de
              coocurrencia ni relaciones espaciales.
    Etapa 3 — Selección secuencial con penalización por redundancia.

Salida:
    ranking_offsets.json     — ranking completo con todas las métricas
    recomendacion_final.json — los dos offsets seleccionados con justificación
    analisis_separabilidad.png — visualización del ranking

Zona de exclusión pseudo-RGB (documentada explícitamente):
    Offsets RGB: [+11, +2, -5]
    Ventanas de exclusión: [-15,+5], [-8,+12], [+1,+21]
    Zona prohibida efectiva: [-15,+21] (solapamiento de las tres ventanas)
"""

import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────

FEATURES_JSON  = r"C:\Users\ASUS\Documents\UIS\Trabajo_de_grado\Analisis_Bandas_de_soporte\analisis_bandas\features_espectrales.json"
CARPETA_SALIDA = r"C:\Users\ASUS\Documents\UIS\Trabajo_de_grado\Analisis_Bandas_de_soporte\analisis_bandas"

# Pseudo-RGB: offsets ya definidos — zona de exclusión ±EXCLUSION_RADIO
PSEUDO_RGB_OFFSETS  = [+11, +2, -5]
EXCLUSION_RADIO     = 10

# Top-N offsets para análisis GLCM
TOP_N_FINALISTAS = 20

# Número de candidatos finales a seleccionar
N_CANDIDATOS_FINALES = 2

# Umbral de correlación para penalizar redundancia entre candidatos
UMBRAL_CORRELACION = 0.85

# Lotes a analizar (para estabilidad por lote)
LOTES_ANALISIS = ["2026_05_15", "2026_05_19", "2026_06_12_A", "2026_06_12_B"]

# ──────────────────────────────────────────────────────────────────────────────


def cohen_d(grupo_a, grupo_b):
    """Cohen's d entre dos grupos 1D."""
    n_a, n_b = len(grupo_a), len(grupo_b)
    if n_a < 2 or n_b < 2:
        return 0.0
    mean_a, mean_b = np.mean(grupo_a), np.mean(grupo_b)
    std_a,  std_b  = np.std(grupo_a, ddof=1), np.std(grupo_b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2)
                         / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return float(abs(mean_a - mean_b) / pooled_std)


def fisher_score(grupo_a, grupo_b):
    """Fisher score: separación entre medias / varianza intra-clase."""
    mean_a, mean_b = np.mean(grupo_a), np.mean(grupo_b)
    var_a,  var_b  = np.var(grupo_a),  np.var(grupo_b)
    denom = var_a + var_b
    if denom == 0:
        return 0.0
    return float((mean_a - mean_b)**2 / denom)


def auc_univariado(grupo_a, grupo_b):
    """
    AUC aproximado por la regla de Wilcoxon-Mann-Whitney.
    Retorna AUC ∈ [0,1]; 0.5 = azar, 1.0 = separación perfecta.
    """
    n_a, n_b = len(grupo_a), len(grupo_b)
    if n_a == 0 or n_b == 0:
        return 0.5
    count = sum(1 for a in grupo_a for b in grupo_b if a > b)
    count += 0.5 * sum(1 for a in grupo_a for b in grupo_b if a == b)
    auc = count / (n_a * n_b)
    return float(max(auc, 1 - auc))   # siempre ≥ 0.5


def zona_exclusion(offset):
    """Retorna True si el offset está en zona de exclusión del pseudo-RGB."""
    for rgb_off in PSEUDO_RGB_OFFSETS:
        if abs(offset - rgb_off) <= EXCLUSION_RADIO:
            return True
    return False


def correlacion_perfiles(perfil_a, perfil_b):
    """Correlación de Pearson entre dos perfiles de métricas."""
    if len(perfil_a) < 2:
        return 0.0
    return float(np.corrcoef(perfil_a, perfil_b)[0, 1])


def main():
    salida_dir = Path(CARPETA_SALIDA)
    salida_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"ANÁLISIS DE SEPARABILIDAD — FASE 2")
    print(f"  Features JSON  : {FEATURES_JSON}")
    print(f"  Pseudo-RGB     : offsets {PSEUDO_RGB_OFFSETS} (excl. ±{EXCLUSION_RADIO})")
    print(f"  Top-N GLCM     : {TOP_N_FINALISTAS}")
    print("="*60)

    with open(FEATURES_JSON) as f:
        datos = json.load(f)

    good_cubos = [d for d in datos if d["class"] == "good"]
    bad_cubos  = [d for d in datos if d["class"] == "bad"]
    print(f"\n  Cubos cargados: {len(datos)} "
          f"(good={len(good_cubos)}, bad={len(bad_cubos)})")

    # Recopilar todos los offsets disponibles
    todos_offsets = set()
    for d in datos:
        todos_offsets.update(int(k) for k in d["offsets"].keys())
    offsets_sorted = sorted(todos_offsets)
    print(f"  Offsets disponibles: {len(offsets_sorted)} "
          f"[{min(offsets_sorted)}, {max(offsets_sorted)}]")

    # ── ETAPA 1: Ranking por métricas estadísticas ────────────────────────────
    print(f"\n[ETAPA 1] Ranking por métricas estadísticas...")

    metricas_clave = ["mean", "std", "p25", "p50", "p75", "energy", "snr",
                      "p10", "p90", "cv", "skewness", "kurtosis"]

    ranking = []

    for offset in offsets_sorted:
        off_str = str(offset)
        excluido = zona_exclusion(offset)

        # Recopilar valores por métrica por clase
        vals_good = defaultdict(list)
        vals_bad  = defaultdict(list)

        for cubo in good_cubos:
            if off_str in cubo["offsets"]:
                for m in metricas_clave:
                    if m in cubo["offsets"][off_str]:
                        vals_good[m].append(cubo["offsets"][off_str][m])

        for cubo in bad_cubos:
            if off_str in cubo["offsets"]:
                for m in metricas_clave:
                    if m in cubo["offsets"][off_str]:
                        vals_bad[m].append(cubo["offsets"][off_str][m])

        if not vals_good["mean"] or not vals_bad["mean"]:
            continue

        # Separabilidad por métrica
        separabilidad_por_metrica = {}
        for m in metricas_clave:
            g = np.array(vals_good[m])
            b = np.array(vals_bad[m])
            if len(g) < 2 or len(b) < 2:
                continue
            separabilidad_por_metrica[m] = {
                "cohen_d"    : cohen_d(g, b),
                "fisher"     : fisher_score(g, b),
                "auc"        : auc_univariado(g.tolist(), b.tolist()),
                "mean_good"  : float(np.mean(g)),
                "mean_bad"   : float(np.mean(b)),
            }

        if not separabilidad_por_metrica:
            continue

        # Score agregado: media de Cohen's d sobre todas las métricas
        cohen_scores = [v["cohen_d"] for v in separabilidad_por_metrica.values()]
        auc_scores   = [v["auc"]     for v in separabilidad_por_metrica.values()]
        score_global = float(np.mean(cohen_scores))

        # Estabilidad por lote: Cohen's d de "mean" en cada lote por separado
        estabilidad_por_lote = {}
        for lot_id in LOTES_ANALISIS:
            g_lot = [d["offsets"][off_str]["mean"]
                     for d in good_cubos
                     if d["lot_id"] == lot_id and off_str in d["offsets"]]
            b_lot = [d["offsets"][off_str]["mean"]
                     for d in bad_cubos
                     if d["lot_id"] == lot_id and off_str in d["offsets"]]
            if len(g_lot) >= 2 and len(b_lot) >= 2:
                estabilidad_por_lote[lot_id] = round(cohen_d(
                    np.array(g_lot), np.array(b_lot)), 4)

        # Varianza entre lotes como penalización
        if len(estabilidad_por_lote) >= 2:
            varianza_lotes = float(np.var(list(estabilidad_por_lote.values())))
        else:
            varianza_lotes = 0.0

        # Score ajustado: score_global penalizado por varianza entre lotes
        score_ajustado = score_global / (1 + varianza_lotes)

        ranking.append({
            "offset"              : offset,
            "excluido_pseudo_rgb" : excluido,
            "score_global"        : round(score_global,   4),
            "score_ajustado"      : round(score_ajustado, 4),
            "auc_medio"           : round(float(np.mean(auc_scores)), 4),
            "varianza_lotes"      : round(varianza_lotes, 4),
            "estabilidad_por_lote": estabilidad_por_lote,
            "n_good"              : len(vals_good["mean"]),
            "n_bad"               : len(vals_bad["mean"]),
            "separabilidad"       : {k: {m2: round(v2, 4) for m2, v2 in v.items()}
                                     for k, v in separabilidad_por_metrica.items()},
        })

    # Ordenar por score ajustado descendente
    ranking.sort(key=lambda x: x["score_ajustado"], reverse=True)

    print(f"  Offsets rankeados: {len(ranking)}")
    print(f"\n  Top 10 (antes de exclusión pseudo-RGB):")
    for i, r in enumerate(ranking[:10]):
        exc = " [EXCLUIDO]" if r["excluido_pseudo_rgb"] else ""
        print(f"    {i+1:2d}. offset={r['offset']:+4d} | "
              f"score_adj={r['score_ajustado']:.4f} | "
              f"auc={r['auc_medio']:.3f} | "
              f"var_lotes={r['varianza_lotes']:.4f}{exc}")

    # Guardar ranking completo
    ranking_path = salida_dir / "ranking_offsets.json"
    with open(ranking_path, 'w') as f:
        json.dump(ranking, f, indent=2)
    print(f"\n  Ranking guardado: {ranking_path}")

    # ── ETAPA 2: Análisis de dispersión basado en percentiles (Top-N) ─────────
    # Se reconstruye la distribución de cada offset a partir de los percentiles
    # almacenados en el JSON de Fase 1 (p50 por cubo) y se calculan métricas
    # de dispersión por clase. NO es GLCM: no hay matriz de coocurrencia
    # ni relaciones espaciales entre píxeles.
    print(f"\n[ETAPA 2] Análisis de dispersión basado en percentiles "
          f"sobre Top-{TOP_N_FINALISTAS}...")

    candidatos_no_excluidos = [r for r in ranking
                                if not r["excluido_pseudo_rgb"]][:TOP_N_FINALISTAS]

    def entropia_histograma(valores, n_bins=32):
        """Entropía de Shannon sobre histograma de valores."""
        hist, _ = np.histogram(valores, bins=n_bins, density=True)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist + 1e-12)))

    def contraste_aproximado(valores):
        """Contraste aproximado: IQR normalizado."""
        p25 = np.percentile(valores, 25)
        p75 = np.percentile(valores, 75)
        rng = np.percentile(valores, 99) - np.percentile(valores, 1)
        return float((p75 - p25) / rng) if rng > 0 else 0.0

    for entry in candidatos_no_excluidos:
        off_str = str(entry["offset"])

        medians_good = [d["offsets"][off_str]["p50"]
                        for d in good_cubos if off_str in d["offsets"]]
        medians_bad  = [d["offsets"][off_str]["p50"]
                        for d in bad_cubos  if off_str in d["offsets"]]

        if medians_good and medians_bad:
            entry["glcm_approx"] = {
                "entropia_good"  : round(entropia_histograma(medians_good), 4),
                "entropia_bad"   : round(entropia_histograma(medians_bad),  4),
                "contraste_good" : round(contraste_aproximado(medians_good), 4),
                "contraste_bad"  : round(contraste_aproximado(medians_bad),  4),
                "diff_entropia"  : round(abs(entropia_histograma(medians_good)
                                            - entropia_histograma(medians_bad)), 4),
            }

    # ── ETAPA 3: Selección secuencial con penalización por redundancia ────────
    print(f"\n[ETAPA 3] Selección secuencial (n={N_CANDIDATOS_FINALES})...")

    # Perfiles discriminativos: vector de Cohen's d por métrica para cada offset
    def perfil_discriminativo(entry):
        return [entry["separabilidad"].get(m, {}).get("cohen_d", 0.0)
                for m in metricas_clave]

    candidatos = candidatos_no_excluidos
    seleccionados = []

    # Primer candidato: mejor score ajustado entre no excluidos
    primero = candidatos[0]
    seleccionados.append(primero)
    perfil_primero = perfil_discriminativo(primero)
    print(f"  Candidato 1: offset={primero['offset']:+d} | "
          f"score_adj={primero['score_ajustado']:.4f}")

    # Segundo candidato: mejor score ajustado con baja correlación respecto
    # al primero Y respecto a los offsets del pseudo-RGB
    mejor_segundo = None
    mejor_score_segundo = -1

    for entry in candidatos[1:]:
        if entry["offset"] == primero["offset"]:
            continue

        perfil_cand = perfil_discriminativo(entry)
        corr_con_primero = correlacion_perfiles(perfil_primero, perfil_cand)

        # Correlación con pseudo-RGB — NOTA METODOLÓGICA:
        # Esta correlación mide similitud de intensidad espectral media entre
        # cubos, NO similitud discriminativa. Es una aproximación conservadora:
        # penaliza candidatos con perfil espectral similar al pseudo-RGB.
        # En una segunda iteración, reemplazar por correlación de perfiles
        # discriminativos (Cohen's d por métrica) para mayor precisión.
        corr_rgb = []
        for rgb_off in PSEUDO_RGB_OFFSETS:
            rgb_str = str(rgb_off)
            perfil_rgb = []
            for cubo_data in datos:
                if rgb_str in cubo_data["offsets"]:
                    perfil_rgb.append(
                        cubo_data["offsets"][rgb_str].get("mean", 0))
            perfil_cand_vals = []
            off_str = str(entry["offset"])
            for cubo_data in datos:
                if off_str in cubo_data["offsets"]:
                    perfil_cand_vals.append(
                        cubo_data["offsets"][off_str].get("mean", 0))
            if len(perfil_rgb) == len(perfil_cand_vals) and len(perfil_rgb) > 1:
                corr_rgb.append(abs(np.corrcoef(
                    perfil_rgb, perfil_cand_vals)[0, 1]))

        max_corr_rgb = max(corr_rgb) if corr_rgb else 0.0

        # Penalizar si alta correlación con el primero o con pseudo-RGB
        penalizacion = max(abs(corr_con_primero), max_corr_rgb)
        score_penalizado = entry["score_ajustado"] * (1 - penalizacion)

        if score_penalizado > mejor_score_segundo:
            mejor_score_segundo = score_penalizado
            mejor_segundo = entry
            mejor_segundo["corr_con_candidato_1"] = round(corr_con_primero, 4)
            mejor_segundo["max_corr_pseudo_rgb"]  = round(max_corr_rgb, 4)
            mejor_segundo["score_penalizado"]      = round(score_penalizado, 4)

    if mejor_segundo:
        seleccionados.append(mejor_segundo)
        print(f"  Candidato 2: offset={mejor_segundo['offset']:+d} | "
              f"score_adj={mejor_segundo['score_ajustado']:.4f} | "
              f"corr_c1={mejor_segundo.get('corr_con_candidato_1', 0):.3f} | "
              f"corr_rgb={mejor_segundo.get('max_corr_pseudo_rgb', 0):.3f}")

    # ── Guardar recomendación final ───────────────────────────────────────────
    canales_finales = PSEUDO_RGB_OFFSETS + [s["offset"] for s in seleccionados]
    canales_finales_str = [f"pico{o:+d}" for o in canales_finales]

    recomendacion = {
        "pseudo_rgb_offsets"     : PSEUDO_RGB_OFFSETS,
        "offsets_soporte"        : [s["offset"] for s in seleccionados],
        "canales_finales_cnn"    : canales_finales,
        "canales_finales_str"    : canales_finales_str,
        "candidatos_seleccionados": seleccionados,
        "criterios_aplicados"    : [
            "Separabilidad good/bad: Cohen's d + Fisher + AUC",
            "Estabilidad entre lotes: Cohen's d por lote, penalización por varianza",
            f"Exclusión zona pseudo-RGB: ±{EXCLUSION_RADIO} de {PSEUDO_RGB_OFFSETS}",
            "Selección secuencial: segundo candidato minimiza redundancia",
        ],
        "nota_background"        : (
            "Background correction desactivada por especificación aprobada: "
            "bajo normalización por_canal es invariante."
        ),
        "nota_revalidacion"      : (
            "Revalidar que fondo OFF sigue siendo adecuado al incorporar "
            "lotes multi-día/multi-sesión futuros."
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    rec_path = salida_dir / "recomendacion_final.json"
    with open(rec_path, 'w') as f:
        json.dump(recomendacion, f, indent=2)

    # ── Visualización ─────────────────────────────────────────────────────────
    offsets_plot  = [r["offset"] for r in ranking]
    scores_plot   = [r["score_ajustado"] for r in ranking]
    colores       = ["red" if r["excluido_pseudo_rgb"] else "steelblue"
                     for r in ranking]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    axes[0].bar(offsets_plot, scores_plot, color=colores, alpha=0.8, width=0.8)
    for s in seleccionados:
        axes[0].axvline(s["offset"], color="gold", linewidth=2.5,
                        label=f"Seleccionado: {s['offset']:+d}")
    for rgb_off in PSEUDO_RGB_OFFSETS:
        axes[0].axvline(rgb_off, color="green", linewidth=1.5,
                        linestyle="--", alpha=0.7)
    axes[0].set_xlabel("Offset respecto al pico")
    axes[0].set_ylabel("Score ajustado (Cohen's d medio × estabilidad)")
    axes[0].set_title("Ranking de offsets — barras rojas = zona exclusión pseudo-RGB")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Estabilidad por lote para los top-20
    top20 = [r for r in ranking if not r["excluido_pseudo_rgb"]][:20]
    off_labels = [f"{r['offset']:+d}" for r in top20]
    for lot_id in LOTES_ANALISIS:
        vals = [r["estabilidad_por_lote"].get(lot_id, 0) for r in top20]
        axes[1].plot(off_labels, vals, marker='o', label=lot_id, alpha=0.8)
    axes[1].set_xlabel("Offset")
    axes[1].set_ylabel("Cohen's d (métrica: mean)")
    axes[1].set_title("Estabilidad entre lotes — Top-20 candidatos")
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig_path = salida_dir / "analisis_separabilidad.png"
    plt.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*60}")
    print(f"RESULTADO FINAL")
    print("="*60)
    print(f"\n  Canales finales CNN:")
    for ch in canales_finales_str:
        print(f"    {ch}")
    print(f"\n  Archivos generados:")
    print(f"    {ranking_path}")
    print(f"    {rec_path}")
    print(f"    {fig_path}")
    print("="*60)


if __name__ == "__main__":
    main()
