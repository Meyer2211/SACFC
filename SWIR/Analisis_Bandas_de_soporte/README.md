# Analisis_Bandas_de_soporte

## Objetivo del análisis espectral

Esta carpeta contiene el análisis espectral offline utilizado para determinar qué canales (bandas) de la cámara hiperespectral SWIR aportan mayor capacidad de separar las clases de calidad de la fruta. El resultado de este análisis es el conjunto de offsets de canal respecto a la banda pico (`OFFSETS_STACK`) que el sistema `Pushbroom/` utiliza en producción.

## Scripts principales

### `extraer_features_espectrales.py`

Calcula estadísticas espectrales por cubo hiperespectral, restringidas a los píxeles de la máscara segmentada de la fruta (no a un ROI rectangular ni al cubo completo), evaluando un barrido de offsets candidatos respecto a la banda pico.

### `analizar_separabilidad.py`

Toma las estadísticas generadas en el paso anterior y ordena (rankea) los offsets candidatos según su capacidad de separar las clases de calidad, produciendo una recomendación de los offsets a utilizar.

## Fuentes de verdad (JSON de entrada)

El análisis depende de dos archivos JSON que deben mantenerse consistentes entre sí:

- **`reporte_picos.json`** — fuente primaria: identifica cada cubo (`cube_id`), su banda pico y su ruta.
- **`cube_index.json`** — fuente secundaria: mapea cada `cube_id` a su lote (`lot_id`).

## Resultados generados

Los scripts producen sus salidas en la subcarpeta `analisis_bandas/`, en forma de archivos JSON (con las estadísticas espectrales extraídas y el ranking/recomendación de offsets) y una visualización de apoyo del análisis de separabilidad.

## Relación con Pushbroom

Los offsets recomendados por este análisis son los que quedan congelados en `OFFSETS_STACK` dentro de la configuración del sistema ejecutable (`SWIR/Pushbroom/`). Esta carpeta no es consultada ni ejecutada por `Pushbroom/` en tiempo de ejecución: su resultado se traslada manualmente a la configuración del sistema. Cambiar estos offsets implica reentrenar los modelos de detección y clasificación que dependen de ellos.

## Carácter del análisis

Todo el contenido de esta carpeta es **offline**: se ejecuta de forma independiente y previa, como insumo para fijar parámetros de configuración. No forma parte del pipeline en tiempo real de captura, detección, clasificación y actuación.
