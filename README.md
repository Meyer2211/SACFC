# SACFC

## Sistema Automático de Clasificación de Frutos Cítricos

## Descripción general

Este repositorio contiene el desarrollo de un trabajo de grado universitario centrado en un sistema de clasificación automática de mandarinas en tiempo real, orientado a una línea de banda transportadora. El sistema combina visión por computador para la detección y evaluación de calidad de la fruta con actuadores físicos (servomotores controlados por Arduino) para desviarla hacia carriles de fruta "buena" o "mala" según el resultado de la clasificación.

## Objetivo del proyecto

Diseñar e implementar un sistema capaz de detectar mandarinas sobre una banda transportadora, clasificar su calidad de forma automática y accionar físicamente su desvío hacia el carril correspondiente, operando en tiempo real durante el proceso de selección.

## Estructura general del repositorio

```
SACFC/
├── SWIR/
├── RGB/
└── Impresiones_3D/
```

El repositorio agrupa el trabajo en tres áreas principales:

- **`SWIR/`** — Sistema actual del proyecto, basado en una cámara hiperespectral SWIR (infrarrojo de onda corta) de tipo *pushbroom*. Es el subsistema en desarrollo activo y el que concentra la mayor parte del trabajo reciente.
- **`RGB/`** — Prototipo inicial del proyecto, desarrollado con una cámara RGB estándar. Corresponde a una etapa exploratoria y anterior, que sirvió de base conceptual antes de migrar al enfoque hiperespectral.
- **`Impresiones_3D/`** — Archivos de diseño para las piezas mecánicas impresas en 3D utilizadas en los mecanismos de actuación (puertas/desviadores) de la banda transportadora. No contiene código.

## RGB vs. SWIR

Ambas carpetas representan el mismo objetivo de clasificación, pero abordado con tecnologías de captura distintas y en momentos distintos del proyecto:

- **RGB** utiliza una cámara de color convencional y representa la fase exploratoria inicial del trabajo de grado.
- **SWIR** utiliza una cámara hiperespectral pushbroom, que captura información espectral adicional a la visible, y representa la evolución del proyecto hacia una solución más robusta.

**El desarrollo activo del proyecto se encuentra en `SWIR/`.** El contenido de `RGB/` se conserva como referencia histórica del proceso, pero no recibe mantenimiento continuo.

## Impresiones 3D

La carpeta `Impresiones_3D/` reúne los archivos STL y recursos de terceros asociados a las piezas mecánicas del sistema, correspondientes a los mecanismos físicos de desvío de fruta sobre la banda transportadora. No forma parte del código del sistema.

## Estado del proyecto

Este repositorio corresponde a un trabajo de grado en desarrollo. La documentación específica de cada subsistema, su arquitectura interna y sus instrucciones de ejecución se encuentran en los README de las carpetas correspondientes (`SWIR/Pushbroom/README.md` y `RGB/README.md`).

## Documentación

Cada subsistema dispone de su propio README con el detalle de su arquitectura, estructura interna y (cuando aplica) instrucciones de ejecución.
