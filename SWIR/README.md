# SWIR

Punto de entrada al subsistema SWIR del proyecto: la implementación actual y activa del sistema de clasificación automática de mandarinas, basada en una cámara hiperespectral SWIR (infrarrojo de onda corta) de tipo *pushbroom*.

## Propósito

Esta carpeta agrupa todo el trabajo correspondiente al enfoque hiperespectral del proyecto: el sistema ejecutable, los análisis offline que sustentan sus parámetros de configuración, y la evidencia de pruebas realizadas sobre el sistema en banco.

## Estructura

SWIR/ se organiza en tres subcarpetas con roles distintos y complementarios:

### `Pushbroom/`

Contiene el **sistema ejecutable**: la implementación completa del pipeline de captura, detección, clasificación y actuación que corre en tiempo real sobre la banda transportadora. Es el subsistema en desarrollo activo del proyecto. Su arquitectura interna, configuración y modo de ejecución están documentados en `Pushbroom/README.md`.

### `Analisis_Bandas_de_soporte/`

Contiene los scripts de análisis offline utilizados para determinar los parámetros espectrales que `Pushbroom/` usa en producción, en particular los offsets de canal (`OFFSETS_STACK`) respecto a la banda pico. Este análisis se hizo una sola vez, de forma previa e independiente a la ejecución en tiempo real, y sus resultados quedaron congelados en la configuración del sistema ejecutable.

### `Pruebas_del_sistema_02_06_26/`

Contiene evidencia (imágenes) de una sesión de pruebas del sistema realizada en una fecha específica. Corresponde a un registro histórico de una corrida de pruebas, no a código ni a documentación funcional.

## Relación entre las carpetas

`Analisis_Bandas_de_soporte/` alimenta con parámetros a `Pushbroom/` (de forma offline, no en tiempo de ejecución); `Pushbroom/` es el sistema que efectivamente se ejecuta; y `Pruebas_del_sistema_02_06_26/` documenta evidencia de una ejecución puntual de ese sistema. No existe una dependencia en sentido inverso: `Pushbroom/` no depende de `Pruebas_del_sistema_02_06_26/` para funcionar.

Para el detalle del funcionamiento interno del pipeline (módulos, hilos, flujo de datos), consultar `Pushbroom/README.md`.
