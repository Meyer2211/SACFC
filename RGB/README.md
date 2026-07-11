# Sistema de Clasificación de Mandarinas — Prototipo RGB

## Descripción general

Esta carpeta contiene el prototipo inicial del sistema de clasificación automática de mandarinas, construido sobre una cámara RGB estándar (sin componente hiperespectral). Integra detección de objetos con YOLOv5, clasificación de calidad con una red neuronal en TensorFlow, y control de actuadores físicos mediante Arduino.

## Objetivo

Este prototipo tuvo como propósito validar, en una etapa temprana del proyecto, la viabilidad de un sistema de clasificación automática de mandarinas basado en visión por computador, clasificación de calidad y actuación física sobre una banda transportadora. Sirvió como base conceptual y experimental para el desarrollo posterior del sistema.

> **Estado:** este subsistema es una etapa exploratoria y anterior del proyecto. El desarrollo activo actual se encuentra en `SWIR/Pushbroom/`, que corresponde a la evolución posterior del proyecto hacia una cámara hiperespectral SWIR pushbroom. El contenido de esta carpeta no recibe mantenimiento continuo.

## Arquitectura del prototipo

El sistema se compone de tres bloques funcionales:

- **Detección** — localización de mandarinas en la imagen mediante YOLOv5.
- **Clasificación** — evaluación de calidad (buena / mala) mediante una red neuronal convolucional entrenada en TensorFlow, a partir de la región detectada.
- **Actuación** — comunicación con un Arduino que acciona servomotores para desviar la fruta según el resultado de la clasificación.

## Evolución del desarrollo

El desarrollo avanzó en tres fases de integración progresiva:

1. **Procesamiento con YOLO** — captura de imagen y detección de mandarinas, sin clasificación ni actuación.
2. **YOLO + red de clasificación** — se incorpora el recorte automático de la región detectada y la clasificación de calidad.
3. **YOLO + red + Arduino** — integración completa: detección, clasificación y activación física de los servomotores según el resultado. Esta fase constituye la base funcional del sistema en tiempo real, previa a optimizaciones finales.

## Estructura del repositorio

- **`Bibliografia/`** — artículos científicos que sustentan el marco teórico del proyecto (clasificación de frutas, visión por computador, uniformidad de color).
- **`Codigos arduino/`** — implementación del control de actuadores en el microcontrolador.
- **`Codigos_de_python_merge/`** — scripts en Python correspondientes a las fases de integración descritas arriba.
- **`Plan y poster/`** — documentación académica del trabajo de grado (plan de trabajo, informe de registro de tema y póster de presentación).
- **`Pruebas_Obtenidas/`** — evidencia visual de las pruebas realizadas sobre el sistema.
- **`codigos_auxiliares/`** — scripts de preprocesamiento de imágenes utilizados como apoyo antes del entrenamiento de la red.
- **`Red cnn/`** — desarrollo del modelo de red neuronal convolucional de clasificación.

## Tecnologías utilizadas

- Python
- TensorFlow
- YOLOv5
- Arduino

No existe en `RGB/` un archivo de dependencias (`requirements.txt` u otro) ni instrucciones de instalación/ejecución, a diferencia de `SWIR/Pushbroom/`, que sí cuenta con esa documentación. Para ejecutar los scripts de este prototipo es necesario revisar directamente su código fuente.

## Notas

- `RGB/` no representa el estado actual del sistema; para el desarrollo vigente consultar `SWIR/Pushbroom/`.
