En esta carpeta se encuentran todos los scripts desarrollados en Python para las distintas etapas del sistema.

Los archivos están organizados según el nivel de integración alcanzado en cada fase:

🔹 1. Procesamiento con YOLO
Scripts dedicados únicamente a:

Captura de imagen desde la cámara.

Detección de mandarinas mediante YOLOv5.

Visualización de bounding boxes.

Validación de coordenadas.

🔹 2. YOLO + Red de Clasificación
En esta etapa se integró:

Detección con YOLOv5.

Recorte automático usando las coordenadas detectadas.

Conversión de color (BGR → RGB).

Clasificación de calidad mediante la red en TensorFlow.

Visualización de la etiqueta (Buena / Mala).

🔹 3. YOLO + Red + Arduino
Fase de integración más completa del sistema, donde:

Se detecta la mandarina.

Se clasifica su calidad.

Se envía la decisión al Arduino.

Se ejecuta la acción física mediante servomotores.

Esta etapa representa la base funcional del sistema en tiempo real previo a optimizaciones finales.
