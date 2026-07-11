# 📦 Sistema de Clasificación de Mandarinas en Tiempo Real

## Descripción General

Este repositorio contiene el desarrollo completo de un sistema de clasificación automática de mandarinas basado en visión por computador y control embebido.

El sistema integra:

- Detección de objetos mediante YOLO
- Clasificación de calidad con redes neuronales (TensorFlow)
- Control físico mediante Arduino y servomotores
- Procesamiento de imágenes (análisis previo al entrenamiento)

El flujo general del sistema es:

1. Captura de imagen  
2. Detección de mandarinas  
3. Clasificación (buena / mala)  
4. Activación de actuadores en banda transportadora  

---

## 📁 Estructura del Repositorio

### 📂 Bibliografia
Contiene artículos científicos y documentos de referencia utilizados durante el desarrollo del proyecto.

Base teórica para:
- Clasificación de frutas  
- Visión por computador  
- Técnicas de aprendizaje automático  

---

### 📂 Codigos_arduino
Incluye todos los programas desarrollados para el microcontrolador.

Contenido:
- Pruebas individuales de componentes (servos, sensores, etc.)  
- Integración progresiva del sistema  
- Código final del sistema completo  

---

### 📂 Codigos_de_python_merge
Contiene los scripts principales en Python utilizados en el procesamiento de imágenes y detección.

Incluye:
- Pruebas con YOLO (detección de mandarinas)  
- Scripts preliminares antes de la integración total  
- Código utilizado en la etapa de integración del sistema  

---

### 📂 Plan y poster
Documentación académica del proyecto:

- Plan de trabajo de grado  
- Póster de presentación inicial  

---

### 📂 Pruebas_Obtenidas
Contiene imágenes obtenidas durante distintas pruebas realizadas a lo largo del proyecto.

Incluye:
- Imágenes capturadas en pruebas reales  
- Resultados visuales del sistema  
- Evidencia del proceso de desarrollo  

---

### 📂 Codigos_auxiliares
Scripts de apoyo para el procesamiento de datos.

Incluye:
- Preprocesamiento de imágenes RGB  
- Preparación de datos antes del entrenamiento  
- Ajustes necesarios para el entrenamiento de la red neuronal  

---

## ⚙️ Tecnologías Utilizadas

- Python  
- TensorFlow  
- YOLO (detección de objetos)  
- Arduino  
- Procesamiento de imágenes  


---

## 📌 Notas

- Los códigos están organizados según las etapas del desarrollo (pruebas → integración).  
- Las pruebas almacenadas permiten analizar el rendimiento del sistema.  
- El repositorio incluye tanto desarrollo teórico como implementación práctica.  
