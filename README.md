# Detection TB Web

Sistema de detección automática de tuberculosis (TB) en radiografías de tórax (CXR) utilizando Deep Learning, con segmentación pulmonar y explicabilidad visual mediante Score-CAM.

## Descripción del proyecto
Este proyecto implementa una solución completa de inteligencia artificial para:

### Clasificar radiografías de tórax en:
- 🟩 NORMAL
- 🟥 TB
- Mejorar imágenes médicas mediante técnicas de procesamiento
Segmentar automáticamente la región pulmonar
Generar mapas de activación (heatmaps) para explicar decisiones del modelo
Proveer una interfaz web interactiva usando Gradio

## Arquitectura del sistema
El sistema sigue un pipeline estructurado:
- `Imagen CXR → Enhancement → Segmentación pulmonar → Clasificación → Score-CAM → Visualización`

## Tecnologías utilizadas
- Deep Learning
PyTorch
Torchvision (DenseNet169)
- Procesamiento de imágenes
OpenCV
PIL (Pillow)
NumPy
- Visualización / UI
Gradio
- Modelos implementados
Clasificador: `DenseNet169`
/ Segmentación:
`Attention U-Net`

## Estructura del proyecto
- `app.py` Interfaz web con Gradio
- `train.py` Entrenamiento + pipeline completo
- `classifier_model.py` Carga del modelo DenseNet
- `generate_cam_grid.py` Generación de grillas Score-CAM
- `unet_model.py` Arquitectura U-Net

## Modelos utilizados
1. Clasificador (DenseNet169)
Se carga desde checkpoint con:
`model = models.densenet169(weights=None)`
Adaptado a número de clases dinámico
Usa normalización personalizada (mean, std)
Threshold optimizado durante entrenamiento

2. Segmentación pulmonar
Se implementan dos enfoques:
- ✔ Heurístico
Umbralización + morfología
Fallback automático
- ✔ Deep Learning
U-Net
Attention U-Net (mejor rendimiento)
`NeuralLungSegmentationEnhancer(...)`

3. Explicabilidad (Score-CAM)
Implementación personalizada
Usa mapas de activación del modelo
Filtrado por cuantiles de activación
`cam = ScoreCAM(...)`

## Preprocesamiento (Enhancement)
El sistema incluye múltiples técnicas:

CLAHE (ecualización adaptativa)
Gamma correction
Histogram Equalization
BCET
Unsharp Mask

Ejemplo:
`--enhancement-mode clahe_gamma`

## Cómo ejecutar el proyecto
Instalar dependencias
`pip install torch torchvision gradio opencv-python pillow numpy scikit-learn`

Ejecutar la aplicación web
`python app.py`

## Funcionamiento de la predicción
Pipeline en predict():

Segmentación pulmonar
Transformación de imagen
Clasificación
Generación de Score-CAM
Overlay del mapa sobre la imagen
`resultado = "🟥 TB" if tb_prob >= clf["threshold"] else "🟩 NORMAL"`
