# Detection TB Web

Aplicacion web para deteccion de Tuberculosis (TB) en radiografias de torax (CXR).
Pipeline principal: `CLAHE+Gamma -> Segmentacion pulmonar -> Clasificacion -> Score-CAM`.

## 1. Que hace proyecto

Proyecto toma imagen CXR, mejora contraste con `CLAHE+Gamma`, enfoca region pulmonar con segmentador U-Net, clasifica `TB` vs `NORMAL` con DenseNet169, luego genera mapa explicativo Score-CAM.

Salida visual:
- Original
- Imagen segmentada pulmon
- Score-CAM superpuesto
- Probabilidad TB + decision final

## 2. Tecnologias usadas

- Python 3.10+
- PyTorch / Torchvision
- OpenCV
- NumPy
- Pillow
- Gradio
- scikit-learn (entrenamiento/evaluacion)

## 3. Flujo tecnico

1. Cargar imagen (single o lote).
2. Segmentacion pulmonar (Attention U-Net, con fallback heuristico si aplica).
3. Clasificacion DenseNet169 sobre imagen segmentada.
4. Explicabilidad con Score-CAM.
5. Render UI con resultados por imagen.

## 4. Estructura de archivos `.py`

### [app.py](/F:/dataset/detection-tb-web/app.py)
Interfaz web Gradio.
- Tab `Imagen unica`.
- Tab `Lote multiples imagenes` para subir varias fotos.
- Soporte drag/drop en ambos tabs.
- Procesamiento automatico al cambiar imagen/archivos.
- Ejecuta pipeline completo de inferencia y muestra galerias + tabla resumen.

### [classifier_model.py](/F:/dataset/detection-tb-web/classifier_model.py)
Carga checkpoint clasificador DenseNet169.
- Reconstruye capa final segun cantidad de clases guardada.
- Devuelve metadata: `img_size`, `threshold`, `mean`, `std`, `tb_index`.

### [train.py](/F:/dataset/detection-tb-web/train.py)
Script principal de entrenamiento y utilidades de inferencia/explicabilidad.
Incluye:
- Segmentacion pulmonar (heuristica, U-Net, Attention U-Net).
- Modelos clasificadores y entrenamiento supervisado.
- Seleccion de threshold (politicas WHO-TPP/strict/balanced).
- Score-CAM y utilidades de visualizacion.
- Generacion de grilla comparativa de CAM.

### [generate_cam_grid.py](/F:/dataset/detection-tb-web/generate_cam_grid.py)
Script para generar imagen comparativa de Score-CAM con varias tecnicas de enhancement.
Usa modelo entrenado sin reentrenar.

### [unet_model.py](/F:/dataset/detection-tb-web/unet_model.py)
Definiciones de arquitectura U-Net / Attention U-Net usadas para segmentacion pulmonar.

## 5. Requisitos e instalacion

```bash
pip install torch torchvision gradio opencv-python pillow numpy scikit-learn
```

## 6. Ejecutar aplicacion web

```bash
python app.py
```

Abrira interfaz local con dos modos:
- Imagen unica
- Lote multiples imagenes

## 7. Procesamiento por lote

En tab `Lote multiples imagenes`:
1. Arrastrar varias imagenes o seleccionar archivos.
2. Click `Procesar lote` o procesado automatico al cambiar lista.
3. Ver galerias por etapa (`Original`, `Lung Segmented`, `Score-CAM`).
4. Ver tabla final con `archivo`, `tb_prob`, `resultado`.

## 8. Entrenamiento clasificador

Ejemplo:

```bash
python train.py \
  --data-dir data_prepared_mixed \
  --epochs 50 \
  --batch-size 16 \
  --img-size 380 \
  --lung-segmentation-mode attention_unet \
  --threshold-policy balanced
```

## 9. Generar CAM grid

```bash
python generate_cam_grid.py \
  --image-path ruta/a/imagen.png \
  --model-path models/densenet_169_tb_best.pt
```

## 10. Notas importantes

- Proyecto es apoyo tecnico, no reemplaza diagnostico medico clinico.
- Calidad depende de dominio de datos y calidad de CXR.
- Casos fuera distribucion pueden degradar segmentacion y clasificacion.
