# Detection TB Web

App web para deteccion de TB en radiografias CXR con:
- Preprocesado `CLAHE+Gamma`
- Segmentacion pulmonar (Attention U-Net)
- Clasificacion TB vs NORMAL (DenseNet169)
- Explicabilidad visual (Score-CAM)

## Estado actual del proyecto

Interfaz actual en `app.py` tiene dos modos:
- `Imagen unica`: arrastrar/subir 1 imagen, reemplaza actual, procesa automatico al cambiar.
- `Lote multiples imagenes`: subir varias imagenes, nuevo arrastre reemplaza lote, procesa automatico al cambiar.

Salidas mostradas:
- Original
- CLAHE+Gamma
- Lung Segmented
- Score-CAM
- Diagnostico (probabilidad + clase)

## Pipeline actual (inferencia web)

1. Cargar CXR.
2. Aplicar enhancement `CLAHE+Gamma`.
3. Obtener mascara pulmonar con `NeuralLungSegmentationEnhancer` sobre imagen mejorada.
4. Atenuar fuera de pulmon (`outside_scale=0.08`).
5. Clasificar con DenseNet169.
6. Generar Score-CAM sobre region segmentada.
7. Mostrar resultados en UI.

## Archivos `.py` y rol

### [app.py](/F:/dataset/detection-tb-web/app.py)
Front-end Gradio + orquestacion de inferencia.
- Single image tab.
- Batch tab con tabla resumen por archivo.
- Eventos `.change(...)` para reproceso automatico al arrastrar nueva imagen/lista.
- Muestra `Original`, `CLAHE+Gamma`, `Lung Segmented`, `Score-CAM`.

### [classifier_model.py](/F:/dataset/detection-tb-web/classifier_model.py)
Carga checkpoint de clasificador.
- Reconstruye DenseNet169 desde `model_state_dict`.
- Devuelve metadata: `img_size`, `threshold`, `mean`, `std`, `tb_index`.
- Incluye metadata de enhancement: `enhancement_mode`, `clahe_clip_limit`, `clahe_tile_grid`, `gamma`.

### [train.py](/F:/dataset/detection-tb-web/train.py)
Script grande de entrenamiento/evaluacion/utilidades.
- Modos de enhancement (CLAHE, gamma, BCET, etc.).
- Segmentacion pulmonar heuristica y U-Net/Attention U-Net.
- Entrenamiento DenseNet169.
- Seleccion de threshold.
- Utilidades Score-CAM y grid comparativo.

### [generate_cam_grid.py](/F:/dataset/detection-tb-web/generate_cam_grid.py)
Genera imagen comparativa de Score-CAM desde checkpoint entrenado.

### [unet_model.py](/F:/dataset/detection-tb-web/unet_model.py)
Definiciones de arquitectura U-Net usadas para segmentacion.

## Requisitos

```bash
pip install torch torchvision gradio opencv-python pillow numpy scikit-learn
```

## Ejecutar web

```bash
python app.py
```

## Uso rapido

### Imagen unica
1. Ir a tab `Imagen unica`.
2. Arrastrar imagen CXR.
3. Ver `Original`, `CLAHE+Gamma`, `Lung Segmented`, `Score-CAM`, `Diagnosis`.

### Lote multiples imagenes
1. Ir a tab `Lote multiples imagenes`.
2. Arrastrar varias imagenes.
3. Ver galerias y tabla:
- `archivo`
- `tb_prob`
- `resultado`

## Ejecutar CAM grid

```bash
python generate_cam_grid.py --image-path ruta/a/cxr.png --model-path models/densenet_169_tb_best.pt
```

## Modelos esperados en `models/`

- `densenet_169_tb_best.pt`
- `lung_attention_unet_best.pt`

## Nota

Proyecto es apoyo tecnico. No reemplaza diagnostico medico clinico.
