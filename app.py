import torch
import gradio as gr
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms

from classifier_model import load_classifier
from train import (
    NeuralLungSegmentationEnhancer,
    ScoreCAM,
    _overlay_cam_on_gray,
)

# --------------------------------------------------------------------------
# 1. Carga de modelos (igual que antes, pero con gestión de advertencias)
# --------------------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        return torch.device("cpu")

device = get_device()

clf = load_classifier("models/densenet_169_tb_best.pt", device)

lung_seg = NeuralLungSegmentationEnhancer(
    checkpoint_path="models/lung_attention_unet_best.pt",
    outside_scale=0.08,
    fallback="heuristic"
)

cam = ScoreCAM(
    model=clf["model"],
    target_layer=clf["model"].features[-1],
    max_maps=32,
    batch_size=8,
    activation_quantile=0.70,
)

# Transformación que se aplicará a la imagen segmentada antes de clasificar
_transform = transforms.Compose([
    transforms.Resize((clf["img_size"], clf["img_size"])),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=clf["mean"], std=clf["std"]),
])

# --------------------------------------------------------------------------
# 2. Función de predicción (con redimensionado explícito de la máscara)
# --------------------------------------------------------------------------
def predict(img):
    # --- 2.1 Segmentación pulmonar ---
    gray_np = np.array(img.convert("L"))
    full_mask = lung_seg.predict_mask(gray_np)                     # tamaño original
    lung_pil = lung_seg(img)                                       # imagen atenuada

    # --- 2.2 Preprocesamiento para el clasificador ---
    input_tensor = _transform(lung_pil).unsqueeze(0).to(device)

    # --- 2.3 Clasificación ---
    with torch.no_grad():
        logits = clf["model"](input_tensor)
        probs = torch.softmax(logits, 1)[0]
    tb_prob = float(probs[clf["tb_index"]])

    # --- 2.4 Score‑CAM (¡la máscara debe coincidir con el tamaño del modelo!) ---
    target_size = (clf["img_size"], clf["img_size"])  # (380, 380)
    mask_resized = cv2.resize(full_mask, target_size, interpolation=cv2.INTER_NEAREST)

    cam_map = cam.compute(input_tensor, class_idx=clf["tb_index"], region_mask=mask_resized)

    # --- 2.5 Superposición sobre la imagen original ---
    overlay_rgb = _overlay_cam_on_gray(
        gray=gray_np,
        cam_map=cam_map,
        restrict_mask=mask_resized,    # aquí también usamos la máscara redimensionada
        sigma=7.0, low_pct=12.0, high_pct=99.5,
        threshold=0.08, alpha=0.55,
    )

    # --- 2.6 Diagnóstico final ---
    resultado = "🟥 TB" if tb_prob >= clf["threshold"] else "🟩 NORMAL"
    return (
        img,
        lung_pil,                         # ← PIL Image directamente
        Image.fromarray(overlay_rgb),     # ← array NumPy → PIL
        f"TB probability: {tb_prob:.3f}\nResult: {resultado}",
    )

# --------------------------------------------------------------------------
# 3. Interfaz Gradio
# --------------------------------------------------------------------------
if __name__ == "__main__":
    gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Image(label="Original"),
            gr.Image(label="Lung Segmented"),
            gr.Image(label="Score-CAM"),
            gr.Text(label="Diagnosis"),
        ],
        title="TB Detection with Lung Segmentation & Score-CAM",
    ).launch()