import os
from typing import List, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from classifier_model import load_classifier
from train import (
    NeuralLungSegmentationEnhancer,
    ScoreCAM,
    _overlay_cam_on_gray,
    build_enhancer,
)


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
    fallback="heuristic",
)

cam = ScoreCAM(
    model=clf["model"],
    target_layer=clf["model"].features[-1],
    max_maps=32,
    batch_size=8,
    activation_quantile=0.70,
)

enhancement_mode = clf.get("enhancement_mode", "clahe_gamma")
enhancer = build_enhancer(
    mode=enhancement_mode,
    clahe_clip_limit=clf.get("clahe_clip_limit", 2.0),
    clahe_tile_grid=clf.get("clahe_tile_grid", 8),
    gamma=clf.get("gamma", 1.1),
)

_transform = transforms.Compose([
    transforms.Resize((clf["img_size"], clf["img_size"])),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=clf["mean"], std=clf["std"]),
])


def _predict_one(
    img: Image.Image,
) -> Tuple[Image.Image, Image.Image, Image.Image, Image.Image, float, str]:
    # 1) CLAHE+Gamma
    enhanced_pil = enhancer(img.convert("L")) if enhancer is not None else img.convert("L")
    enhanced_np = np.array(enhanced_pil.convert("L"), dtype=np.uint8)

    # 2) Segmentacion pulmonar sobre imagen mejorada
    full_mask = lung_seg.predict_mask(enhanced_np)
    lung_pil = lung_seg(enhanced_pil)

    # 3) Clasificacion
    input_tensor = _transform(lung_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = clf["model"](input_tensor)
        probs = torch.softmax(logits, 1)[0]
    tb_prob = float(probs[clf["tb_index"]])

    # 4) Score-CAM
    target_size = (clf["img_size"], clf["img_size"])
    mask_resized = cv2.resize(full_mask, target_size, interpolation=cv2.INTER_NEAREST)
    cam_map = cam.compute(input_tensor, class_idx=clf["tb_index"], region_mask=mask_resized)

    gray_np = np.array(img.convert("L"), dtype=np.uint8)
    overlay_rgb = _overlay_cam_on_gray(
        gray=gray_np,
        cam_map=cam_map,
        restrict_mask=mask_resized,
        sigma=7.0,
        low_pct=12.0,
        high_pct=99.5,
        threshold=0.08,
        alpha=0.55,
    )

    result = "TB" if tb_prob >= clf["threshold"] else "NORMAL"
    return img, enhanced_pil, lung_pil, Image.fromarray(overlay_rgb), tb_prob, result


def predict_single(img: Image.Image):
    if img is None:
        return None, None, None, None, "Sin imagen"
    o, e, l, c, p, r = _predict_one(img)
    return o, e, l, c, (
        f"Pipeline: CLAHE+Gamma -> Segmentacion pulmonar -> Clasificacion -> Score-CAM\n"
        f"Enhancement mode: {enhancement_mode}\n"
        f"TB probability: {p:.3f}\nResult: {r}"
    )


def predict_batch(files: List[str]):
    if not files:
        return [], [], [], [], []

    originals, enhanced, segmented, cams = [], [], [], []
    rows = []
    for path in files:
        try:
            img = Image.open(path).convert("RGB")
            o, e, l, c, p, r = _predict_one(img)
            originals.append(o)
            enhanced.append(e)
            segmented.append(l)
            cams.append(c)
            rows.append([os.path.basename(path), f"{p:.4f}", r])
        except Exception as ex:
            rows.append([os.path.basename(path), "ERROR", str(ex)])

    return originals, enhanced, segmented, cams, rows


def clear_single():
    return None, None, None, None, "Sin imagen"


def clear_batch():
    return [], [], [], [], []


if __name__ == "__main__":
    with gr.Blocks(title="TB Detection Web") as demo:
        gr.Markdown("# TB Detection Web")

        with gr.Tab("Imagen unica"):
            in_img = gr.Image(type="pil", label="Arrastra imagen (reemplaza actual)")
            with gr.Row():
                run_one = gr.Button("Procesar Imagen", variant="primary")
                clear_one = gr.Button("Limpiar")
            with gr.Row():
                out_o = gr.Image(label="Original", height=220)
                out_e = gr.Image(label="CLAHE+Gamma", height=220)
            with gr.Row():
                out_l = gr.Image(label="Lung Segmented", height=220)
                out_c = gr.Image(label="Score-CAM", height=220)
            with gr.Row():
                out_t = gr.Textbox(label="Diagnosis", lines=12, max_lines=12)

            run_one.click(predict_single, [in_img], [out_o, out_e, out_l, out_c, out_t])
            in_img.change(predict_single, [in_img], [out_o, out_e, out_l, out_c, out_t])
            clear_one.click(clear_single, [], [out_o, out_e, out_l, out_c, out_t])

        with gr.Tab("Lote multiples imagenes"):
            in_files = gr.File(
                file_count="multiple",
                file_types=["image"],
                type="filepath",
                label="Arrastra varias imagenes (nuevo arrastre reemplaza lote)",
            )
            with gr.Row():
                run_batch = gr.Button("Procesar Lote", variant="primary")
                clear_batch_btn = gr.Button("Limpiar lote")

            out_go = gr.Gallery(label="Originales", columns=4, height=210)
            out_ge = gr.Gallery(label="CLAHE+Gamma", columns=4, height=210)
            out_gl = gr.Gallery(label="Lung Segmented", columns=4, height=210)
            out_gc = gr.Gallery(label="Score-CAM", columns=4, height=210)
            out_table = gr.Dataframe(
                headers=["archivo", "tb_prob", "resultado"],
                datatype=["str", "str", "str"],
                interactive=False,
                label="Resumen por imagen",
            )

            run_batch.click(predict_batch, [in_files], [out_go, out_ge, out_gl, out_gc, out_table])
            in_files.change(predict_batch, [in_files], [out_go, out_ge, out_gl, out_gc, out_table])
            clear_batch_btn.click(clear_batch, [], [out_go, out_ge, out_gl, out_gc, out_table])

    demo.launch()
