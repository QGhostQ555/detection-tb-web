import argparse
import os
from types import SimpleNamespace
import torch
from torchvision.models import densenet169
from train import generate_cam_grid_figure

def parse_args():
    base = os.path.dirname(os.path.abspath(__file__))
    default_model = os.path.abspath(os.path.join(base, "..", "models", "densenet_169_tb_best.pt"))
    if not os.path.isfile(default_model):
        # buscar alternativas
        for alt in ["models/modelo final/densenet_169_tb_best.pt",
                    "models/modelo bien/densenet_169_tb_best.pt",
                    "models/1.4/densenet_169_tb_best.pt"]:
            cand = os.path.abspath(os.path.join(base, "..", alt))
            if os.path.isfile(cand):
                default_model = cand
                break
    default_output = os.path.abspath(os.path.join(base, "..", "models", "tb_enhancement_cam_grid.png"))
    parser = argparse.ArgumentParser(description="Genera grilla Score‑CAM desde checkpoint")
    parser.add_argument("--model-path", type=str, default=default_model, help="Checkpoint .pt entrenado")
    parser.add_argument("--image-path", type=str, required=True, help="Radiografía CXR individual")
    parser.add_argument("--output-path", type=str, default=default_output)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    parser.add_argument("--img-size", type=int, default=380)   # se sobrescribe con el del ckpt
    return parser.parse_args()

def main():
    cli = parse_args()
    ckpt = torch.load(cli.model_path, map_location="cpu", weights_only=False)

    enhancement_mode = ckpt.get("enhancement_mode", "clahe_gamma")
    lung_seg_mode = ckpt.get("lung_segmentation_mode", "attention_unet")
    lung_unet_ckpt = ckpt.get("lung_unet_checkpoint", "models/lung_attention_unet_best.pt")
    mean = ckpt.get("mean", [0.485,0.456,0.406])
    std = ckpt.get("std", [0.229,0.224,0.225])
    img_size = ckpt.get("img_size", cli.img_size)

    class_to_idx = ckpt.get("class_to_idx_train", {"NORMAL":0,"TB":1})
    model = densenet169(weights=None)
    num_classes = len(class_to_idx)
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    tb_idx = ckpt.get("tb_index_train", class_to_idx.get("TB",1))

    cam_args = SimpleNamespace(
        img_size=img_size,
        enhancement_mode=enhancement_mode,
        clahe_clip_limit=ckpt.get("clahe_clip_limit",2.0),
        clahe_tile_grid=ckpt.get("clahe_tile_grid",8),
        gamma=ckpt.get("gamma",1.1),
        bcet_target_mean=ckpt.get("bcet_target_mean",110.0),
        unsharp_sigma=ckpt.get("unsharp_sigma",1.0),
        unsharp_amount=ckpt.get("unsharp_amount",1.0),
        scorecam_max_maps=32,
        scorecam_batch_size=8,
        scorecam_layer=8,
        scorecam_activation_quantile=0.70,
        cam_smooth_sigma=7.0,
        cam_low_percentile=12.0,
        cam_high_percentile=99.5,
        cam_threshold=0.08,
        cam_alpha=0.55,
        segment_outside_scale=ckpt.get("lung_segmentation_outside_scale",0.08),
        lung_segmentation_mode=lung_seg_mode,
        lung_segmentation_outside_scale=ckpt.get("lung_segmentation_outside_scale",0.08),
        lung_unet_checkpoint=lung_unet_ckpt,
        allow_collage_image=False
    )

    generate_cam_grid_figure(
        model=model, device=device, tb_index=tb_idx,
        img_path=cli.image_path, output_path=cli.output_path,
        mean=mean, std=std, args=cam_args
    )
    print(f"Grilla guardada en: {cli.output_path}")

if __name__ == "__main__":
    main()