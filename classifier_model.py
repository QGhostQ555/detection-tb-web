import torch
import torch.nn as nn
from torchvision import models

def load_classifier(path, device):
    ckpt = torch.load(path, map_location="cpu")
    # DenseNet169
    model = models.densenet169(weights=None)
    num_classes = ckpt["model_state_dict"]["classifier.weight"].shape[0]
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return {
        "model": model,
        "img_size": ckpt["img_size"],
        "threshold": ckpt["threshold"],
        "mean": ckpt["mean"],
        "std": ckpt["std"],
        "tb_index": ckpt["tb_index_train"],
        "enhancement_mode": ckpt.get("enhancement_mode", "clahe_gamma"),
        "clahe_clip_limit": ckpt.get("clahe_clip_limit", 2.0),
        "clahe_tile_grid": ckpt.get("clahe_tile_grid", 8),
        "gamma": ckpt.get("gamma", 1.1),
    }
