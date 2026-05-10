"""
Export resnet50_phase2.pt → model.onnx
Run: python export_onnx.py
"""

import os
import sys
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

NUM_CLASSES = 196
CHECKPOINT = os.path.join(os.path.dirname(__file__), "../L3/checkpoints/resnet50_phase2.pt")
OUTPUT = os.path.join(os.path.dirname(__file__), "model.onnx")


def build_resnet50(num_classes: int) -> nn.Module:
    model = resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes),
    )
    return model


def export():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = build_resnet50(NUM_CLASSES)
    state = torch.load(CHECKPOINT, map_location=device)
    # checkpoints may be bare state_dicts or wrapped dicts
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval().to(device)

    dummy = torch.randn(1, 3, 224, 224, device=device)

    torch.onnx.export(
        model,
        dummy,
        OUTPUT,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported → {OUTPUT}")


if __name__ == "__main__":
    export()