"""
Model loading, preprocessing, and Grad-CAM inference for the DeepFake Face Detector.

The checkpoint format and preprocessing pipeline here must exactly match what was
used during training in the Colab notebook (see notebooks/AiFake.ipynb), otherwise
predictions will be silently wrong.
"""
from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms
from torchvision.models import resnet18

logger = logging.getLogger("deepfake_detector")

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.pth"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Split into two stages so we can keep the unnormalized 224x224 crop around for the
# Grad-CAM overlay while still feeding the network exactly what it was trained on.
_RESIZE_CROP = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ]
)
_TO_MODEL_INPUT = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


@dataclass
class ModelBundle:
    model: nn.Module
    idx_to_class: dict[int, str]
    cam: GradCAM


def build_model() -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(512, 2)
    return model


def load_model(model_path: Path = MODEL_PATH) -> Optional[ModelBundle]:
    """Load the checkpoint once at startup. Returns None (never raises) if the
    weights file is missing or corrupt, so the API can still start and report
    the problem via /health and a 503 on /predict."""
    if not model_path.exists():
        logger.warning("Model file not found at %s — /predict will return 503.", model_path)
        return None

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        model = build_model()
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        class_to_idx: dict[str, int] = checkpoint["class_to_idx"]
        idx_to_class = {idx: name for name, idx in class_to_idx.items()}

        target_layers = [model.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)

        logger.info(
            "Loaded model from %s (epoch=%s, val_acc=%s)",
            model_path,
            checkpoint.get("epoch"),
            checkpoint.get("val_acc"),
        )
        return ModelBundle(model=model, idx_to_class=idx_to_class, cam=cam)
    except Exception:
        logger.exception("Failed to load model from %s — /predict will return 503.", model_path)
        return None


def preprocess(image: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
    """Returns (normalized input tensor for the model, unnormalized HWC float
    [0,1] array for the Grad-CAM overlay), both at 224x224."""
    rgb_image = image.convert("RGB")
    cropped = _RESIZE_CROP(rgb_image)
    rgb_float = np.asarray(cropped).astype(np.float32) / 255.0
    tensor = _TO_MODEL_INPUT(cropped).unsqueeze(0)
    return tensor, rgb_float


def _encode_png_base64(rgb_array: np.ndarray) -> str:
    image = Image.fromarray(rgb_array)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def run_inference(image: Image.Image, bundle: ModelBundle) -> dict:
    start = time.perf_counter()

    tensor, rgb_float = preprocess(image)

    with torch.no_grad():
        logits = bundle.model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(torch.argmax(probs).item())
    label = bundle.idx_to_class[pred_idx]
    confidence = float(probs[pred_idx].item())
    probabilities = {bundle.idx_to_class[i]: float(probs[i].item()) for i in range(probs.shape[0])}

    # Grad-CAM needs its own forward+backward pass with gradients enabled,
    # which is why this runs outside the no_grad() block above.
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = bundle.cam(input_tensor=tensor, targets=targets)[0]
    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    heatmap_base64 = _encode_png_base64(overlay)

    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "heatmap_base64": heatmap_base64,
        "latency_ms": round(latency_ms, 2),
    }
