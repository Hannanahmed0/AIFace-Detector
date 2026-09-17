"""
Shared pytest fixtures. Tests never touch the real trained weights in
models/best_model.pth — instead we save a randomly initialized resnet18 in
the exact checkpoint format the app expects, so tests are fast and don't
depend on training having happened.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
import torch
from PIL import Image

from backend.model import build_model


@pytest.fixture
def fake_checkpoint_path(tmp_path: Path) -> Path:
    model = build_model()
    checkpoint = {
        "model_state": model.state_dict(),
        "class_to_idx": {"fake": 0, "real": 1},
        "epoch": 0,
        "val_acc": 0.5,
    }
    path = tmp_path / "fake_model.pth"
    torch.save(checkpoint, path)
    return path


@pytest.fixture
def client(monkeypatch, fake_checkpoint_path: Path):
    monkeypatch.setenv("DEEPFAKE_MODEL_PATH", str(fake_checkpoint_path))
    from backend.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_no_model(monkeypatch, tmp_path: Path):
    missing_path = tmp_path / "does_not_exist.pth"
    monkeypatch.setenv("DEEPFAKE_MODEL_PATH", str(missing_path))
    from backend.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_image_bytes() -> bytes:
    image = Image.new("RGB", (256, 256), color=(120, 140, 160))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
