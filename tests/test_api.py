from __future__ import annotations

import base64


def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_health_reports_model_not_loaded(client_no_model):
    response = client_no_model.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model_loaded"] is False


def test_predict_valid_image_returns_expected_shape(client, sample_image_bytes):
    response = client.post(
        "/predict",
        files={"file": ("face.png", sample_image_bytes, "image/png")},
    )
    assert response.status_code == 200
    body = response.json()

    assert body["label"] in {"fake", "real"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["probabilities"].keys()) == {"fake", "real"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-4
    assert body["latency_ms"] > 0

    # heatmap should be a real, decodable PNG
    decoded = base64.b64decode(body["heatmap_base64"])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_predict_rejects_non_image_upload(client):
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"this is not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()


def test_predict_returns_503_when_model_missing(client_no_model, sample_image_bytes):
    response = client_no_model.post(
        "/predict",
        files={"file": ("face.png", sample_image_bytes, "image/png")},
    )
    assert response.status_code == 503
