"""
FastAPI app: loads the model once at startup, exposes /health and /predict,
and serves the single-page frontend so the whole project runs with one command.
"""
from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from backend.model import MODEL_PATH, ModelBundle, load_model, run_inference

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepfake_detector")

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tests point this at a throwaway checkpoint (or a missing path) via env
    # var so they never depend on the real trained weights.
    override = os.environ.get("DEEPFAKE_MODEL_PATH")
    path = Path(override) if override else MODEL_PATH
    app.state.model_bundle = load_model(path)
    yield


app = FastAPI(title="DeepFake Face Detector API", lifespan=lifespan)

# Populated on startup (see lifespan() above); stays None if weights are
# missing/corrupt so the app can still boot and /predict can report a clean
# 503 instead of crashing.
app.state.model_bundle = None


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": app.state.model_bundle is not None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    bundle: ModelBundle | None = app.state.model_bundle
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Place a valid checkpoint at models/best_model.pth and restart the server.",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type '{file.content_type}'. Please upload an image file.",
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10 MB.")
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Could not read file as an image.")

    return run_inference(image, bundle)


# Registered last so /health and /predict are matched before falling through
# to static file serving of the frontend at "/".
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
