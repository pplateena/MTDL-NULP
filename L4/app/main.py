"""
FastAPI inference service for Stanford Cars ResNet-50 classifier.
"""

import io
import os
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image

from app.inference import OnnxInferenceSession
from app.schemas import HealthResponse, PredictionResponse

MODEL_PATH = os.getenv("MODEL_PATH", "model.onnx")
PROVIDER   = os.getenv("PROVIDER", "cpu")          # cpu | cuda | tensorrt

# Stanford Cars 196 class names (index-ordered)
_CLASS_NAMES_FILE = os.path.join(os.path.dirname(__file__), "class_names.txt")

def _load_class_names():
    if os.path.exists(_CLASS_NAMES_FILE):
        with open(_CLASS_NAMES_FILE) as f:
            return [l.strip() for l in f if l.strip()]
    return [f"class_{i}" for i in range(196)]


_session: OnnxInferenceSession = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _session
    class_names = _load_class_names()
    _session = OnnxInferenceSession(MODEL_PATH, provider=PROVIDER, class_names=class_names)
    print(f"[startup] provider={_session.provider}  model={MODEL_PATH}")
    yield
    _session = None


app = FastAPI(title="Stanford Cars Classifier", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    if _session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ok",
        provider=_session.provider,
        model=os.path.basename(MODEL_PATH),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="File must be an image")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Cannot decode image")

    result = _session.predict(np.array(img))
    return PredictionResponse(**result)
