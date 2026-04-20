import io
from contextlib import asynccontextmanager
from collections import Counter

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from core.config import REPO_ID
from services.model_service import load_model, get_model, get_cfg, run_inference
from services.preprocessing import preprocess_signal
from schemas.prediction import SignalInput, PredictionResponse, WindowPrediction, HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="ECG Arrhythmia Detection API",
    description="Classifies ECG beats into N, L, R, V, A using a CNN-BiLSTM model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_response(result: dict) -> PredictionResponse:
    counts = Counter(result["predicted_classes"])
    dominant = counts.most_common(1)[0][0]
    dominant_idx = result["predicted_classes"].index(dominant)
    dominant_conf = result["confidences"][dominant_idx]

    windows = [
        WindowPrediction(
            window_index=i,
            predicted_class=cls,
            confidence=round(conf, 4),
            probabilities=probs,
        )
        for i, (cls, conf, probs) in enumerate(zip(
            result["predicted_classes"],
            result["confidences"],
            result["probabilities"],
        ))
    ]

    return PredictionResponse(
        total_windows=len(windows),
        dominant_class=dominant,
        dominant_confidence=round(dominant_conf, 4),
        windows=windows,
    )


@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "model": REPO_ID}


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    return HealthResponse(
        status="healthy",
        model_loaded=get_model() is not None,
        repo_id=REPO_ID,
    )


@app.post("/predict/json", response_model=PredictionResponse, tags=["predict"])
def predict_json(body: SignalInput):
    """Accept a raw ECG signal as a JSON array of floats."""
    cfg = get_cfg()
    signal = np.array(body.signal, dtype=np.float32)
    _, segments = preprocess_signal(signal, cfg["window_size"])
    if len(segments) == 0:
        raise HTTPException(status_code=422, detail="Signal too short to extract any windows.")
    result = run_inference(segments)
    return build_response(result)


@app.post("/predict/csv", response_model=PredictionResponse, tags=["predict"])
async def predict_csv(file: UploadFile = File(...)):
    """Accept a single-column CSV file containing ECG signal values."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=422, detail="Only .csv files are accepted.")
    contents = await file.read()
    try:
        signal = np.loadtxt(io.StringIO(contents.decode("utf-8")), delimiter=",")
        if signal.ndim > 1:
            signal = signal[:, 0]
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")
    cfg = get_cfg()
    _, segments = preprocess_signal(signal, cfg["window_size"])
    if len(segments) == 0:
        raise HTTPException(status_code=422, detail="Signal too short to extract any windows.")
    result = run_inference(segments)
    return build_response(result)
