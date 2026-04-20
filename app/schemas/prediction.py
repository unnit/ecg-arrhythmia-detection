from typing import List, Dict
from pydantic import BaseModel, field_validator


class SignalInput(BaseModel):
    signal: List[float]

    @field_validator("signal")
    @classmethod
    def signal_must_have_minimum_length(cls, v):
        if len(v) < 10:
            raise ValueError("Signal must have at least 10 samples.")
        return v


class WindowPrediction(BaseModel):
    window_index: int
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]


class PredictionResponse(BaseModel):
    total_windows: int
    dominant_class: str
    dominant_confidence: float
    windows: List[WindowPrediction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    repo_id: str
