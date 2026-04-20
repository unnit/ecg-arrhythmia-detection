import json
import pickle

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from core.config import REPO_ID, MODEL_FILES
from core.model import CNNBiLSTM

DEVICE = torch.device("cpu")

# module-level singletons
_model = None
_label_encoder = None
_cfg = None


def load_model():
    """Download artifacts from HuggingFace Hub and load model into memory."""
    global _model, _label_encoder, _cfg

    print("Downloading model artifacts from HuggingFace Hub...")

    config_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILES["config"])
    with open(config_path) as f:
        _cfg = json.load(f)

    encoder_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILES["encoder"])
    with open(encoder_path, "rb") as f:
        _label_encoder = pickle.load(f)

    weights_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILES["weights"])
    _model = CNNBiLSTM(_cfg).to(DEVICE)
    _model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    _model.eval()

    print("Model loaded successfully.")


def get_model():
    return _model


def get_encoder():
    return _label_encoder


def get_cfg():
    return _cfg


def run_inference(segments: np.ndarray) -> dict:
    """
    Run model inference on preprocessed segments.
    Returns predicted classes, confidences and per-class probabilities.
    """
    tensor = torch.tensor(segments, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    with torch.no_grad():
        logits = _model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    classes = _label_encoder.classes_.tolist()
    pred_indices = np.argmax(probs, axis=1)
    pred_classes = [classes[i] for i in pred_indices]
    confidences = probs.max(axis=1).tolist()
    probabilities = [
        {cls: round(float(p), 4) for cls, p in zip(classes, prob_row)}
        for prob_row in probs
    ]

    return {
        "predicted_classes": pred_classes,
        "confidences": confidences,
        "probabilities": probabilities,
        "classes": classes,
    }
