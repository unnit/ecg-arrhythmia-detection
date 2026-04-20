REPO_ID = "dheerajthuvara/ecg-arrhythmia-detection"

MODEL_FILES = {
    "config": "models/model_config.json",
    "encoder": "models/label_encoder.pkl",
    "weights": "models/best_model.pt",
}

CLASS_DESCRIPTIONS = {
    "N": "Normal sinus rhythm",
    "L": "Left bundle branch block (LBBB)",
    "R": "Right bundle branch block (RBBB)",
    "V": "Premature ventricular contraction (PVC)",
    "A": "Atrial premature contraction (APC)",
}

CLASS_COLORS = {
    "N": "#2196F3",
    "L": "#4CAF50",
    "R": "#FF9800",
    "V": "#F44336",
    "A": "#9C27B0",
}

SAMPLE_RATE = 360
MIN_SIGNAL_LENGTH = 10
