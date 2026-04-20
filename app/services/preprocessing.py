import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal: np.ndarray, lowcut: float = 0.5, highcut: float = 50.0,
                    fs: int = 360, order: int = 4) -> np.ndarray:
    """Remove baseline wander and high-frequency noise."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal)


def normalize_segment(segment: np.ndarray) -> np.ndarray:
    """Z-score normalize a single beat segment."""
    std = np.std(segment)
    return (segment - np.mean(segment)) / (std if std > 0 else 1.0)


def extract_center_window(signal: np.ndarray, window_size: int) -> np.ndarray:
    """Extract a single center window from a signal for single-beat inference."""
    if len(signal) >= window_size:
        start = (len(signal) - window_size) // 2
        segment = signal[start: start + window_size]
    else:
        segment = np.pad(signal, (0, window_size - len(signal)))
    return normalize_segment(segment)


def segment_signal(signal: np.ndarray, window_size: int) -> np.ndarray:
    """Split a long signal into sequential fixed-size windows."""
    segments = []
    for start in range(0, len(signal) - window_size + 1, window_size):
        seg = signal[start: start + window_size]
        segments.append(normalize_segment(seg))
    return np.array(segments, dtype=np.float32)


def preprocess_signal(signal: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline.
    Returns filtered signal and segmented windows.
    """
    signal = signal.astype(np.float32)
    filtered = bandpass_filter(signal)
    segments = segment_signal(filtered, window_size)
    return filtered, segments
