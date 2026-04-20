import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'app'))

import numpy as np
import pytest
from services.preprocessing import (
    bandpass_filter,
    normalize_segment,
    extract_center_window,
    segment_signal,
    preprocess_signal,
)

WINDOW_SIZE = 180


def make_signal(length=720):
    """Generate a synthetic ECG-like signal for testing."""
    t = np.linspace(0, 2, length)
    return np.sin(2 * np.pi * 1.2 * t).astype(np.float32)


# ── bandpass filter ───────────────────────────────────────────────────────────

def test_bandpass_filter_output_shape():
    signal = make_signal(720)
    filtered = bandpass_filter(signal)
    assert filtered.shape == signal.shape


def test_bandpass_filter_reduces_noise():
    # high frequency noise should be attenuated
    t = np.linspace(0, 1, 360)
    noise = np.sin(2 * np.pi * 100 * t).astype(np.float32)
    filtered = bandpass_filter(noise)
    assert np.std(filtered) < np.std(noise)


# ── normalize segment ─────────────────────────────────────────────────────────

def test_normalize_segment_mean_near_zero():
    segment = make_signal(WINDOW_SIZE)
    normalized = normalize_segment(segment)
    assert abs(np.mean(normalized)) < 1e-5


def test_normalize_segment_std_near_one():
    segment = make_signal(WINDOW_SIZE)
    normalized = normalize_segment(segment)
    assert abs(np.std(normalized) - 1.0) < 1e-4


def test_normalize_segment_flat_signal():
    # flat signal should not raise division by zero
    segment = np.zeros(WINDOW_SIZE, dtype=np.float32)
    normalized = normalize_segment(segment)
    assert np.all(normalized == 0.0)


# ── extract center window ─────────────────────────────────────────────────────

def test_extract_center_window_correct_length():
    signal = make_signal(720)
    window = extract_center_window(signal, WINDOW_SIZE)
    assert len(window) == WINDOW_SIZE


def test_extract_center_window_pads_short_signal():
    signal = make_signal(100)
    window = extract_center_window(signal, WINDOW_SIZE)
    assert len(window) == WINDOW_SIZE


# ── segment signal ────────────────────────────────────────────────────────────

def test_segment_signal_correct_number_of_windows():
    signal = make_signal(720)
    segments = segment_signal(signal, WINDOW_SIZE)
    assert len(segments) == 720 // WINDOW_SIZE


def test_segment_signal_correct_window_shape():
    signal = make_signal(720)
    segments = segment_signal(signal, WINDOW_SIZE)
    assert segments.shape == (4, WINDOW_SIZE)


def test_segment_signal_too_short_returns_empty():
    signal = make_signal(50)
    segments = segment_signal(signal, WINDOW_SIZE)
    assert len(segments) == 0


# ── preprocess signal ─────────────────────────────────────────────────────────

def test_preprocess_signal_returns_filtered_and_segments():
    signal = make_signal(720)
    filtered, segments = preprocess_signal(signal, WINDOW_SIZE)
    assert len(filtered) == 720
    assert len(segments) == 4


def test_preprocess_signal_filtered_shape_matches_input():
    signal = make_signal(360)
    filtered, _ = preprocess_signal(signal, WINDOW_SIZE)
    assert filtered.shape == signal.shape
