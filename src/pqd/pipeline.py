"""The whole chain in one call: window -> classify -> events -> report -> correction.

predictor is any function  windows -> (class index per window, probabilities).
Use pqd.models.predict.make_predictor(model) for the trained CNN.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .correction.router import correct_signal
from .events.detector import detect_events
from .processing.windowing import sliding_windows
from .report.builder import build_report


def analyze(signal, predictor, cfg: Config, smooth: int = 5) -> dict:
    x = np.asarray(signal, dtype=float)
    windows, starts = sliding_windows(x, cfg.window_samples, cfg.step_samples)
    pred, probs = predictor(windows)
    events = detect_events(starts, pred, probs, x, cfg, smooth=smooth)
    report = build_report(x, events, cfg)
    corrected, actions = correct_signal(x, report, cfg)
    return {
        "starts": starts, "window_labels": pred, "window_probs": probs,
        "events": events, "report": report, "corrected": corrected, "actions": actions,
    }
