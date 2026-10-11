"""Verification: did the correction really bring the signal back to the clean one?

Works because the generated data keeps the clean signal. Compares, per event and
overall: error against the clean signal before vs after, RMS, THD (harmonics) and
noise level, and (optionally) re-classifies the corrected signal with the CNN.
"""
from __future__ import annotations

import numpy as np

from ..config import CLASSES, Config
from ..processing.windowing import sliding_windows, window_overlap
from ..report.metrics import noise_std_pu, rms_pu, thd_percent


def _rmse(a) -> float:
    return float(np.sqrt(np.mean(np.square(a))))


def evaluate_correction(clean, disturbed, corrected, report: list[dict], cfg: Config, predictor=None) -> dict:
    clean, disturbed, corrected = (np.asarray(v, dtype=float) for v in (clean, disturbed, corrected))
    out = {
        "overall": {
            "rmse_before": _rmse(disturbed - clean),
            "rmse_after": _rmse(corrected - clean),
        },
        "events": [],
    }
    pred = starts = None
    if predictor is not None:
        windows, starts = sliding_windows(corrected, cfg.window_samples, cfg.step_samples)
        pred, _ = predictor(windows)

    for row in report:
        i0, i1, label = row["start_idx"], row["end_idx"], row["label"]
        item = {
            "id": row["id"], "label": label,
            "rmse_before": _rmse(disturbed[i0:i1] - clean[i0:i1]),
            "rmse_after": _rmse(corrected[i0:i1] - clean[i0:i1]),
        }
        if label in ("sag", "swell", "interruption"):
            item["rms_pu_before"] = rms_pu(disturbed[i0:i1])
            item["rms_pu_after"] = rms_pu(corrected[i0:i1])
            item["rms_pu_clean"] = rms_pu(clean[i0:i1])
        if label == "harmonics":
            item["thd_pct_before"] = thd_percent(disturbed[i0:i1], cfg)
            item["thd_pct_after"] = thd_percent(corrected[i0:i1], cfg)
        if label == "noise":
            item["noise_pct_before"] = _pct(noise_std_pu(disturbed, i0, i1, cfg))
            item["noise_pct_after"] = _pct(noise_std_pu(corrected, i0, i1, cfg))
        if pred is not None:
            W = cfg.window_samples
            required = max(1.0, min(cfg.min_cover * W, 0.9 * (i1 - i0)))
            mask = window_overlap(starts, W, i0, i1) >= required
            if mask.any():
                item["pct_windows_normal_after"] = float(np.mean(pred[mask] == CLASSES.index("normal")) * 100)
        out["events"].append(item)
    return out


def _pct(v):
    return None if v is None else v * 100.0
