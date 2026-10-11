"""Check detected events against the true event list of a generated stream."""
from __future__ import annotations

import numpy as np


def _iou(a0, a1, b0, b1) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union > 0 else 0.0


def match_events(true_events, detected, fs: float, iou_threshold: float = 0.3) -> dict:
    """Greedy match by overlap (IoU). Works with Event / DetectedEvent objects."""
    pairs = sorted(
        ((_iou(t.start, t.end, d.start, d.end), ti, di)
         for ti, t in enumerate(true_events) for di, d in enumerate(detected)),
        reverse=True,
    )
    used_t, used_d, matches = set(), set(), []
    for iou, ti, di in pairs:
        if iou < iou_threshold:
            break
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        t, d = true_events[ti], detected[di]
        matches.append({
            "true_label": t.label, "detected_label": d.label, "iou": round(iou, 3),
            "start_error_ms": round((d.start - t.start) / fs * 1000, 1),
            "end_error_ms": round((d.end - t.end) / fs * 1000, 1),
        })
    n_match = len(matches)
    return {
        "n_true": len(true_events),
        "n_detected": len(detected),
        "matched": n_match,
        "missed": len(true_events) - n_match,
        "false_alarms": len(detected) - n_match,
        "label_correct": sum(m["true_label"] == m["detected_label"] for m in matches),
        "mean_abs_start_error_ms": float(np.mean([abs(m["start_error_ms"]) for m in matches])) if matches else None,
        "mean_abs_end_error_ms": float(np.mean([abs(m["end_error_ms"]) for m in matches])) if matches else None,
        "matches": matches,
    }
