"""Sliding window: a small box that moves along the long signal, one piece at a time."""
from __future__ import annotations

import numpy as np

from ..config import CLASS_TO_IDX, Config


def sliding_windows(x, win: int, step: int):
    """Return (windows, starts). windows has shape (n, win); starts are sample indices."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) < win:
        raise ValueError(f"signal ({len(x)} samples) is shorter than one window ({win})")
    view = np.lib.stride_tricks.sliding_window_view(x, win)[::step]
    starts = np.arange(0, len(x) - win + 1, step)
    return np.ascontiguousarray(view), starts


def window_overlap(starts, win: int, a: int, b: int):
    """How many samples of [a, b) each window covers."""
    s = np.asarray(starts)
    return np.clip(np.minimum(s + win, b) - np.maximum(s, a), 0, None)


def label_windows(events, starts, cfg: Config):
    """Answer-key label for each window of a stream (class index).

    A window gets an event's class when it covers at least min_cover of the window
    (or 90% of a very short event, such as a transient). Otherwise it is 'normal'.
    """
    W = cfg.window_samples
    labels = np.zeros(len(starts), dtype=int)
    best = np.zeros(len(starts))
    for ev in events:
        length = ev.end - ev.start
        required = max(1.0, min(cfg.min_cover * W, 0.9 * length))
        ov = window_overlap(starts, W, ev.start, ev.end)
        score = ov / required
        ok = (ov >= required) & (score > best)
        labels[ok] = CLASS_TO_IDX[ev.label]
        best[ok] = score[ok]
    return labels
