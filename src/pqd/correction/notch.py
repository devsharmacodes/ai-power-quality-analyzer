"""Harmonics: notch filters remove one exact frequency each (3rd, 5th, 7th)."""
from __future__ import annotations

import numpy as np
from scipy import signal

from ..config import Config


def notch_correct(x, i0: int, i1: int, cfg: Config, orders=(3, 5, 7), q: float = 30.0):
    pad = 3 * cfg.cycle_samples
    lo, hi = max(0, i0 - pad), min(len(x), i1 + pad)
    seg = np.asarray(x[lo:hi], dtype=float)
    for h in orders:
        f = h * cfg.f_nominal
        if f >= cfg.fs / 2:
            continue
        b, a = signal.iirnotch(f, q, fs=cfg.fs)
        seg = signal.filtfilt(b, a, seg)      # zero-phase: does not shift the waveform
    y = np.array(x, dtype=float, copy=True)
    y[i0:i1] = seg[i0 - lo: i1 - lo]
    return y
