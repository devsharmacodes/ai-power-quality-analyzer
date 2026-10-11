"""Sag and swell: multiply the faulty part by (nominal / measured RMS)."""
from __future__ import annotations

import numpy as np


def gain_correct(x, i0: int, i1: int, rms_value_pu: float, max_gain: float = 10.0):
    y = np.array(x, dtype=float, copy=True)
    gain = min(1.0 / max(rms_value_pu, 1e-6), max_gain)
    y[i0:i1] *= gain
    return y
