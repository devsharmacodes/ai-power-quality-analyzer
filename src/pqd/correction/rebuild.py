"""Interruption and transient: replace the damaged part with a sine fitted on clean cycles nearby.

For an interruption the real voltage information is gone, so this is only an ESTIMATE
of what the waveform would have been (real systems use a UPS or battery backup).
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..processing.sinefit import reference_fit


def rebuild_sine(x, i0: int, i1: int, cfg: Config, pad: int = 0):
    a, b = max(0, i0 - pad), min(len(x), i1 + pad)
    ref = reference_fit(x, a, b, cfg)
    y = np.array(x, dtype=float, copy=True)
    y[a:b] = ref(np.arange(a, b))
    return y
