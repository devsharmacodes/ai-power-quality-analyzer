"""Fit a clean 50 Hz sine to a stretch of signal (used to measure and to repair)."""
from __future__ import annotations

import numpy as np

from ..config import Config


def fit_fundamental(x, start: int, stop: int, cfg: Config):
    """Least-squares fit of A*sin + B*cos at f_nominal over x[start:stop].

    Returns a function that gives the fitted sine for any sample indices.
    """
    idx = np.arange(start, stop)
    w = 2 * np.pi * cfg.f_nominal / cfg.fs
    basis = np.column_stack([np.sin(w * idx), np.cos(w * idx)])
    coef, *_ = np.linalg.lstsq(basis, np.asarray(x[start:stop], dtype=float), rcond=None)

    def fitted(i):
        i = np.asarray(i)
        return coef[0] * np.sin(w * i) + coef[1] * np.cos(w * i)

    return fitted


def reference_fit(x, event_start: int, event_end: int, cfg: Config, n_cycles: int = 2):
    """Fit the sine on clean cycles just before the event (or just after, if none before)."""
    n = cfg.cycle_samples
    b0, b1 = max(0, event_start - n_cycles * n), event_start
    if b1 - b0 >= n:
        return fit_fundamental(x, b0, b1, cfg)
    a0, a1 = event_end, min(len(x), event_end + n_cycles * n)
    if a1 - a0 >= n:
        return fit_fundamental(x, a0, a1, cfg)
    return fit_fundamental(x, 0, len(x), cfg)
