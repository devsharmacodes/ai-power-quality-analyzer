"""Turn window labels into events with a start time and an end time.

"Window 45 is a sag" becomes "a sag from 2.00 s to 2.18 s":
  1. smooth the labels (a majority vote over neighbouring windows),
  2. join neighbouring windows with the same label into one event,
  3. estimate the edges from the window positions,
  4. sharpen the edges from the signal itself (RMS for sag/swell/interruption,
     sharp changes for transients).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import CLASSES, Config

AMPLITUDE_LABELS = ("sag", "swell", "interruption")


@dataclass
class DetectedEvent:
    label: str
    start: int          # sample index
    end: int            # sample index (exclusive)
    confidence: float   # mean CNN probability of this class over the event's windows
    n_windows: int


def _mode_filter(labels: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return labels.copy()
    half = width // 2
    out = labels.copy()
    for i in range(len(labels)):
        out[i] = np.bincount(labels[max(0, i - half): i + half + 1]).argmax()
    return out


def _rolling_rms_pu(x: np.ndarray, n: int) -> np.ndarray:
    """RMS over one cycle at every start sample, in p.u. (a clean sine gives 1.0)."""
    cs = np.concatenate(([0.0], np.cumsum(np.square(x, dtype=float))))
    return np.sqrt((cs[n:] - cs[:-n]) / n) * np.sqrt(2.0)


def _refine_amplitude_rms(x, a, b, label, cfg: Config):
    """Rough edges (about +-3 ms) from the one-cycle rolling RMS."""
    n = cfg.cycle_samples
    lo, hi = max(0, a - 3 * n), min(len(x), b + 3 * n)
    seg = np.asarray(x[lo:hi], dtype=float)
    if len(seg) < 2 * n:
        return a, b
    r = _rolling_rms_pu(seg, n)
    k0 = max(0, (a - lo) - n // 2)
    k1 = min(len(r), (b - lo) - n // 2 + 1)
    if k1 <= k0:
        return a, b
    region = r[k0:k1]
    level = region.max() if label == "swell" else region.min()
    if abs(level - 1.0) < 0.03:
        return a, b
    thr = 1.0 + 0.5 * (level - 1.0)             # halfway between normal and the event level
    idx = np.flatnonzero(r > thr if label == "swell" else r < thr)
    if idx.size == 0:
        return a, b
    # a window that is half-way changed overlaps the event edge by eps of a cycle
    eps = (level + 3.0) / (4.0 * (level + 1.0))
    start = lo + int(round(idx[0] + n * (1.0 - eps)))
    end = lo + int(round(idx[-1] + eps * n))
    if end - start < n:
        return a, b
    return start, end


def _refine_amplitude_fine(x, a, b, cfg: Config):
    """Sharp edges (a few samples) from a quarter-cycle envelope.

    For a sine A*sin(wt), x[i]^2 + x[i - N/4]^2 equals A^2 exactly, whatever the phase.
    So this envelope is flat inside the event (at the event level squared) and equals 1.0
    before it; the edges are where it leaves those flat levels.
    """
    n = cfg.cycle_samples
    q = n // 4
    if n % 4 or b - a < 3 * q + 40:
        return a, b
    xf = np.asarray(x, dtype=float)
    env = np.full(len(xf), np.nan)
    env[q:] = xf[q:] ** 2 + xf[:-q] ** 2
    i0, i1 = a + q + 20, b - 20
    if i1 - i0 < 20:
        return a, b
    level = float(np.median(env[i0:i1]))
    if abs(level - 1.0) < 0.05:
        return a, b
    tol = max(0.03, 0.1 * abs(level - 1.0))
    lo = max(q, a - 60)
    idx = np.flatnonzero(np.abs(env[lo: a + q + 21] - 1.0) > tol)
    start = lo + int(idx[0]) if idx.size else a
    hi = min(len(xf), b + 60)
    idx2 = np.flatnonzero(np.abs(env[i1:hi] - level) > tol)
    end = i1 + int(idx2[0]) if idx2.size else b
    if end - start < n:
        return a, b
    return start, end


def _refine_amplitude(x, a, b, label, cfg: Config):
    a, b = _refine_amplitude_rms(x, a, b, label, cfg)
    return _refine_amplitude_fine(x, a, b, cfg)


def _refine_transient(x, a, b, cfg: Config):
    lo, hi = max(0, a - 50), min(len(x) - 1, b + 50)
    if hi - lo < 8:
        return a, b
    seg = np.asarray(x[lo:hi], dtype=float)
    d2 = np.abs(np.diff(seg, 2))                  # sharp changes show up in the 2nd difference
    sigma = np.median(d2) / 0.6745
    thr = max(6.0 * sigma, 0.02)
    idx = np.flatnonzero(d2 > thr)
    if idx.size == 0:
        mid = (a + b) // 2
        return mid - 10, mid + 10
    return lo + int(idx[0]), lo + int(idx[-1]) + 3


def detect_events(starts, labels, probs, signal, cfg: Config, smooth: int = 5,
                  min_windows: int = 2):
    """Return a list of DetectedEvent from per-window labels (class indices)."""
    labels = np.asarray(labels, dtype=int)
    starts = np.asarray(starts)
    probs = np.asarray(probs)
    sm = _mode_filter(labels, smooth)
    W = cfg.window_samples
    margin = (0.5 - cfg.min_cover) * W
    n_sig = len(signal)
    events = []
    i = 0
    while i < len(sm):
        if sm[i] == 0:
            i += 1
            continue
        j = i
        while j + 1 < len(sm) and sm[j + 1] == sm[i]:
            j += 1
        if j - i + 1 >= min_windows:
            cls = int(sm[i])
            name = CLASSES[cls]
            first_c, last_c = starts[i] + W / 2.0, starts[j] + W / 2.0
            a, b = int(round(first_c + margin)), int(round(last_c - margin))
            if b <= a:  # very small run: centre a one-cycle event on it
                mid = (first_c + last_c) / 2.0
                a, b = int(mid - cfg.cycle_samples / 2), int(mid + cfg.cycle_samples / 2)
            a, b = max(0, a), min(n_sig, b)
            if name in AMPLITUDE_LABELS:
                a, b = _refine_amplitude(signal, a, b, name, cfg)
            elif name == "transient":
                a, b = _refine_transient(signal, a, b, cfg)
            a, b = max(0, a), min(n_sig, b)
            if b > a:
                events.append(DetectedEvent(name, a, b, float(probs[i:j + 1, cls].mean()), j - i + 1))
        i = j + 1
    return events
