"""Exact formulas for the report. The CNN names the disturbance; these measure it."""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..processing.sinefit import fit_fundamental, reference_fit

SQRT2 = float(np.sqrt(2.0))


def rms_pu(x) -> float:
    """RMS in per-unit: a clean nominal sine gives 1.0 (so 0.7 means 70 % of normal)."""
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(np.square(x))) * SQRT2)


def harmonic_spectrum(x, cfg: Config, max_order: int = 15):
    """FFT over a whole number of cycles. Returns (fundamental amplitude, {order: amplitude}) or None."""
    n_cyc = len(x) // cfg.cycle_samples
    if n_cyc < 1:
        return None
    seg = np.asarray(x[: n_cyc * cfg.cycle_samples], dtype=float)
    amp = np.abs(np.fft.rfft(seg)) * 2.0 / len(seg)
    fundamental = amp[n_cyc]                 # the 50 Hz component sits in bin n_cyc
    orders = {h: amp[h * n_cyc] for h in range(2, max_order + 1) if h * n_cyc < len(amp)}
    return fundamental, orders


def thd_percent(x, cfg: Config, max_order: int = 15):
    """Total Harmonic Distortion in %, or None when the segment is shorter than one cycle."""
    spec = harmonic_spectrum(x, cfg, max_order)
    if spec is None or spec[0] <= 1e-9:
        return None
    fund, orders = spec
    return float(np.sqrt(sum(a * a for a in orders.values())) / fund * 100.0)


def harmonic_percent(x, cfg: Config, order: int):
    """Size of one harmonic as % of the fundamental (None if not measurable)."""
    spec = harmonic_spectrum(x, cfg, max(order, 15))
    if spec is None or spec[0] <= 1e-9 or order not in spec[1]:
        return None
    return float(spec[1][order] / spec[0] * 100.0)


def noise_std_pu(x, i0: int, i1: int, cfg: Config):
    """Spread of what is left after removing the fitted 50 Hz sine (p.u.)."""
    if i1 - i0 < cfg.cycle_samples:
        return None
    fit = fit_fundamental(x, i0, i1, cfg)
    resid = np.asarray(x[i0:i1], dtype=float) - fit(np.arange(i0, i1))
    return float(np.std(resid))


def transient_stats(x, i0: int, i1: int, cfg: Config):
    """Peak deviation (p.u.) and ringing frequency (Hz) of a transient."""
    ref = reference_fit(x, i0, i1, cfg)
    resid = np.asarray(x[i0:i1], dtype=float) - ref(np.arange(i0, i1))
    if len(resid) < 4:
        return None, None
    peak = float(np.max(np.abs(resid)))
    nfft = 4096
    spec = np.abs(np.fft.rfft(resid * np.hanning(len(resid)), nfft))
    freq = float(np.argmax(spec[1:]) + 1) * cfg.fs / nfft
    return peak, freq
