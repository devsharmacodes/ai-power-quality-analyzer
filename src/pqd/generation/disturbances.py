"""Disturbance models.

All signals are per-unit (p.u.): amplitude 1.0 is the nominal peak voltage
(325 V for a 230 V RMS supply), so a clean sine has RMS 0.7071 p.u.
Each model changes the samples x[i0:i1] of a signal and returns its parameters.
"""
from __future__ import annotations

import numpy as np

from ..config import DISTURBANCES, Config

TWO_PI = 2 * np.pi


def event_length_samples(label: str, cfg: Config, rng: np.random.Generator) -> int:
    """Random event length (in samples) for a class."""
    if label == "transient":
        lo, hi = cfg.transient_ms
        return max(8, int(round(rng.uniform(lo, hi) * cfg.fs / 1000.0)))
    lo, hi = cfg.event_cycles[label]
    return int(round(rng.uniform(lo, hi) * cfg.cycle_samples))


def _sag(y, i0, i1, cfg, rng, sev):
    depth = 0.12 + 0.73 * sev            # residual 0.15 .. 0.88 p.u.
    y[i0:i1] *= 1.0 - depth
    return {"residual_pu": round(1.0 - depth, 3)}


def _swell(y, i0, i1, cfg, rng, sev):
    gain = 1.1 + 0.7 * sev               # 1.1 .. 1.8 p.u.
    y[i0:i1] *= gain
    return {"gain_pu": round(gain, 3)}


def _interruption(y, i0, i1, cfg, rng, sev):
    residual = 0.09 * (1.0 - sev)        # below 0.1 p.u.
    y[i0:i1] *= residual
    return {"residual_pu": round(residual, 3)}


def _harmonics(y, i0, i1, cfg, rng, sev):
    s = 0.05 + 0.13 * sev
    amps = {3: s * rng.uniform(0.7, 1.0), 5: s * rng.uniform(0.5, 0.9), 7: s * rng.uniform(0.2, 0.6)}
    t = np.arange(i0, i1) / cfg.fs
    add = np.zeros(i1 - i0)
    used = {}
    for h, a in amps.items():
        if h * cfg.f_nominal < cfg.fs / 2:
            add += a * np.sin(TWO_PI * h * cfg.f_nominal * t + rng.uniform(0, TWO_PI))
            used[h] = a
    y[i0:i1] += add
    thd = float(np.sqrt(sum(a * a for a in used.values())) * 100.0)
    return {"thd_pct": round(thd, 2), "orders": sorted(used)}


def _transient(y, i0, i1, cfg, rng, sev):
    f_hi = min(2500.0, 0.45 * cfg.fs)
    f_lo = min(800.0, 0.5 * f_hi)
    f = rng.uniform(f_lo, f_hi)
    amp = 0.3 + 0.9 * sev
    n = i1 - i0
    tt = np.arange(n) / cfg.fs
    tau = max(n / cfg.fs / 4.0, 1.0 / cfg.fs)
    y[i0:i1] += amp * np.exp(-tt / tau) * np.sin(TWO_PI * f * tt + rng.uniform(0, TWO_PI))
    return {"freq_hz": round(float(f), 1), "peak_pu": round(float(amp), 3)}


def _noise(y, i0, i1, cfg, rng, sev):
    sigma = 0.04 + 0.12 * sev
    y[i0:i1] += rng.normal(0.0, sigma, i1 - i0)
    return {"noise_std_pu": round(float(sigma), 3)}


_FUNCS = {
    "sag": _sag,
    "swell": _swell,
    "harmonics": _harmonics,
    "interruption": _interruption,
    "transient": _transient,
    "noise": _noise,
}
assert set(_FUNCS) == set(DISTURBANCES)


def apply_disturbance(label, x, i0, i1, cfg, rng, severity=None):
    """Return (modified copy of x, parameters). severity is 0..1, random when None."""
    if label not in _FUNCS:
        raise ValueError(f"unknown disturbance '{label}', choose from {sorted(_FUNCS)}")
    sev = float(rng.random()) if severity is None else float(np.clip(severity, 0.0, 1.0))
    y = np.array(x, dtype=float, copy=True)
    params = _FUNCS[label](y, i0, i1, cfg, rng, sev)
    params["severity"] = round(sev, 3)
    return y, params
