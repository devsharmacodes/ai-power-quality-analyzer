"""Rule table that maps measured values to IEEE 1159 style categories.

NOTE: the limits below are written from memory of IEEE Std 1159 (power quality
monitoring). Check them against your own copy of the standard before quoting
them in the final report.
"""
from __future__ import annotations


def amplitude_class(rms_pu: float):
    """Class from the measured RMS alone. None means 'within 0.9-1.1 p.u.' (no event limit crossed)."""
    if rms_pu < 0.1:
        return "interruption"
    if rms_pu < 0.9:
        return "sag"
    if rms_pu > 1.1:
        return "swell"
    return None


def duration_class(cycles: float, seconds: float) -> str:
    if cycles < 0.5:
        return "sub-cycle"
    if cycles <= 30:
        return "instantaneous"
    if seconds <= 3:
        return "momentary"
    if seconds <= 60:
        return "temporary"
    return "sustained"


_SWELL_LIMIT = {"instantaneous": 1.8, "momentary": 1.4, "temporary": 1.2}


def categorize(label, duration_s, f0, rms_pu=None, thd_pct=None, noise_pct=None):
    cycles = duration_s * f0
    dur = duration_class(cycles, duration_s)
    if label == "sag":
        if rms_pu is not None and 0.1 <= rms_pu < 0.9:
            return f"{dur} sag (0.1-0.9 p.u.)"
        return "sag-like event outside the IEEE 1159 sag range (0.1-0.9 p.u.)"
    if label == "swell":
        limit = _SWELL_LIMIT.get(dur)
        if rms_pu is not None and rms_pu > 1.1 and (limit is None or rms_pu <= limit):
            return f"{dur} swell (1.1-{limit} p.u.)" if limit else f"{dur} swell"
        return "swell-like event outside the typical IEEE 1159 swell range"
    if label == "interruption":
        kind = "momentary" if duration_s <= 3 else "temporary" if duration_s <= 60 else "sustained"
        return f"{kind} interruption (below 0.1 p.u.)"
    if label == "harmonics":
        if thd_pct is not None and thd_pct > 20:
            return "harmonic distortion above the typical IEEE 1159 range (0-20 %)"
        return "harmonic distortion (steady state, typical 0-20 %)"
    if label == "transient":
        ms = duration_s * 1000
        note = "" if 0.3 <= ms <= 50 else " (duration outside 0.3-50 ms)"
        return "low-frequency oscillatory transient (below 5 kHz)" + note
    if label == "noise":
        if noise_pct is not None and noise_pct > 1.0:
            return f"noise above the typical IEEE 1159 range (0-1 %), measured {noise_pct:.1f} %"
        return "noise (typical 0-1 %)"
    return "unclassified"
