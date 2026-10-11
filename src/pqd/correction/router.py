"""Pick the right fix for each event from its (final) class label."""
from __future__ import annotations

import numpy as np

from ..config import Config
from .denoise import wavelet_denoise
from .gain import gain_correct
from .notch import notch_correct
from .rebuild import rebuild_sine


def correct_signal(x, report: list[dict], cfg: Config):
    """Return (corrected signal, list of what was done to each event)."""
    y = np.array(x, dtype=float, copy=True)
    actions = []
    for row in report:
        label, i0, i1 = row["label"], row["start_idx"], row["end_idx"]
        if label in ("sag", "swell"):
            rms = row.get("rms_pu")
            if rms is None:
                actions.append({"id": row["id"], "label": label, "method": "skipped (no RMS value)"})
                continue
            y = gain_correct(y, i0, i1, rms)
            method = "gain correction (nominal / measured RMS)"
        elif label == "interruption":
            y = rebuild_sine(y, i0, i1, cfg)
            method = "sine rebuilt from nearby cycles (an estimate)"
        elif label == "harmonics":
            y = notch_correct(y, i0, i1, cfg)
            method = "notch filters at the 3rd, 5th and 7th harmonics"
        elif label == "noise":
            y = wavelet_denoise(y, i0, i1)
            method = "wavelet denoising"
        elif label == "transient":
            y = rebuild_sine(y, i0, i1, cfg, pad=int(0.003 * cfg.fs))
            method = "burst replaced by fitted sine"
        else:
            method = "no correction"
        actions.append({"id": row["id"], "label": label, "method": method})
    return y, actions
