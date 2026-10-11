"""Noise: wavelet denoising removes the small high-frequency wavelet parts that are noise."""
from __future__ import annotations

import numpy as np
import pywt


def wavelet_denoise(x, i0: int, i1: int, wavelet: str = "db4", level: int = 4):
    y = np.array(x, dtype=float, copy=True)
    seg = y[i0:i1]
    level = max(1, min(level, pywt.dwt_max_level(len(seg), pywt.Wavelet(wavelet).dec_len)))
    coeffs = pywt.wavedec(seg, wavelet, mode="symmetric", level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745          # noise level from the finest details
    thr = sigma * np.sqrt(2.0 * np.log(len(seg)))           # universal threshold
    coeffs[1:] = [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
    y[i0:i1] = pywt.waverec(coeffs, wavelet, mode="symmetric")[: len(seg)]
    return y
