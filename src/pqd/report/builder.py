"""Join the CNN's label, the formulas and the IEEE rules into one report.

CNN says WHAT it is, formulas say HOW MUCH, rules say WHICH STANDARD CATEGORY.
One extra rule is applied on top of the CNN: for sag / swell / interruption the
measured RMS decides the final class, which fixes the sag-vs-interruption mix-up.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from . import ieee1159
from .metrics import harmonic_percent, noise_std_pu, rms_pu, thd_percent, transient_stats

AMPLITUDE = ("sag", "swell", "interruption")


def build_report(signal, events, cfg: Config) -> list[dict]:
    x = np.asarray(signal, dtype=float)
    rows = []
    for k, ev in enumerate(events, start=1):
        i0, i1 = int(ev.start), int(ev.end)
        dur_s = (i1 - i0) / cfg.fs
        label, note = ev.label, ""
        row = {
            "id": k,
            "cnn_label": ev.label,
            "label": label,
            "confidence": round(float(ev.confidence), 3),
            "start_idx": i0,
            "end_idx": i1,
            "start_s": round(i0 / cfg.fs, 4),
            "end_s": round(i1 / cfg.fs, 4),
            "duration_ms": round(dur_s * 1000, 1),
            "duration_cycles": round(dur_s * cfg.f_nominal, 1),
        }
        r = thd = noise = None
        if ev.label in AMPLITUDE:
            trim = max(5, cfg.cycle_samples // 20)   # skip ~1 ms at each edge so edge error cannot bias RMS
            core = x[i0 + trim: i1 - trim] if (i1 - i0) > 8 * trim else x[i0:i1]
            r = rms_pu(core)
            ruled = ieee1159.amplitude_class(r)
            if ruled is None:
                note = "measured RMS is within 0.9-1.1 p.u., below the IEEE 1159 event limits"
            elif ruled != ev.label:
                note = f"CNN said {ev.label}; measured RMS {r:.2f} p.u. gives {ruled} (IEEE rule)"
                label = ruled
            row.update(rms_pu=round(r, 3), rms_v=round(r * cfg.v_rms_nominal, 1),
                       deviation_pct=round(abs(1.0 - r) * 100.0, 1))
        elif ev.label == "harmonics":
            thd = thd_percent(x[i0:i1], cfg)
            row.update(thd_pct=None if thd is None else round(thd, 2),
                       h3_pct=_r(harmonic_percent(x[i0:i1], cfg, 3)),
                       h5_pct=_r(harmonic_percent(x[i0:i1], cfg, 5)),
                       h7_pct=_r(harmonic_percent(x[i0:i1], cfg, 7)))
        elif ev.label == "noise":
            s = noise_std_pu(x, i0, i1, cfg)
            noise = None if s is None else s * 100.0
            row.update(noise_std_pu=_r(s, 4), noise_pct=_r(noise, 2))
        elif ev.label == "transient":
            peak, freq = transient_stats(x, i0, i1, cfg)
            row.update(peak_pu=_r(peak, 3), ring_hz=_r(freq, 0))
        row["label"] = label
        row["ieee_category"] = ieee1159.categorize(
            label, dur_s, cfg.f_nominal, rms_pu=r, thd_pct=thd, noise_pct=noise)
        row["note"] = note
        rows.append(row)
    return rows


def _r(v, nd=2):
    return None if v is None else round(float(v), nd)


def format_report(rows: list[dict]) -> str:
    if not rows:
        return "No disturbance events found."
    lines = []
    for r in rows:
        head = (f"Event {r['id']}: {r['label']} | {r['start_s']:.3f}-{r['end_s']:.3f} s | "
                f"{r['duration_ms']:.0f} ms ({r['duration_cycles']:.1f} cycles) | "
                f"CNN confidence {r['confidence'] * 100:.0f} %")
        extra = []
        if "rms_pu" in r:
            extra.append(f"RMS {r['rms_pu']:.2f} p.u. ({r['rms_v']:.0f} V), deviation {r['deviation_pct']:.0f} %")
        if r.get("thd_pct") is not None:
            extra.append(f"THD {r['thd_pct']:.1f} %, 3rd {r['h3_pct']} %, 5th {r['h5_pct']} %, 7th {r['h7_pct']} %")
        if r.get("noise_pct") is not None:
            extra.append(f"noise {r['noise_pct']:.1f} % of nominal")
        if r.get("peak_pu") is not None:
            extra.append(f"peak {r['peak_pu']:.2f} p.u., ringing about {r['ring_hz']:.0f} Hz")
        extra.append(f"IEEE 1159: {r['ieee_category']}")
        if r["note"]:
            extra.append(f"note: {r['note']}")
        lines.append(head + "\n    " + "\n    ".join(extra))
    return "\n".join(lines)
