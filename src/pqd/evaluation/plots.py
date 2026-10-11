"""Plot helpers. CLASS_COLORS is shared with the dashboard so colours match everywhere."""
from __future__ import annotations

import numpy as np

from ..config import Config

CLASS_COLORS = {
    "normal": "#8A8F98",
    "sag": "#E08A00",
    "swell": "#7B4FC9",
    "harmonics": "#13857F",
    "interruption": "#C0392B",
    "transient": "#D6338A",
    "noise": "#5B7DB1",
}


def plot_stream_overview(stream, result: dict, cfg: Config, path, title: str | None = None) -> None:
    """Save a PNG for the slides.

    Top: the disturbed signal, shaded where events were DETECTED, with bars at the
    bottom for the TRUE events (answer key). Bottom: corrected signal vs clean signal.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fs = cfg.fs
    t = np.arange(len(stream.disturbed)) / fs
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    ax1.plot(t, stream.disturbed, lw=0.6, color="#2B2F36")
    lo, hi = float(np.min(stream.disturbed)), float(np.max(stream.disturbed))
    pad = 0.15 * (hi - lo)
    ax1.set_ylim(lo - pad * 1.6, hi + pad)
    for ev in result["events"]:
        ax1.axvspan(ev.start / fs, ev.end / fs, color=CLASS_COLORS[ev.label], alpha=0.22, lw=0)
        ax1.text((ev.start + ev.end) / 2 / fs, hi + pad * 0.3, ev.label, ha="center",
                 fontsize=8, color=CLASS_COLORS[ev.label])
    for e in stream.events:
        ax1.hlines(lo - pad * 1.1, e.start / fs, e.end / fs, color=CLASS_COLORS[e.label], lw=6)
    ax1.set_ylabel("Voltage (p.u.)")
    ax1.set_title(title or "Shaded = detected events, bars at the bottom = true events (answer key)")

    ax2.plot(t, stream.disturbed - stream.clean, lw=0.6, color="#8A8F98", label="error before correction")
    ax2.plot(t, np.asarray(result["corrected"]) - stream.clean, lw=0.8, color="#13857F",
             label="error after correction")
    ax2.set_ylabel("Error vs clean signal (p.u.)")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Correction check: the closer the green line is to zero, the better", fontsize=10)
    ax2.legend(loc="upper right", frameon=False)

    handles = [Patch(color=c, label=n) for n, c in CLASS_COLORS.items() if n != "normal"]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=140)
    plt.close(fig)
