"""Long continuous signal with disturbances placed at random times.

generate_stream() returns three things: the disturbed signal, the clean signal
(for verification) and the true event list (the answer key; the CNN never sees it).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ..config import CLASSES, DISTURBANCES, Config
from .disturbances import apply_disturbance, event_length_samples

TWO_PI = 2 * np.pi


@dataclass
class Event:
    label: str
    start: int   # sample index
    end: int     # sample index (exclusive)
    params: dict = field(default_factory=dict)

    def to_dict(self, fs: float) -> dict:
        return {
            "label": self.label,
            "start": int(self.start),
            "end": int(self.end),
            "start_s": self.start / fs,
            "end_s": self.end / fs,
            "params": self.params,
        }


@dataclass
class StreamResult:
    disturbed: np.ndarray
    clean: np.ndarray
    events: list
    fs: int
    mode: str
    seed: int

    @property
    def t(self) -> np.ndarray:
        return np.arange(len(self.disturbed)) / self.fs


def generate_stream(
    mode: str = "mixed",
    duration: float = 10.0,
    fs: int | None = None,
    n_events: int = 5,
    severity: float | None = None,
    seed: int | None = None,
    cfg: Config | None = None,
) -> StreamResult:
    """mode: 'mixed' (default, random classes) or one class name ('normal', 'sag', ...)."""
    cfg = cfg or Config()
    if fs is not None and fs != cfg.fs:
        cfg = replace(cfg, fs=int(fs))
    if mode != "mixed" and mode not in CLASSES:
        raise ValueError(f"mode must be 'mixed' or one of {CLASSES}")
    seed = cfg.seed if seed is None else seed
    rng = np.random.default_rng(seed)

    n_total = int(round(duration * cfg.fs))
    if mode == "normal":
        labels = []
    elif mode == "mixed":
        labels = [str(l) for l in rng.choice(DISTURBANCES, size=n_events)]
    else:
        labels = [mode] * n_events
    lengths = [event_length_samples(l, cfg, rng) for l in labels]

    lead = int(round(cfg.lead_s * cfg.fs))
    gap = int(round(cfg.min_gap_s * cfg.fs))

    def slack_for(n):
        return n_total - sum(lengths[:n]) - 2 * lead - max(0, n - 1) * gap

    n = len(labels)
    while n > 0 and slack_for(n) < 0:
        n -= 1
    labels, lengths = labels[:n], lengths[:n]

    # random positions: spread the free time over the n+1 gaps
    starts = []
    if n > 0:
        extras = rng.dirichlet(np.ones(n + 1)) * slack_for(n)
        pos = lead + extras[0]
        for i in range(n):
            s = int(pos)
            starts.append(s)
            pos = s + lengths[i] + gap + extras[i + 1]

    t = np.arange(n_total) / cfg.fs
    phase0 = rng.uniform(0, TWO_PI)
    clean = np.sin(TWO_PI * cfg.f_nominal * t + phase0)
    y = clean.copy()
    events = []
    for label, s, length in zip(labels, starts, lengths):
        e = min(s + length, n_total)
        y, params = apply_disturbance(label, y, s, e, cfg, rng, severity)
        events.append(Event(label, s, e, params))
    y = y + rng.normal(0.0, cfg.base_noise_std, n_total)
    return StreamResult(y, clean, events, cfg.fs, mode, seed)


def save_stream(path, s: StreamResult) -> None:
    np.savez_compressed(
        path,
        disturbed=s.disturbed.astype(np.float32),
        clean=s.clean.astype(np.float32),
        fs=np.int64(s.fs),
        mode=np.str_(s.mode),
        seed=np.int64(s.seed),
        events=np.str_(json.dumps([e.to_dict(s.fs) for e in s.events])),
    )


def load_stream(path) -> StreamResult:
    d = np.load(Path(path), allow_pickle=False)
    events = [Event(e["label"], e["start"], e["end"], e["params"]) for e in json.loads(str(d["events"]))]
    return StreamResult(
        d["disturbed"].astype(float), d["clean"].astype(float), events,
        int(d["fs"]), str(d["mode"]), int(d["seed"]),
    )
