"""Generate the labelled window dataset for the CNN, plus a few long test streams.

Run:  python -m pqd.generation.make_dataset --per-class 2000

Files written to data/generated/:
  X_train.npy y_train.npy  X_val.npy y_val.npy  X_test.npy y_test.npy
  meta.json
  streams/stream_00.npz ...   (long signals with clean copy and true events)
Each window is independent, so the train / val / test splits never share a signal.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..config import CLASSES, DATA_DIR, Config, load_config
from .disturbances import apply_disturbance, event_length_samples
from .stream import generate_stream, save_stream

TWO_PI = 2 * np.pi


def _event_span(label: str, W: int, cfg: Config, rng) -> tuple[int, int]:
    """Where inside a window the event sits (it may cover the whole window or only a part)."""
    if label == "transient":
        L = min(event_length_samples("transient", cfg, rng), W - 1)
        i0 = int(rng.integers(0, W - L + 1))
        return i0, i0 + L
    L = int(rng.uniform(cfg.min_cover, 1.0) * W)
    kind = rng.choice(["full", "prefix", "suffix", "middle"], p=[0.3, 0.25, 0.25, 0.2])
    if kind == "full":
        return 0, W
    if kind == "prefix":
        return 0, L
    if kind == "suffix":
        return W - L, W
    i0 = int(rng.integers(0, W - L + 1))
    return i0, i0 + L


def make_window(label: str, cfg: Config, rng) -> np.ndarray:
    W = cfg.window_samples
    t = np.arange(W) / cfg.fs
    x = np.sin(TWO_PI * cfg.f_nominal * t + rng.uniform(0, TWO_PI))
    if label != "normal":
        i0, i1 = _event_span(label, W, cfg, rng)
        x, _ = apply_disturbance(label, x, i0, i1, cfg, rng)
    x = x + rng.normal(0.0, cfg.base_noise_std, W)
    return x.astype(np.float32)


def make_dataset(per_class=2000, cfg=None, out_dir=None, n_streams=5,
                 val_frac=0.15, test_frac=0.15, verbose=True) -> Path:
    cfg = cfg or load_config()
    out_dir = Path(out_dir or DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    n_val = int(round(per_class * val_frac))
    n_test = int(round(per_class * test_frac))
    n_train = per_class - n_val - n_test
    counts = {"train": n_train, "val": n_val, "test": n_test}

    for split, n in counts.items():
        X = np.empty((n * len(CLASSES), cfg.window_samples), dtype=np.float32)
        y = np.empty(n * len(CLASSES), dtype=np.int64)
        k = 0
        for ci, label in enumerate(CLASSES):
            for _ in range(n):
                X[k] = make_window(label, cfg, rng)
                y[k] = ci
                k += 1
        order = rng.permutation(len(y))
        np.save(out_dir / f"X_{split}.npy", X[order])
        np.save(out_dir / f"y_{split}.npy", y[order])
        if verbose:
            print(f"{split}: {len(y)} windows ({n} per class) shape {X.shape}")

    stream_dir = out_dir / "streams"
    stream_dir.mkdir(exist_ok=True)
    for i in range(n_streams):
        s = generate_stream("mixed", duration=10.0, n_events=int(rng.integers(4, 7)),
                            seed=cfg.seed + 1000 + i, cfg=cfg)
        save_stream(stream_dir / f"stream_{i:02d}.npz", s)
    if verbose and n_streams:
        print(f"saved {n_streams} test streams to {stream_dir}")

    meta = {
        "classes": CLASSES,
        "per_class": {"train": n_train, "val": n_val, "test": n_test},
        "window_samples": cfg.window_samples,
        "fs": cfg.fs,
        "seed": cfg.seed,
        "config": asdict(cfg),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Generate the PQD window dataset")
    p.add_argument("--per-class", type=int, default=2000)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--n-streams", type=int, default=5)
    p.add_argument("--config", type=str, default=None)
    a = p.parse_args()
    make_dataset(a.per_class, load_config(a.config), a.out, a.n_streams)


if __name__ == "__main__":
    main()
