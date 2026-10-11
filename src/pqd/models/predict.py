"""Load the trained CNN and label windows of a signal."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..config import MODEL_DIR, Config
from ..processing.windowing import sliding_windows
from .cnn import build_model


def load_model(path=None, device=None):
    """Return (model, checkpoint info)."""
    path = Path(path or MODEL_DIR / "pqd_cnn.pt")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # older PyTorch without the weights_only argument
        ckpt = torch.load(path, map_location=device)
    model = build_model(len(ckpt["classes"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, {k: v for k, v in ckpt.items() if k != "state_dict"}


def predict_windows(model, windows, device=None, batch_size: int = 512):
    """windows: (n, samples). Returns (class index per window, probabilities (n, classes))."""
    device = torch.device(device or next(model.parameters()).device)
    model.eval()
    probs = []
    X = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32)).unsqueeze(1)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            logits = model(X[i:i + batch_size].to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.concatenate(probs)
    return probs.argmax(1), probs


def make_predictor(model, device=None):
    """A function windows -> (labels, probs). The pipeline accepts any such function."""
    return lambda windows: predict_windows(model, windows, device)


def classify_stream(model, signal, cfg: Config, device=None):
    """Slide a window over a stream and classify every window."""
    windows, starts = sliding_windows(signal, cfg.window_samples, cfg.step_samples)
    pred, probs = predict_windows(model, windows, device)
    return starts, pred, probs
