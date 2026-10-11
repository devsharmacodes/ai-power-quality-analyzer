"""Train the CNN on the generated windows.

Run:  python -m pqd.models.train --epochs 30
Saves the best model (by validation accuracy) to models/pqd_cnn.pt and a
history file next to it. The test set is used once, at the very end.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..config import CLASSES, DATA_DIR, MODEL_DIR, Config, load_config
from ..evaluation.classification import classification_report, confusion_matrix, format_report
from .cnn import build_model


def load_split(data_dir: Path, split: str):
    X = np.load(data_dir / f"X_{split}.npy")
    y = np.load(data_dir / f"y_{split}.npy")
    return X, y


def _loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X).unsqueeze(1), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
    return total / n, correct / n


def train(data_dir=None, model_path=None, epochs=30, batch_size=128, lr=1e-3,
          device=None, cfg: Config | None = None, verbose=True) -> dict:
    cfg = cfg or load_config()
    data_dir = Path(data_dir or DATA_DIR)
    model_path = Path(model_path or MODEL_DIR / "pqd_cnn.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.seed)

    Xtr, ytr = load_split(data_dir, "train")
    Xva, yva = load_split(data_dir, "val")
    Xte, yte = load_split(data_dir, "test")
    train_loader = _loader(Xtr, ytr, batch_size, True)
    val_loader = _loader(Xva, yva, batch_size, False)

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_val, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, device, optimizer)
        va_loss, va_acc = _run_epoch(model, val_loader, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})
        if va_acc > best_val:
            best_val = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"epoch {epoch:3d}  train acc {tr_acc:.3f}  val acc {va_acc:.3f}  val loss {va_loss:.3f}")

    model.load_state_dict(best_state)
    torch.save({
        "state_dict": best_state,
        "classes": CLASSES,
        "window_samples": cfg.window_samples,
        "fs": cfg.fs,
        "val_acc": best_val,
        "config": asdict(cfg),
    }, model_path)

    # the test set is used once, here
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in _loader(Xte, yte, 512, False):
            preds.append(model(xb.to(device)).argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    cm = confusion_matrix(yte, pred, len(CLASSES))
    report = classification_report(cm, CLASSES)
    if verbose:
        print("\nTEST SET (used once)\n" + format_report(report))

    result = {"best_val_acc": best_val, "test_acc": report["accuracy"],
              "confusion_matrix": cm.tolist(), "history": history}
    model_path.with_name(model_path.stem + "_history.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Train the PQD CNN")
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--config", type=str, default=None)
    a = p.parse_args()
    train(a.data, a.out, a.epochs, a.batch_size, a.lr, cfg=load_config(a.config))


if __name__ == "__main__":
    main()
