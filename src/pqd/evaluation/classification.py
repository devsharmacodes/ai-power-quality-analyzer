"""Accuracy, precision, recall, F1 and the confusion matrix (numpy only)."""
from __future__ import annotations

import numpy as np

from ..config import CLASSES


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)), 1)
    return cm


def classification_report(cm: np.ndarray, class_names=CLASSES) -> dict:
    cm = np.asarray(cm)
    tp = np.diag(cm).astype(float)
    pred_total = cm.sum(axis=0).astype(float)
    true_total = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred_total, out=np.zeros_like(tp), where=pred_total > 0)
    recall = np.divide(tp, true_total, out=np.zeros_like(tp), where=true_total > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    per_class = {
        name: {"precision": float(precision[i]), "recall": float(recall[i]),
               "f1": float(f1[i]), "support": int(true_total[i])}
        for i, name in enumerate(class_names)
    }
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": per_class,
    }


def format_report(rep: dict) -> str:
    lines = [f"{'class':<14}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}"]
    for name, m in rep["per_class"].items():
        lines.append(f"{name:<14}{m['precision']:>10.3f}{m['recall']:>9.3f}{m['f1']:>8.3f}{m['support']:>9d}")
    lines.append(f"{'macro avg':<14}{rep['macro_precision']:>10.3f}{rep['macro_recall']:>9.3f}{rep['macro_f1']:>8.3f}")
    lines.append(f"accuracy: {rep['accuracy']:.3f}")
    return "\n".join(lines)
