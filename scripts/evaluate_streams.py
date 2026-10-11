#!/usr/bin/env python
"""Test the trained CNN on long continuous streams (the real test of the project).

Window accuracy only says the CNN labels short pieces correctly. This script checks the
whole chain on long signals: are the events found, are start and end times right, does the
report look right, and does the correction bring the signal back to the clean one?

Run from the project folder:
    python scripts/evaluate_streams.py                 # the saved streams in data/generated/streams
    python scripts/evaluate_streams.py --fresh 10      # also 10 brand-new streams (unseen seeds)

Writes to reports/stream_eval/:  summary.json, stream_XX_report.txt, stream_XX.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import pqd  # noqa: F401
except ImportError:  # not installed with pip: use the src folder next to this script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pqd.config import CLASSES, DATA_DIR, DISTURBANCES, MODEL_DIR, PROJECT_ROOT, load_config
from pqd.evaluation.classification import classification_report, confusion_matrix
from pqd.evaluation.classification import format_report as format_classification
from pqd.evaluation.correction import evaluate_correction
from pqd.evaluation.detection import match_events
from pqd.generation.stream import generate_stream, load_stream
from pqd.models.predict import load_model, make_predictor
from pqd.pipeline import analyze
from pqd.processing.windowing import label_windows
from pqd.report.builder import format_report


def _mean(values):
    return float(np.mean(values)) if len(values) else None


def _fmt(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the CNN pipeline on long streams")
    p.add_argument("--model", default=str(MODEL_DIR / "pqd_cnn.pt"))
    p.add_argument("--streams", default=str(DATA_DIR / "streams"), help="folder with stream_*.npz")
    p.add_argument("--fresh", type=int, default=0, help="also generate N new unseen streams")
    p.add_argument("--duration", type=float, default=10.0, help="length of fresh streams (s)")
    p.add_argument("--out", default=str(PROJECT_ROOT / "reports" / "stream_eval"))
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--config", default=None)
    a = p.parse_args()

    cfg = load_config(a.config)
    if not Path(a.model).exists():
        sys.exit(f"Model file not found: {a.model}\nTrain it first:  python -m pqd.models.train --epochs 30")
    model, info = load_model(a.model)
    predictor = make_predictor(model)
    if info.get("fs") not in (None, cfg.fs) or info.get("window_samples") not in (None, cfg.window_samples):
        print(f"WARNING: the model was trained with fs={info.get('fs')} and "
              f"window={info.get('window_samples')} samples, but config.yaml says "
              f"fs={cfg.fs}, window={cfg.window_samples}. Results may be wrong.")

    streams = []
    sdir = Path(a.streams)
    if sdir.exists():
        for f in sorted(sdir.glob("stream_*.npz")):
            streams.append((f.stem, load_stream(f)))
    rng = np.random.default_rng(cfg.seed + 5000)
    for i in range(a.fresh):
        s = generate_stream("mixed", a.duration, n_events=int(rng.integers(4, 7)),
                            seed=cfg.seed + 5000 + i, cfg=cfg)
        streams.append((f"fresh_{i:02d}", s))
    if not streams:
        sys.exit(f"No streams found in {sdir}. Run  python -m pqd.generation.make_dataset  "
                 "or use  --fresh 5")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plots_ok = not a.no_plots
    if plots_ok:
        try:
            from pqd.evaluation.plots import plot_stream_overview
            import matplotlib  # noqa: F401
        except ImportError:
            print("matplotlib not found: skipping PNG figures (pip install matplotlib)")
            plots_ok = False

    agg = {"true": 0, "detected": 0, "matched": 0, "label_correct": 0, "false_alarms": 0}
    per_class = {c: {"true": 0, "matched": 0, "correct": 0, "start_err": [], "end_err": []}
                 for c in DISTURBANCES}
    win_true, win_pred = [], []
    rmse_before, rmse_after, normal_after = [], [], []

    for name, s in streams:
        if s.fs != cfg.fs:
            print(f"skipping {name}: sampling rate {s.fs} differs from the config ({cfg.fs})")
            continue
        res = analyze(s.disturbed, predictor, cfg)
        m = match_events(s.events, res["events"], cfg.fs)
        ver = evaluate_correction(s.clean, s.disturbed, res["corrected"], res["report"], cfg, predictor)

        agg["true"] += m["n_true"]
        agg["detected"] += m["n_detected"]
        agg["matched"] += m["matched"]
        agg["label_correct"] += m["label_correct"]
        agg["false_alarms"] += m["false_alarms"]
        for e in s.events:
            per_class[e.label]["true"] += 1
        for mt in m["matches"]:
            pc = per_class[mt["true_label"]]
            pc["matched"] += 1
            pc["correct"] += int(mt["true_label"] == mt["detected_label"])
            pc["start_err"].append(abs(mt["start_error_ms"]))
            pc["end_err"].append(abs(mt["end_error_ms"]))
        win_true.append(label_windows(s.events, res["starts"], cfg))
        win_pred.append(res["window_labels"])
        rmse_before.append(ver["overall"]["rmse_before"])
        rmse_after.append(ver["overall"]["rmse_after"])
        normal_after += [e["pct_windows_normal_after"] for e in ver["events"]
                         if "pct_windows_normal_after" in e]

        lines = [f"STREAM {name}  (mode {s.mode}, seed {s.seed}, {len(s.disturbed) / s.fs:.1f} s)", "",
                 "TRUE EVENTS (answer key)"]
        for e in s.events:
            lines.append(f"  {e.label:13s} {e.start / cfg.fs:7.3f} - {e.end / cfg.fs:7.3f} s   {e.params}")
        lines += ["", "REPORT", format_report(res["report"]), "", "CORRECTION ACTIONS"]
        lines += [f"  event {a_['id']}: {a_['label']} -> {a_['method']}" for a_ in res["actions"]]
        lines += ["", "VERIFICATION (against the clean signal)"]
        for e in ver["events"]:
            lines.append("  " + ", ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in e.items()))
        lines += ["", f"DETECTION: matched {m['matched']}/{m['n_true']}, missed {m['missed']}, "
                      f"false alarms {m['false_alarms']}, labels correct {m['label_correct']}"]
        (out / f"{name}_report.txt").write_text("\n".join(lines))
        if plots_ok:
            plot_stream_overview(s, res, cfg, out / f"{name}.png", title=f"{name}: {s.mode} stream")
        print(f"{name}: matched {m['matched']}/{m['n_true']}, false alarms {m['false_alarms']}, "
              f"labels correct {m['label_correct']}, "
              f"RMSE {ver['overall']['rmse_before']:.4f} -> {ver['overall']['rmse_after']:.4f} p.u.")

    if not win_true:
        sys.exit("No usable streams were evaluated.")
    cm = confusion_matrix(np.concatenate(win_true), np.concatenate(win_pred), len(CLASSES))
    wrep = classification_report(cm, CLASSES)

    print("\n" + "=" * 70)
    print(f"EVENTS   matched {agg['matched']}/{agg['true']} "
          f"({100 * agg['matched'] / max(agg['true'], 1):.0f} %), "
          f"labels correct {agg['label_correct']}/{agg['matched']}, "
          f"false alarms {agg['false_alarms']}")
    print(f"{'class':<14}{'found':>8}{'label ok':>10}{'start err ms':>14}{'end err ms':>12}")
    for c, d in per_class.items():
        print(f"{c:<14}{d['matched']:>4}/{d['true']:<3}{d['correct']:>8}"
              f"{_fmt(_mean(d['start_err'])):>14}{_fmt(_mean(d['end_err'])):>12}")
    print(f"\nWINDOWS  accuracy {wrep['accuracy']:.3f}  macro F1 {wrep['macro_f1']:.3f}  "
          "(most stream windows are normal, so read the per-class table too)")
    print(format_classification(wrep))
    print(f"\nCORRECTION  mean RMSE vs clean signal: {np.mean(rmse_before):.4f} -> "
          f"{np.mean(rmse_after):.4f} p.u.   "
          f"windows re-classified as normal after correction: {_fmt(_mean(normal_after), 0)} %")

    summary = {
        "streams": len(win_true),
        "events": agg,
        "per_class": {c: {"true": d["true"], "matched": d["matched"], "label_correct": d["correct"],
                          "mean_abs_start_err_ms": _mean(d["start_err"]),
                          "mean_abs_end_err_ms": _mean(d["end_err"])} for c, d in per_class.items()},
        "window_accuracy": wrep["accuracy"],
        "window_macro_f1": wrep["macro_f1"],
        "window_confusion_matrix": cm.tolist(),
        "correction": {"mean_rmse_before": float(np.mean(rmse_before)),
                       "mean_rmse_after": float(np.mean(rmse_after)),
                       "mean_pct_windows_normal_after": _mean(normal_after)},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved reports and figures to {out}")


if __name__ == "__main__":
    main()
