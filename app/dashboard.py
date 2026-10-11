"""Streamlit dashboard: generate a signal, find the disturbances with the CNN, read the
report, and see the corrected signal.

Run from the project folder:
    streamlit run app/dashboard.py

Needs the trained model (python -m pqd.models.train) in models/pqd_cnn.pt.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    import pqd  # noqa: F401
except ImportError:  # not installed with pip: use the src folder next to this app
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pqd.config import CLASSES, DISTURBANCES, MODEL_DIR, load_config
from pqd.evaluation.correction import evaluate_correction
from pqd.evaluation.detection import match_events
from pqd.evaluation.plots import CLASS_COLORS
from pqd.generation.stream import generate_stream
from pqd.pipeline import analyze
from pqd.report.builder import format_report

st.set_page_config(page_title="PQD Analyzer", layout="wide")

SOURCES = {
    "Random mixed stream (default)": "mixed",
    "Normal": "normal",
    "Sag": "sag",
    "Swell": "swell",
    "Harmonics": "harmonics",
    "Interruption": "interruption",
    "Transient": "transient",
    "Noise": "noise",
}

COLUMN_NAMES = {
    "id": "Event", "label": "Class", "start_s": "Start (s)", "end_s": "End (s)",
    "duration_ms": "Duration (ms)", "duration_cycles": "Cycles", "confidence": "CNN confidence",
    "rms_pu": "RMS (p.u.)", "deviation_pct": "Deviation (%)", "thd_pct": "THD (%)",
    "h3_pct": "3rd (%)", "h5_pct": "5th (%)", "h7_pct": "7th (%)", "noise_pct": "Noise (%)",
    "peak_pu": "Peak (p.u.)", "ring_hz": "Ringing (Hz)", "ieee_category": "IEEE 1159 category",
    "note": "Note",
}


# ---------------------------------------------------------------- small helpers
def show_chart(fig) -> None:
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # older Streamlit
        st.plotly_chart(fig, use_container_width=True)


def show_table(df: pd.DataFrame) -> None:
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)


def new_seed() -> None:
    st.session_state["seed"] = random.randint(0, 99999)


# ---------------------------------------------------------------- cached work
@st.cache_resource
def get_cfg():
    return load_config()


@st.cache_resource(show_spinner="Loading the CNN...")
def get_predictor(model_path: str):
    from pqd.models.predict import load_model, make_predictor  # imports PyTorch only here

    model, info = load_model(model_path)
    return make_predictor(model), info


@st.cache_data(show_spinner="Analysing the signal...")
def run_analysis(mode, duration, n_events, severity, seed, model_path):
    cfg = get_cfg()
    predictor, _ = get_predictor(model_path)
    s = generate_stream(mode, float(duration), n_events=int(n_events), severity=severity,
                        seed=int(seed), cfg=cfg)
    res = analyze(s.disturbed, predictor, cfg)
    ver = evaluate_correction(s.clean, s.disturbed, res["corrected"], res["report"], cfg, predictor)
    match = match_events(s.events, res["events"], cfg.fs)
    return s, res, ver, match


# ---------------------------------------------------------------- figures
def signal_figure(s, res, cfg):
    fs = cfg.fs
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22],
                        vertical_spacing=0.06)
    fig.add_trace(go.Scattergl(x=s.t, y=s.disturbed, mode="lines", name="signal",
                               line=dict(width=1, color="#2B2F36"), showlegend=False), row=1, col=1)
    for ev in res["events"]:
        fig.add_vrect(x0=ev.start / fs, x1=ev.end / fs, fillcolor=CLASS_COLORS[ev.label],
                      opacity=0.22, line_width=0, row=1, col=1)
    for name in DISTURBANCES:  # legend entries
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker=dict(size=11, color=CLASS_COLORS[name], symbol="square")),
                      row=1, col=1)
    for e in s.events:  # timeline: true events (answer key)
        fig.add_trace(go.Scatter(x=[e.start / fs, e.end / fs], y=[1, 1], mode="lines",
                                 line=dict(width=12, color=CLASS_COLORS[e.label]), showlegend=False,
                                 hoverinfo="text",
                                 text=f"true {e.label}: {e.start / fs:.3f}-{e.end / fs:.3f} s"),
                      row=2, col=1)
    for ev in res["events"]:  # timeline: detected events
        fig.add_trace(go.Scatter(x=[ev.start / fs, ev.end / fs], y=[0, 0], mode="lines",
                                 line=dict(width=12, color=CLASS_COLORS[ev.label]), showlegend=False,
                                 hoverinfo="text",
                                 text=f"detected {ev.label}: {ev.start / fs:.3f}-{ev.end / fs:.3f} s "
                                      f"({ev.confidence * 100:.0f} % sure)"),
                      row=2, col=1)
    fig.update_yaxes(title_text="Voltage (p.u.)", row=1, col=1)
    fig.update_yaxes(tickvals=[0, 1], ticktext=["detected", "true"], range=[-0.7, 1.7], row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.08, x=0))
    return fig


def correction_figure(s, res, cfg, row):
    fs = cfg.fs
    i0, i1 = row["start_idx"], row["end_idx"]
    pad = int(0.1 * fs)
    a, b = max(0, i0 - pad), min(len(s.disturbed), i1 + pad)
    t = np.arange(a, b) / fs
    fig = go.Figure()
    fig.add_vrect(x0=i0 / fs, x1=i1 / fs, fillcolor=CLASS_COLORS[row["label"]], opacity=0.12,
                  line_width=0)
    fig.add_trace(go.Scatter(x=t, y=s.disturbed[a:b], name="original",
                             line=dict(color="#8A8F98", width=1.5)))
    fig.add_trace(go.Scatter(x=t, y=res["corrected"][a:b], name="corrected",
                             line=dict(color="#13857F", width=2)))
    fig.add_trace(go.Scatter(x=t, y=s.clean[a:b], name="clean (answer key)",
                             line=dict(color="#222222", width=1.2, dash="dash")))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Time (s)",
                      yaxis_title="Voltage (p.u.)", legend=dict(orientation="h", y=1.1, x=0))
    return fig


def spectrum_figure(s, res, cfg, row):
    i0, i1 = row["start_idx"], row["end_idx"]
    n_min = 5 * cfg.cycle_samples
    if i1 - i0 < n_min:  # short event: look at a few cycles around it
        mid = (i0 + i1) // 2
        i0 = max(0, mid - n_min // 2)
        i1 = min(len(s.disturbed), i0 + n_min)

    def spec(x):
        seg = np.asarray(x[i0:i1], dtype=float)
        w = np.hanning(len(seg))
        return np.fft.rfftfreq(len(seg), 1.0 / cfg.fs), np.abs(np.fft.rfft(seg * w)) * 2.0 / np.sum(w)

    f, a_orig = spec(s.disturbed)
    _, a_corr = spec(res["corrected"])
    keep = f <= 2500
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f[keep], y=a_orig[keep], name="original",
                             line=dict(color="#8A8F98", width=1.5)))
    fig.add_trace(go.Scatter(x=f[keep], y=a_corr[keep], name="corrected",
                             line=dict(color="#13857F", width=2)))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Frequency (Hz)",
                      yaxis_title="Amplitude (p.u.)", legend=dict(orientation="h", y=1.12, x=0))
    return fig


def confidence_figure(res, cfg):
    centres = (res["starts"] + cfg.window_samples / 2) / cfg.fs
    fig = go.Figure(go.Heatmap(z=res["window_probs"].T, x=centres, y=CLASSES, zmin=0, zmax=1,
                               colorscale="Blues", colorbar=dict(title="probability")))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Window centre (s)")
    return fig


# ---------------------------------------------------------------- sidebar
cfg = get_cfg()
st.session_state.setdefault("seed", 1)

with st.sidebar:
    st.header("Signal")
    source = st.selectbox("Signal source", list(SOURCES), key="source")
    mode = SOURCES[source]
    duration = st.slider("Duration (s)", 4, 20, 10, key="duration")
    n_events = st.slider("Number of events", 1, 8, 5, key="n_events", disabled=(mode == "normal"))
    random_sev = st.checkbox("Random severity", True, key="random_sev")
    severity = None if random_sev else st.slider("Severity (0 mild, 1 strong)", 0.0, 1.0, 0.5,
                                                 key="severity")
    st.number_input("Seed (same seed = same signal)", min_value=0, step=1, key="seed")
    st.button("Generate new signal", on_click=new_seed)
    st.caption(f"Sampling frequency: {cfg.fs / 1000:g} kHz. It is fixed because the CNN was "
               "trained at this rate.")
    st.header("Model")
    model_path = st.text_input("CNN file", str(MODEL_DIR / "pqd_cnn.pt"), key="model_path")

# ---------------------------------------------------------------- main page
st.title("Power Quality Disturbance Analyzer")
st.caption("A 1-D CNN finds and names disturbances in a continuous signal. Formulas then measure "
           "them, IEEE 1159 rules classify them, and filters correct them.")

if not Path(model_path).exists():
    st.error(f"Model file not found: {model_path}")
    st.code("python -m pqd.models.train --epochs 30")
    st.stop()
try:
    get_predictor(model_path)
except ImportError:
    st.error("PyTorch is not installed. Run:  pip install torch")
    st.stop()
except Exception as exc:  # wrong file, version mismatch, ...
    st.error(f"Could not load the model: {exc}")
    st.stop()

s, res, ver, match = run_analysis(mode, duration, n_events, severity, st.session_state["seed"],
                                  model_path)
report = res["report"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Signal length", f"{len(s.disturbed) / s.fs:.0f} s")
c2.metric("Events found", len(res["events"]))
c3.metric("True events (answer key)", match["n_true"])
c4.metric("Found correctly", f"{match['matched']} / {match['n_true']}" if match["n_true"] else "-")

tab_sig, tab_rep, tab_cor, tab_conf = st.tabs(
    ["Signal and events", "Report", "Correction", "CNN confidence"])

with tab_sig:
    show_chart(signal_figure(s, res, cfg))
    st.caption("Top: the signal, shaded where the CNN found an event. Bottom: when the events "
               "really are (answer key, true) and when the CNN found them (detected).")

with tab_rep:
    if not report:
        st.info("No disturbance events were found in this signal.")
    else:
        df = pd.DataFrame(report)
        df = df[[c for c in COLUMN_NAMES if c in df.columns]].dropna(axis=1, how="all")
        show_table(df.rename(columns=COLUMN_NAMES))
        with st.expander("Report as text"):
            st.code(format_report(report), language=None)
    st.subheader("Detection check against the answer key")
    st.caption("Only possible because this signal was generated, so the true events are known.")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Missed events", match["missed"])
    d2.metric("False alarms", match["false_alarms"])
    d3.metric("Mean start error", "-" if match["mean_abs_start_error_ms"] is None
              else f"{match['mean_abs_start_error_ms']:.1f} ms")
    d4.metric("Mean end error", "-" if match["mean_abs_end_error_ms"] is None
              else f"{match['mean_abs_end_error_ms']:.1f} ms")
    if match["matches"]:
        show_table(pd.DataFrame(match["matches"]))

with tab_cor:
    o = ver["overall"]
    k1, k2 = st.columns(2)
    k1.metric("Error vs clean signal, before", f"{o['rmse_before']:.4f} p.u.")
    k2.metric("Error vs clean signal, after", f"{o['rmse_after']:.4f} p.u.",
              delta=f"{o['rmse_after'] - o['rmse_before']:.4f}", delta_color="inverse")
    if not report:
        st.info("Nothing to correct.")
    else:
        actions = {a["id"]: a["method"] for a in res["actions"]}
        vdf = pd.DataFrame(ver["events"])
        vdf["method"] = vdf["id"].map(actions)
        show_table(vdf.dropna(axis=1, how="all").round(4))
        st.caption("Interruption correction is only an estimate: the real voltage information "
                   "is gone, so the sine is rebuilt from the cycles before the event.")
        labels = [f"Event {r['id']}: {r['label']} ({r['start_s']:.2f} to {r['end_s']:.2f} s)"
                  for r in report]
        pick = st.selectbox("Look closely at an event", range(len(report)),
                            format_func=lambda i: labels[i], key="event_pick")
        show_chart(correction_figure(s, res, cfg, report[pick]))
        st.markdown("**Spectrum of the same stretch** (how much of each frequency is present)")
        show_chart(spectrum_figure(s, res, cfg, report[pick]))

with tab_conf:
    show_chart(confidence_figure(res, cfg))
    st.caption("How sure the CNN is about each class for every window. Dark = sure. "
               "The CNN only sees the signal, never the answer key.")

st.divider()
st.caption("Signals here are generated (synthetic). Good results do not prove the same "
           "performance on real measured signals.")
