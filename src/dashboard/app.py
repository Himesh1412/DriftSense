# src/dashboard/app.py
"""
DriftSense Dashboard (FR-12)
Run with: streamlit run src/dashboard/app.py
Wired to real modules — falls back to a clear "no data yet" state
rather than fake placeholder numbers if the pipeline hasn't run.
"""
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from src.config_loader import load_config, resolve_path
from src.ingestion.yahoo_ingestor import load_live_window
from src.retraining.registry import load_current_model, get_model_history


st.set_page_config(page_title="DriftSense", layout="wide", page_icon="◆")
cfg = load_config()

BG, SURFACE, BORDER = "#0B0E13", "#12151C", "#232733"
TEXT, TEXT_MUTED = "#E7E9EE", "#8891A5"
TEAL, CORAL, AMBER = "#2FD4C4", "#F9695F", "#F5A623"

# NOTE: keep this <style> block free of blank lines — Streamlit's markdown
# renderer treats a blank line inside it as ending the raw-HTML block early,
# which dumps the remaining CSS onto the page as plain visible text instead
# of applying it as a stylesheet.
#
# .eyebrow: section labels now get real breathing room below them (was a
#   2px margin, which read as "congested" against whatever followed).
# .ds-card / .ds-metric-row / .ds-num: a pure-HTML card, used where the
#   content is plain numbers rather than a live Streamlit widget — it
#   genuinely wraps its contents since it's rendered in one st.markdown
#   call, unlike the old open-div/close-div-in-two-separate-calls pattern,
#   which never actually nested anything (the real source of the gap bug).
# stVerticalBlockBorderWrapper: Streamlit's own st.container(border=True),
#   reskinned to match .ds-card, used wherever the card holds a real widget
#   (chart, dataframe) that can't be flattened into raw HTML.
st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}
  .stApp {{ background-color: {BG}; color: {TEXT}; }}
  section[data-testid="stSidebar"] {{ background-color: {SURFACE}; border-right: 1px solid {BORDER}; }}
  .eyebrow {{ font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.12em;
    text-transform: uppercase; color: {TEXT_MUTED}; margin-bottom: 14px; }}
  .ds-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px; padding: 22px 24px; }}
  .ds-metric-row {{ display: flex; gap: 40px; flex-wrap: wrap; }}
  .ds-metric-label {{ font-size: 13px; color: {TEXT_MUTED}; margin-bottom: 6px; }}
  .ds-num {{ font-family: 'IBM Plex Mono', monospace; font-size: 30px; font-weight: 600; line-height: 1.2; }}
  .ds-warning {{ color: {AMBER}; font-size: 14px; }}
  .pill {{ display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    padding: 2px 8px; border-radius: 3px; }}
  div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {SURFACE}; border: 1px solid {BORDER} !important; border-radius: 6px; }}
  hr {{ margin: 36px 0 !important; border-color: {BORDER}; }}
  #MainMenu, footer, header {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="eyebrow">Configuration</div>', unsafe_allow_html=True)
    ticker = st.text_input("Ticker", value=cfg["ticker"])
    st.caption("Edit config/settings.yaml for threshold changes.")

h1, h2 = st.columns([3, 1])
with h1:
    st.markdown('<div class="eyebrow">DriftSense</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:22px;font-weight:600;">{ticker} — Forecast & Drift Monitor</div>', unsafe_allow_html=True)
with h2:
    model, version = load_current_model(cfg)
    st.markdown('<div class="eyebrow">Current Model</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ds-num">{"v" + str(version) if version else "none yet"}</div>', unsafe_allow_html=True)

st.markdown("---")

# FR-11/FR-12: the live prediction, its confidence, and whether the system
# abstained — this was previously only ever printed to the console by
# run_daily_cycle.py, never surfaced on the dashboard. It reads the last
# line of prediction_log.jsonl, which run_daily_cycle.py now writes on
# every cycle regardless of which branch (ok / abstained / no data / no
# model) it lands in.
st.markdown('<div class="eyebrow">Latest Prediction</div>', unsafe_allow_html=True)
prediction_log_path = resolve_path(cfg["paths"]["prediction_log"])
latest_prediction = None
if prediction_log_path.exists():
    pred_entries = [json.loads(l) for l in open(prediction_log_path)]
    if pred_entries:
        latest_prediction = pred_entries[-1]

if latest_prediction is None:
    st.markdown("""
    <div class="ds-card">
      <div class="ds-warning">No prediction logged yet — run <code>python -m scripts.run_daily_cycle</code> first.</div>
    </div>
    """, unsafe_allow_html=True)
elif latest_prediction["status"] == "no_model":
    st.markdown("""
    <div class="ds-card">
      <div class="ds-warning">No trained model exists yet — run scripts/backfill_history.py then src/forecasting/train.py.</div>
    </div>
    """, unsafe_allow_html=True)
elif latest_prediction["status"] == "insufficient_data":
    st.markdown("""
    <div class="ds-card">
      <div class="ds-warning">Not enough live data yet for a prediction.</div>
    </div>
    """, unsafe_allow_html=True)
else:
    as_of = latest_prediction["date"][:16].replace("T", " ")
    if latest_prediction.get("abstained"):
        st.markdown(f"""
        <div class="ds-card" style="border-color:{AMBER};">
          <div class="ds-metric-row">
            <div>
              <div class="ds-metric-label">Status</div>
              <div class="ds-num" style="color:{AMBER};">ABSTAINED</div>
            </div>
            <div>
              <div class="ds-metric-label">Confidence (below {cfg['confidence']['abstain_below']} threshold)</div>
              <div class="ds-num">{latest_prediction['confidence']:.4f}</div>
            </div>
            <div>
              <div class="ds-metric-label">Model / As of</div>
              <div class="ds-num" style="font-size:16px;">v{latest_prediction['model_version']} &middot; {as_of}</div>
            </div>
          </div>
          <div class="ds-metric-label" style="margin-top:14px;">FR-11: confidence fell below the configured threshold, so no directional call is issued — this is the system correctly recognizing it lacks a reliable signal, not a failure.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        color = TEAL if latest_prediction["direction"] == "UP" else CORAL
        st.markdown(f"""
        <div class="ds-card" style="border-color:{color};">
          <div class="ds-metric-row">
            <div>
              <div class="ds-metric-label">Direction</div>
              <div class="ds-num" style="color:{color};">{latest_prediction['direction']}</div>
            </div>
            <div>
              <div class="ds-metric-label">Confidence</div>
              <div class="ds-num">{latest_prediction['confidence']:.4f}</div>
            </div>
            <div>
              <div class="ds-metric-label">Model / As of</div>
              <div class="ds-num" style="font-size:16px;">v{latest_prediction['model_version']} &middot; {as_of}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns([1, 2], gap="large")
with col1:
    # Rendered as one HTML block so the label and numbers are genuinely
    # nested inside the same card (see .ds-card / .ds-metric-row above).
    try:
        live_df = load_live_window(ticker, cfg)
        st.markdown(f"""
        <div class="ds-card">
          <div class="eyebrow">Live Data Window</div>
          <div class="ds-metric-row">
            <div>
              <div class="ds-metric-label">Rows loaded</div>
              <div class="ds-num">{len(live_df)}</div>
            </div>
            <div>
              <div class="ds-metric-label">Latest close</div>
              <div class="ds-num">{live_df['Close'].iloc[-1]:.2f}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        live_df = None
        st.markdown("""
        <div class="ds-card">
          <div class="eyebrow">Live Data Window</div>
          <div class="ds-warning">No live data yet — run <code>python -m src.ingestion.yahoo_ingestor</code> first.</div>
        </div>
        """, unsafe_allow_html=True)

with col2:
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Price History</div>', unsafe_allow_html=True)
        if live_df is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=live_df["Date"], y=live_df["Close"], mode="lines",
                                      line=dict(color=TEAL, width=1.3)))
            fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=250,
                               margin=dict(l=0, r=0, t=10, b=0),
                               font=dict(color=TEXT_MUTED),
                               xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info("Chart will appear once live data has been ingested.")

st.markdown("---")

col3, col4 = st.columns(2, gap="large")
with col3:
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Recent Drift Events</div>', unsafe_allow_html=True)
        drift_log_path = resolve_path(cfg["paths"]["drift_log"])
        if drift_log_path.exists():
            events = [json.loads(l) for l in open(drift_log_path)]
            st.dataframe(pd.DataFrame(events[-10:]), width="stretch", hide_index=True)
        else:
            st.caption("No drift events logged yet.")

with col4:
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Retraining History</div>', unsafe_allow_html=True)
        history = get_model_history(cfg)
        if history:
            st.dataframe(pd.DataFrame(history)[["version", "trained_at", "metrics", "promoted"]],
                         width="stretch", hide_index=True)
        else:
            st.caption("No models trained yet — run scripts/backfill_history.py then src/forecasting/train.py.")

st.markdown("---")

col5, col6 = st.columns(2, gap="large")
with col5:
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Model Accuracy Over Time</div>', unsafe_allow_html=True)
        # Reuses `history` loaded above (get_model_history) — every promoted
        # *and* rejected candidate the retraining manager has ever produced,
        # plotted against when it was trained, not just its version number.
        if history:
            hist_sorted = sorted(history, key=lambda m: m["trained_at"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[m["trained_at"] for m in hist_sorted],
                y=[m["metrics"]["accuracy"] for m in hist_sorted],
                mode="lines+markers",
                line=dict(color=TEXT_MUTED, width=1.3),
                marker=dict(size=9, color=[TEAL if m["promoted"] else CORAL for m in hist_sorted]),
                text=[f"v{m['version']}" for m in hist_sorted],
                hovertemplate="%{text}<br>%{x}<br>accuracy=%{y:.4f}<extra></extra>",
            ))
            fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=250,
                               margin=dict(l=0, r=0, t=10, b=0), font=dict(color=TEXT_MUTED),
                               xaxis=dict(gridcolor=BORDER),
                               yaxis=dict(gridcolor=BORDER, range=[0, 1], title="accuracy"))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.caption(f":large_blue_circle: promoted &nbsp;&nbsp; :red_circle: rejected (failed the FR-9 benchmark gate)")
        else:
            st.caption("No models trained yet — run scripts/backfill_history.py then src/forecasting/train.py.")

with col6:
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Drift Score Over Time</div>', unsafe_allow_html=True)
        # Reuses the same drift_events.jsonl as "Recent Drift Events" above —
        # this is empty until run_daily_cycle.py actually flags drift on a
        # real run, same honest "no data yet" state as everywhere else here.
        drift_log_path = resolve_path(cfg["paths"]["drift_log"])
        drift_events = [json.loads(l) for l in open(drift_log_path)] if drift_log_path.exists() else []
        if drift_events:
            ddf = pd.DataFrame(drift_events)
            color_map = {"Regime Shift": CORAL, "Anomaly Spike": AMBER}
            marker_colors = [color_map.get(c, TEAL) for c in ddf["classification"]]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ddf["date"], y=ddf["drift_score"], mode="lines+markers",
                                      line=dict(color=TEXT_MUTED, width=1.3),
                                      marker=dict(size=9, color=marker_colors),
                                      text=ddf["classification"],
                                      hovertemplate="%{text}<br>%{x}<br>drift_score=%{y:.4f}<extra></extra>"))
            fig.add_hline(y=cfg["drift"]["threshold"], line_dash="dash", line_color=AMBER)
            fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=250,
                               margin=dict(l=0, r=0, t=10, b=0), font=dict(color=TEXT_MUTED),
                               xaxis=dict(gridcolor=BORDER),
                               yaxis=dict(gridcolor=BORDER, title="drift_score"))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.caption("dashed line = configured drift.threshold")
        else:
            st.caption("No drift events logged yet.")