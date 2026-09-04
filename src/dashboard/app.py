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

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; }}
  .stApp {{ background-color: {BG}; color: {TEXT}; }}
  section[data-testid="stSidebar"] {{ background-color: {SURFACE}; border-right: 1px solid {BORDER}; }}
  .eyebrow {{ font-family: 'IBM Plex Mono', monospace; font-size: 11px; letter-spacing: 0.12em;
    text-transform: uppercase; color: {TEXT_MUTED}; margin-bottom: 2px; }}
  .ds-card {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 4px; padding: 18px 20px; }}
  .ds-num {{ font-family: 'IBM Plex Mono', monospace; font-size: 30px; font-weight: 600; }}
  .pill {{ display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 11px;
    padding: 2px 8px; border-radius: 3px; }}
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

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown('<div class="ds-card">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Live Data Window</div>', unsafe_allow_html=True)
    try:
        live_df = load_live_window(ticker, cfg)
        st.metric("Rows loaded", len(live_df))
        st.metric("Latest close", f"{live_df['Close'].iloc[-1]:.2f}")
    except Exception as e:
        st.warning("No live data yet — run `python -m src.ingestion.yahoo_ingestor` first.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="ds-card">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Price History</div>', unsafe_allow_html=True)
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=live_df["Date"], y=live_df["Close"], mode="lines",
                                  line=dict(color=TEAL, width=1.3)))
        fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=250,
                           margin=dict(l=0, r=0, t=10, b=0),
                           font=dict(color=TEXT_MUTED),
                           xaxis=dict(gridcolor=BORDER), yaxis=dict(gridcolor=BORDER))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception:
        st.info("Chart will appear once live data has been ingested.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

col3, col4 = st.columns(2)
with col3:
    st.markdown('<div class="eyebrow">Recent Drift Events</div>', unsafe_allow_html=True)
    drift_log_path = resolve_path(cfg["paths"]["drift_log"])
    if drift_log_path.exists():
        events = [json.loads(l) for l in open(drift_log_path)]
        st.dataframe(pd.DataFrame(events[-10:]), use_container_width=True, hide_index=True)
    else:
        st.caption("No drift events logged yet.")

with col4:
    st.markdown('<div class="eyebrow">Retraining History</div>', unsafe_allow_html=True)
    history = get_model_history(cfg)
    if history:
        st.dataframe(pd.DataFrame(history)[["version", "trained_at", "metrics", "promoted"]],
                     use_container_width=True, hide_index=True)
    else:
        st.caption("No models trained yet — run scripts/backfill_history.py then src/forecasting/train.py.")