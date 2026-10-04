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
    model, version = load_current_model(ticker, cfg)
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
    # Only ever show a genuine live daily-cycle run here, never a backfilled
    # historical entry (scripts/build_prediction_history.py) — otherwise
    # this card could silently present reconstructed history as if it were
    # today's actual live prediction. Backfilled entries still show up
    # further down in the Actual vs. Predicted table, clearly labeled.
    live_entries = [p for p in pred_entries if p.get("source", "live_cycle") == "live_cycle"]
    if live_entries:
        latest_prediction = live_entries[-1]

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
    # The threshold that actually decided this prediction is the model's
    # own calibrated one (train.py's calibrate_abstain_threshold), not the
    # global cfg fallback — look it up from that specific version's
    # metadata so this label is never misleading.
    _threshold_by_version = {m["version"]: m.get("calibrated_abstain_threshold")
                              for m in get_model_history(ticker, cfg)}
    _used_threshold = _threshold_by_version.get(latest_prediction.get("model_version"))
    threshold_label = f"{_used_threshold:.4f}" if _used_threshold is not None else str(cfg["confidence"]["abstain_below"])
    if latest_prediction.get("abstained"):
        st.markdown(f"""
        <div class="ds-card" style="border-color:{AMBER};">
          <div class="ds-metric-row">
            <div>
              <div class="ds-metric-label">Status</div>
              <div class="ds-num" style="color:{AMBER};">ABSTAINED</div>
            </div>
            <div>
              <div class="ds-metric-label">Confidence (below {threshold_label} threshold)</div>
              <div class="ds-num">{latest_prediction['confidence']:.4f}</div>
            </div>
            <div>
              <div class="ds-metric-label">Model / As of</div>
              <div class="ds-num" style="font-size:16px;">v{latest_prediction['model_version']} &middot; {as_of}</div>
            </div>
          </div>
          <div class="ds-metric-label" style="margin-top:14px;">FR-11: confidence fell below this model's own calibrated threshold (bottom {cfg['confidence']['abstain_percentile']}% of its validation-set confidence distribution), so no directional call is issued — this is the system correctly recognizing it lacks a reliable signal, not a failure.</div>
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

# Backtest Evidence: the model's demonstrated real accuracy across multiple
# independently-trained time periods (src/forecasting/backtest.py), not
# just today's single live MC-Dropout reading — added because live
# confidence turned out to cluster narrowly regardless of a model's real
# backtested skill (see calibrate_abstain_threshold in train.py), so this
# is often the more honest evidence of whether the model actually works.
st.markdown('<div class="eyebrow">Backtest Evidence</div>', unsafe_allow_html=True)
backtest_path = resolve_path(cfg["paths"]["backtest_log_dir"]) / f"{ticker}.json"
if backtest_path.exists():
    bt = json.loads(backtest_path.read_text())
    summary = bt.get("summary", {})
    folds = bt.get("folds", [])
    if summary and folds:
        st.markdown(f"""
        <div class="ds-card">
          <div class="ds-metric-row">
            <div>
              <div class="ds-metric-label">Accuracy ({len(folds)}-fold mean)</div>
              <div class="ds-num">{summary.get('accuracy_mean', 0):.1%}</div>
            </div>
            <div>
              <div class="ds-metric-label">Precision (mean)</div>
              <div class="ds-num" style="font-size:22px;">{summary.get('precision_mean', 0):.1%}</div>
            </div>
            <div>
              <div class="ds-metric-label">Recall (mean)</div>
              <div class="ds-num" style="font-size:22px;">{summary.get('recall_mean', 0):.1%}</div>
            </div>
          </div>
          <div class="ds-metric-label" style="margin-top:14px;">Walk-forward backtest across {len(folds)} independently-trained time periods, no lookahead — this is the model's demonstrated accuracy over real history, distinct from today's single live reading (which may abstain when confidence is low that specific day).</div>
        </div>
        """, unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"fold {f['fold']}" for f in folds], y=[f["accuracy"] for f in folds],
                              marker_color=TEAL))
        fig.add_hline(y=0.5, line_dash="dash", line_color=TEXT_MUTED, annotation_text="chance (50%)")
        fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=200,
                           margin=dict(l=0, r=0, t=25, b=0), font=dict(color=TEXT_MUTED),
                           xaxis=dict(gridcolor=BORDER),
                           yaxis=dict(gridcolor=BORDER, title="accuracy", range=[0, 1]))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    else:
        st.caption("Backtest ran but produced no folds — check backtest config.")
else:
    st.caption(f"No backtest results for {ticker} yet — run `python -m scripts.run_backtest` first.")

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
        history = get_model_history(ticker, cfg)
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

st.markdown("---")

# Actual vs. Predicted: for every logged prediction, look up what the price
# actually did the next trading day (once that day's data exists in the
# live window) and show whether the call was right. Abstained predictions
# are shown but excluded from the accuracy count — the system deliberately
# made no call on those. Predictions whose next day hasn't happened yet
# show as "pending", not wrong — that's honest, not a placeholder.
st.markdown('<div class="eyebrow">Actual vs. Predicted</div>', unsafe_allow_html=True)
prediction_log_path = resolve_path(cfg["paths"]["prediction_log"])
ok_predictions = []
if prediction_log_path.exists():
    ok_predictions = [p for p in (json.loads(l) for l in open(prediction_log_path)) if p.get("status") == "ok"]

if ok_predictions and live_df is not None:
    price_by_date = pd.Series(live_df["Close"].values, index=live_df["Date"].astype(str).str[:10].values)
    price_by_date = price_by_date[~price_by_date.index.duplicated(keep="last")]
    known_dates = sorted(price_by_date.index)

    rows = []
    for p in ok_predictions:
        as_of = p.get("as_of_date", p["date"][:10])  # older log entries lack as_of_date
        predicted_dir = p["direction"]
        abstained = bool(p.get("abstained"))

        later_dates = [d for d in known_dates if d > as_of]
        if as_of in price_by_date.index and later_dates:
            next_date = later_dates[0]
            close_as_of = float(price_by_date[as_of])
            close_next = float(price_by_date[next_date])
            pct_change = (close_next / close_as_of - 1) * 100
            actual_dir = "UP" if close_next > close_as_of else "DOWN"
            result = "abstained (no call made)" if abstained else ("correct" if predicted_dir == actual_dir else "wrong")
        else:
            next_date, close_as_of, close_next, pct_change = None, price_by_date.get(as_of), None, None
            actual_dir, result = "pending", "pending"

        rows.append({
            "date": as_of,
            "predicted": "ABSTAINED" if abstained else predicted_dir,
            "price_that_day": round(close_as_of, 2) if close_as_of is not None else None,
            "actual_price_next_day": round(close_next, 2) if close_next is not None else None,
            "result": result,
        })

    rows_sorted = sorted(rows, key=lambda r: r["date"], reverse=True)
    with st.container(border=True):
        st.dataframe(pd.DataFrame(rows_sorted[:30]), width="stretch", hide_index=True)
        if len(rows_sorted) > 30:
            st.caption(f"Showing the most recent 30 of {len(rows_sorted)} logged predictions.")
        resolved = [r for r in rows if r["result"] in ("correct", "wrong")]
        if resolved:
            n_correct = sum(1 for r in resolved if r["result"] == "correct")
            st.caption(f"{n_correct}/{len(resolved)} resolved (non-abstained) predictions correct. "
                       f"Abstained and still-pending predictions are excluded from this count.")
        else:
            st.caption("No predictions have a resolved next trading day yet.")
else:
    st.caption("No predictions logged yet — run scripts/run_daily_cycle.py to build history.")

st.markdown("---")
st.caption(
    "DriftSense is a course/portfolio project, not a financial product. "
    "It does not provide financial advice and must not be used to make "
    "real trading or investment decisions. (SRS §5.5 Safety.)"
)