# src/dashboard/app.py
"""
DriftSense Dashboard (FR-12)
Run with: streamlit run src/dashboard/app.py
Wired to real modules — falls back to a clear "no data yet" state
rather than fake placeholder numbers if the pipeline hasn't run.

Layout: a top bar, a watchlist of every configured stock on the left (click one to
switch), and for the selected stock a price chart with the model's next-day call,
followed by the evidence / drift / retraining / actual-vs-predicted sections.
"""
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.config_loader import load_config, resolve_path
from src.drift.detector import compute_drift_score
from src.ingestion.yahoo_ingestor import load_live_window
from src.retraining.manager import assemble_training_data
from src.retraining.registry import load_current_model, get_model_history
from src.simulation.launcher import simulator_running, simulator_url, start_simulator
from src.dashboard.theme import (ACCENT, ACCENT_LINE, ACCENT_SOFT, AMBER, CORAL, GRID, INK, SURFACE, TEAL,
                                 TEXT_MUTED, apply_theme)

st.set_page_config(page_title="DriftSense", layout="wide", page_icon="◆", initial_sidebar_state="collapsed")
cfg = load_config()
apply_theme()

# ───────────────────────── live refresh (NFR-4) ─────────────────────────
# Streamlit only redraws when something makes it rerun. This small fragment checks the log files every few
# seconds and reruns the page when a new prediction, drift alert or retraining entry has been written, so the
# dashboard follows a daily cycle within the SRS's 10 seconds without anyone pressing refresh.
_WATCHED = [cfg["paths"][k] for k in ("prediction_log", "drift_log", "retrain_log")]


@st.fragment(run_every=cfg.get("dashboard", {}).get("refresh_check_seconds", 5))
def _watch_logs():
    stamp = tuple(resolve_path(p).stat().st_mtime if resolve_path(p).exists() else 0 for p in _WATCHED)
    if st.session_state.setdefault("_log_stamp", stamp) != stamp:
        st.session_state["_log_stamp"] = stamp
        st.rerun(scope="app")


_watch_logs()

# ───────────────────────── which stock ─────────────────────────
GROUPS = {"nasdaq": "Nasdaq", "india": "NSE"}
all_tickers = [t for g in cfg["tickers"].values() for t in g]
ticker = st.query_params.get("ticker", cfg["ticker"])

# The nav's "Simulation engine" pill points at ?start_sim=1 when the engine isn't running yet.
if st.query_params.get("start_sim"):
    with st.spinner("Starting the Simulation Engine…"):
        start_simulator(cfg)
    st.query_params.pop("start_sim", None)
    st.query_params["ticker"] = ticker
    st.rerun()


def style_fig(fig, height, **yaxis):
    fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
                      margin=dict(l=0, r=0, t=10, b=0), font=dict(color=TEXT_MUTED, family="IBM Plex Mono"),
                      xaxis=dict(gridcolor=GRID, linecolor=GRID), yaxis=dict(gridcolor=GRID, **yaxis))
    return fig


@st.cache_data(ttl=120)
def watchlist_quotes(tickers: tuple, live_dir: str) -> dict:
    """Last close and day change per ticker from the saved live windows (no network)."""
    quotes = {}
    for t in tickers:
        try:
            closes = pd.read_csv(Path(live_dir) / f"{t}_live.csv", usecols=["Close"])["Close"].dropna().tail(2).tolist()
            quotes[t] = (closes[-1], closes[-1] / closes[-2] - 1)
        except Exception:
            quotes[t] = None
    return quotes


model, version = load_current_model(ticker, cfg)

# ───────────────────────── data for the selected stock ─────────────────────────
live_df, hist_df, prices, drift_result = None, None, None, None
try:
    live_df = load_live_window(ticker, cfg)
    snap = resolve_path(cfg["paths"]["snapshot_dir"]) / f"{ticker}_historical.csv"
    hist_df = pd.read_csv(snap) if snap.exists() else None
    prices = assemble_training_data(hist_df, live_df) if hist_df is not None else live_df
    prices = prices.assign(Date=pd.to_datetime(prices["Date"])).reset_index(drop=True)
    if hist_df is not None:
        drift_result = compute_drift_score(hist_df, live_df)
except Exception:
    live_df = None

# the latest genuine live prediction for this stock (never a backfilled one — see build_prediction_history.py)
prediction_log_path = resolve_path(cfg["paths"]["prediction_log"])
latest_prediction = None
if prediction_log_path.exists():
    live_entries = [p for p in (json.loads(l) for l in open(prediction_log_path))
                    if p.get("source", "live_cycle") == "live_cycle" and p.get("ticker") == ticker]
    if live_entries:
        latest_prediction = live_entries[-1]

drift_flagged = bool(drift_result and drift_result["drift_score"] > cfg["drift"]["threshold"])

# ───────────────────────── top bar ─────────────────────────
if simulator_running(cfg):
    sim_pill = f'<a class="pill" href="{simulator_url(cfg)}" target="_blank">Simulation engine ↗</a>'
else:
    sim_pill = f'<a class="pill" href="?ticker={ticker}&start_sim=1" target="_self">Start simulation engine</a>'
status_tag = ('<span class="tag warn">Drift flagged</span>' if drift_flagged
              else '<span class="tag">Pipeline healthy</span>')
st.markdown(f"""
<div class="nav">
  <div class="left">
    <div class="brand">DriftSense</div>
    <div><span class="pill on">Signals</span><a class="pill" href="#evidence" target="_self">Evidence</a><a class="pill" href="#drift" target="_self">Drift</a><a class="pill" href="#calls" target="_self">Calls</a>{sim_pill}</div>
  </div>
  <div class="right">{status_tag}<span class="tag">Updated {datetime.now():%H:%M:%S}</span><span class="tag">Model {"v" + str(version) if version else "none yet"}</span></div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 2.7], gap="small")

# ───────────────────────── watchlist ─────────────────────────
with left:
    quotes = watchlist_quotes(tuple(all_tickers), str(resolve_path(cfg["paths"]["live_dir"])))
    latest_call = {}  # most recent genuine live call per ticker
    if prediction_log_path.exists():
        for p in (json.loads(l) for l in open(prediction_log_path)):
            if p.get("source", "live_cycle") == "live_cycle" and p.get("status") == "ok":
                latest_call[p["ticker"]] = p
    rows_html = ""
    for key, label in GROUPS.items():
        rows_html += f'<div class="wl-group">{label}</div>'
        for t in cfg["tickers"].get(key, []):
            q = quotes.get(t)
            px, chg = (f"{q[0]:,.2f}", f"{q[1]:+.1%}") if q else ("—", "")
            color = (TEAL if q and q[1] >= 0 else CORAL) if q else TEXT_MUTED
            sel = " sel" if t == ticker else ""
            sym = t.partition(".")[0]
            c = latest_call.get(t)
            if c is None:
                sub, sub_color = "No call yet", TEXT_MUTED
            elif c.get("abstained"):
                sub, sub_color = "Abstained", AMBER
            else:
                up = c["direction"] == "UP"
                sub, sub_color = f'{"▲ Up" if up else "▼ Down"} · {c["confidence"]:.2f}', (TEAL if up else CORAL)
            rows_html += (f'<a class="wl-row{sel}" href="?ticker={t}" target="_self">'
                          f'<div><div class="wl-sym">{sym}</div><div class="wl-sub" style="color:{sub_color};">{sub}</div></div>'
                          f'<div><div class="wl-px">{px}</div><div class="wl-chg" style="color:{color};">{chg}</div></div></a>')
    st.markdown(f'<div class="wl"><div class="eyebrow">Watchlist · {len(all_tickers)}</div>'
                f'<div class="wl-scroll">{rows_html}</div></div>', unsafe_allow_html=True)
    typed = st.text_input("Search ticker", placeholder="Any Yahoo ticker, e.g. TSLA", label_visibility="collapsed")
    if typed.strip() and typed.strip().upper() != ticker:
        st.query_params["ticker"] = typed.strip().upper()
        st.rerun()

# ───────────────────────── main card: price + chart + the call ─────────────────────────
with right:
    with st.container(border=True):
        if prices is None:
            st.markdown(f'<div class="sub">{ticker}</div>', unsafe_allow_html=True)
            st.warning("No price data for this ticker yet — run `python -m scripts.onboard_tickers "
                       f"{ticker}` to set it up.")
        else:
            last_close, prev_close = float(prices["Close"].iloc[-1]), float(prices["Close"].iloc[-2])
            day_chg = last_close - prev_close
            chg_color = TEAL if day_chg >= 0 else CORAL
            top_l, top_r = st.columns([3, 1.6])
            with top_l:
                st.markdown(f'<div class="sub">{ticker} · as of {prices["Date"].iloc[-1]:%d %b %Y}</div>'
                            f'<div><span class="bigpx">{last_close:,.2f}</span>'
                            f'<span class="chg" style="color:{chg_color};">{day_chg:+,.2f} ({day_chg / prev_close:+.1%})</span></div>',
                            unsafe_allow_html=True)
            with top_r:
                span = st.segmented_control("Range", ["1M", "3M", "1Y", "All"], default="3M",
                                            label_visibility="collapsed") or "3M"

            n_rows = {"1M": 21, "3M": 63, "1Y": 252, "All": len(prices)}[span]
            view = prices.tail(n_rows)
            dates = view["Date"].dt.to_pydatetime()
            last_x = dates[-1]
            end_x = last_x + (dates[-1] - dates[0]) * 0.2

            # what the model says about the NEXT day (direction only — DriftSense does not forecast a price)
            if latest_prediction is None or latest_prediction.get("status") != "ok":
                call_text, call_color, call_conf = "No call", TEXT_MUTED, None
            elif latest_prediction.get("abstained"):
                call_text, call_color, call_conf = "Abstained", AMBER, latest_prediction["confidence"]
            elif latest_prediction["direction"] == "UP":
                call_text, call_color, call_conf = "▲ UP", TEAL, latest_prediction["confidence"]
            else:
                call_text, call_color, call_conf = "▼ DOWN", CORAL, latest_prediction["confidence"]

            fig = go.Figure()
            fig.add_shape(type="rect", x0=last_x, x1=end_x, y0=0, y1=1, yref="paper", fillcolor=ACCENT_SOFT,
                          line_width=0, layer="below")
            fig.add_shape(type="line", x0=last_x, x1=last_x, y0=0, y1=1, yref="paper",
                          line=dict(color=ACCENT_LINE, dash="dash", width=1.4))
            fig.add_trace(go.Scatter(x=dates, y=view["Close"], mode="lines", line=dict(color=INK, width=2.2),
                                     hovertemplate="%{x|%d %b %Y}<br>%{y:,.2f}<extra></extra>", showlegend=False))
            label = call_text + (f"<br><span style='font-size:12px'>confidence {call_conf:.2f}</span>" if call_conf is not None else "")
            fig.add_annotation(x=last_x + (end_x - last_x) / 2, y=0.5, yref="paper", text=label, showarrow=False,
                               font=dict(size=18, color=call_color))
            fig.add_annotation(x=0, xref="paper", y=1.0, yref="paper", yshift=14, text="Price history", showarrow=False,
                               xanchor="left", font=dict(size=12, color=TEXT_MUTED))
            fig.add_annotation(x=1, xref="paper", y=1.0, yref="paper", yshift=14, text="Next-day call", showarrow=False,
                               xanchor="right", font=dict(size=12, color=TEXT_MUTED))
            style_fig(fig, 330)
            fig.update_layout(margin=dict(l=44, r=12, t=34, b=24), xaxis=dict(range=[dates[0], end_x]),
                              plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

            # the four numbers that matter, under the chart
            threshold = float(model.abstain_threshold.item()) if model is not None else None
            summary_bt = {}
            bt_path = resolve_path(cfg["paths"]["backtest_log_dir"]) / f"{ticker}.json"
            if bt_path.exists():
                summary_bt = json.loads(bt_path.read_text()).get("summary", {})
            drift_text = f"{drift_result['drift_score']:.2f}" if drift_result else "—"   # (nested same-quote f-strings
            acc_text = f"{summary_bt['accuracy_mean']:.1%}" if summary_bt else "—"       #  would need Python 3.12; SRS says 3.10+)
            t1, t2, t3, t4 = st.columns(4)
            tile_kind = {"▲ UP": "up", "▼ DOWN": "down", "Abstained": "warn"}.get(call_text, "")
            t1.markdown(f'<div class="tile {tile_kind}"><div class="ds-metric-label">Next-day call</div>'
                        f'<div class="ds-num" style="color:{call_color};">{call_text}</div></div>', unsafe_allow_html=True)
            t2.markdown(f'<div class="tile"><div class="ds-metric-label">Confidence'
                        f'{f" · abstains < {threshold:.2f}" if threshold is not None else ""}</div>'
                        f'<div class="ds-num">{f"{call_conf:.2f}" if call_conf is not None else "—"}</div></div>',
                        unsafe_allow_html=True)
            t3.markdown(f'<div class="tile"><div class="ds-metric-label">Drift score · limit {cfg["drift"]["threshold"]}</div>'
                        f'<div class="ds-num" style="color:{AMBER if drift_flagged else INK};">'
                        f'{drift_text}</div></div>', unsafe_allow_html=True)
            t4.markdown(f'<div class="tile"><div class="ds-metric-label">Backtest accuracy</div>'
                        f'<div class="ds-num">{acc_text}</div></div>',
                        unsafe_allow_html=True)
            if latest_prediction is None:
                st.caption("No live prediction yet for this stock — run `python -m scripts.run_daily_cycle` "
                           "(it uses the ticker set in config/settings.yaml).")
            elif latest_prediction.get("abstained"):
                st.caption("Abstained: confidence fell below this model's own calibrated threshold, so no call is "
                           "issued. That is the system recognising it has no reliable signal, not a failure.")

# ───────────────────────── Backtest evidence ─────────────────────────
# The model's demonstrated accuracy across several independently-trained time periods
# (src/forecasting/backtest.py), not today's single live MC-Dropout reading — live
# confidence clusters narrowly regardless of real skill, so this is the honest evidence.
with st.container(border=True):
    st.markdown('<div class="eyebrow" id="evidence">Backtest evidence</div>', unsafe_allow_html=True)
    backtest_path = resolve_path(cfg["paths"]["backtest_log_dir"]) / f"{ticker}.json"
    if backtest_path.exists():
        bt = json.loads(backtest_path.read_text())
        summary, folds = bt.get("summary", {}), bt.get("folds", [])
        if summary and folds:
            n_calls = sum(f["n_test"] for f in folds)
            st.markdown(f"""
            <div class="ds-metric-row">
              <div><div class="ds-metric-label">Accuracy ({len(folds)}-fold mean)</div><div class="ds-num">{summary.get('accuracy_mean', 0):.1%}</div></div>
              <div><div class="ds-metric-label">Precision (mean)</div><div class="ds-num">{summary.get('precision_mean', 0):.1%}</div></div>
              <div><div class="ds-metric-label">Recall (mean)</div><div class="ds-num">{summary.get('recall_mean', 0):.1%}</div></div>
              <div><div class="ds-metric-label">Predictions scored</div><div class="ds-num">{n_calls}</div></div>
            </div>
            <div class="ds-metric-label" style="margin-top:14px;">Walk-forward backtest across {len(folds)} independently-trained time periods, no lookahead. This is the model's demonstrated accuracy over real history, distinct from today's single live reading (which may abstain when confidence is low that day). Around 50% is a coin flip.</div>
            """, unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[f"fold {f['fold']}" for f in folds], y=[f["accuracy"] for f in folds],
                                 marker_color=ACCENT))
            fig.add_hline(y=0.5, line_dash="dash", line_color=TEXT_MUTED, annotation_text="chance (50%)")
            style_fig(fig, 200, title="accuracy", range=[0, 1])
            fig.update_layout(margin=dict(l=0, r=0, t=25, b=0))
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            st.caption("Backtest ran but produced no folds — check backtest config.")
    else:
        st.caption(f"No backtest results for {ticker} yet — run `python -m scripts.run_backtest {ticker}` first.")

# ───────────────────────── Drift ─────────────────────────
col3, col4 = st.columns(2, gap="small")
with col3:
    with st.container(border=True):
        st.markdown('<div class="eyebrow" id="drift">Recent drift events</div>', unsafe_allow_html=True)
        drift_log_path = resolve_path(cfg["paths"]["drift_log"])
        # events from before drift logs carried a ticker have no way to be matched to a stock, so they are skipped
        events = ([e for e in (json.loads(l) for l in open(drift_log_path)) if e.get("ticker") == ticker]
                  if drift_log_path.exists() else [])
        if events:
            st.dataframe(pd.DataFrame(events[-10:]), width="stretch", hide_index=True)
        else:
            st.caption(f"No drift events logged for {ticker} yet.")

with col4:
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Retraining history</div>', unsafe_allow_html=True)
        history = get_model_history(ticker, cfg)
        if history:
            st.dataframe(pd.DataFrame(history)[["version", "trained_at", "metrics", "promoted"]],
                         width="stretch", hide_index=True)
        else:
            st.caption("No models trained yet — run `python -m scripts.onboard_tickers " + ticker + "`.")

col5, col6 = st.columns(2, gap="small")
with col5:
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Model accuracy over time</div>', unsafe_allow_html=True)
        # every promoted *and* rejected candidate the retraining manager has produced
        if history:
            hist_sorted = sorted(history, key=lambda m: m["trained_at"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[m["trained_at"] for m in hist_sorted],
                y=[m["metrics"]["accuracy"] for m in hist_sorted],
                mode="lines+markers", line=dict(color=TEXT_MUTED, width=1.3),
                marker=dict(size=9, color=[TEAL if m["promoted"] else CORAL for m in hist_sorted]),
                text=[f"v{m['version']}" for m in hist_sorted],
                hovertemplate="%{text}<br>%{x}<br>accuracy=%{y:.4f}<extra></extra>",
            ))
            style_fig(fig, 250, title="accuracy", range=[0, 1])
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.caption("Green dot = promoted. Red dot = rejected (failed the FR-9 benchmark gate).")
        else:
            st.caption("No models trained yet.")

with col6:
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Drift score over time</div>', unsafe_allow_html=True)
        drift_log_path = resolve_path(cfg["paths"]["drift_log"])
        drift_events = ([e for e in (json.loads(l) for l in open(drift_log_path)) if e.get("ticker") == ticker]
                        if drift_log_path.exists() else [])
        if drift_events:
            ddf = pd.DataFrame(drift_events)
            color_map = {"Regime Shift": CORAL, "Anomaly Spike": AMBER}
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ddf["date"], y=ddf["drift_score"], mode="lines+markers",
                                     line=dict(color=TEXT_MUTED, width=1.3),
                                     marker=dict(size=9, color=[color_map.get(c, TEAL) for c in ddf["classification"]]),
                                     text=ddf["classification"],
                                     hovertemplate="%{text}<br>%{x}<br>drift_score=%{y:.4f}<extra></extra>"))
            fig.add_hline(y=cfg["drift"]["threshold"], line_dash="dash", line_color=AMBER)
            style_fig(fig, 250, title="drift_score")
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
            st.caption("Dashed line = configured drift.threshold.")
        else:
            st.caption("No drift events logged yet.")

# ───────────────────────── Actual vs. predicted ─────────────────────────
# For every logged prediction, look up what the price actually did the next trading day (once
# that day's data exists) and show whether the call was right. Abstained predictions are shown
# but excluded from the accuracy count; predictions whose next day hasn't happened show as
# "pending", not wrong.
with st.container(border=True):
    st.markdown('<div class="eyebrow" id="calls">Actual vs. predicted</div>', unsafe_allow_html=True)
    ok_predictions = []
    if prediction_log_path.exists():
        # only this ticker's rows, and if a day was logged more than once keep the most recent entry for it
        _latest_per_day = {}
        for p in (json.loads(l) for l in open(prediction_log_path)):
            if p.get("status") == "ok" and p.get("ticker") == ticker:
                _latest_per_day[(p.get("as_of_date", p["date"][:10]), p.get("source", "live_cycle"))] = p
        ok_predictions = list(_latest_per_day.values())

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
                close_as_of = float(price_by_date[as_of])
                close_next = float(price_by_date[later_dates[0]])
                actual_dir = "UP" if close_next > close_as_of else "DOWN"
                result = "abstained (no call made)" if abstained else ("correct" if predicted_dir == actual_dir else "wrong")
            else:
                close_as_of, close_next = price_by_date.get(as_of), None
                result = "pending"

            rows.append({
                "date": as_of,
                "predicted": "ABSTAINED" if abstained else predicted_dir,
                "price_that_day": round(close_as_of, 2) if close_as_of is not None else None,
                "actual_price_next_day": round(close_next, 2) if close_next is not None else None,
                "result": result,
            })

        rows_sorted = sorted(rows, key=lambda r: r["date"], reverse=True)
        st.dataframe(pd.DataFrame(rows_sorted[:30]), width="stretch", hide_index=True)
        if len(rows_sorted) > 30:
            st.caption(f"Showing the most recent 30 of {len(rows_sorted)} logged predictions.")
        made = [r["predicted"] for r in rows if r["predicted"] in ("UP", "DOWN")]
        if made:
            up_share = made.count("UP") / len(made)
            lo, hi = cfg["training"]["balance_min_up_share"], cfg["training"]["balance_max_up_share"]
            msg = f"Calls made: {made.count('UP')} UP · {made.count('DOWN')} DOWN ({up_share:.0%} UP)."
            if lo <= up_share <= hi:
                st.caption(msg + " Balanced ✓")
            else:
                st.warning(msg + " Lopsided: this model is leaning one way, so its accuracy here mostly reflects "
                           "which way the market recently went, not skill.")
        resolved = [r for r in rows if r["result"] in ("correct", "wrong")]
        if resolved:
            n_correct = sum(1 for r in resolved if r["result"] == "correct")
            st.caption(f"{n_correct}/{len(resolved)} resolved (non-abstained) predictions correct. "
                       f"Abstained and still-pending predictions are excluded from this count.")
        else:
            st.caption("No predictions have a resolved next trading day yet.")
    else:
        st.caption("No predictions logged yet for this stock.")

st.caption("Want to demo a drift event? Use **Simulation engine** in the top bar — it's a separate tool "
           "where you pick a stock and inject your own false data.")
st.caption(
    "DriftSense is a course/portfolio project, not a financial product. "
    "It does not provide financial advice and must not be used to make "
    "real trading or investment decisions. (SRS §5.5 Safety.)"
)
