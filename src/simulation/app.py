# src/simulation/app.py
"""
DriftSense Simulation Engine — a standalone test bench, separate from the
monitoring dashboard.

    streamlit run src/simulation/app.py --server.port 8502

Pick any stock, design some false data (a spike, a regime shift, or type your
own daily moves), preview it, then run it through the REAL drift pipeline.
Everything is synthetic and sandboxed: your real models, logs and predictions
are never touched, and a stock that isn't onboarded is fetched in memory only.
"""
import html as _html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config_loader import load_config, resolve_path
from src.dashboard.theme import AMBER, CORAL, GRID, INK, SURFACE, TEAL, TEXT_MUTED, apply_theme
from src.simulation.scenarios import (
    MAX_DAILY_MOVE, inject_anomaly_spike, inject_custom_returns, inject_regime_shift, recent_sigma,
)

st.set_page_config(page_title="DriftSense · Simulation Engine", layout="wide", page_icon="◆")
cfg = load_config()
SIM = cfg["simulator"]
apply_theme(roomy=True)

MODES = ["Anomaly spike", "Regime shift", "Type my own moves"]


def trained_tickers() -> list:
    model_dir = resolve_path(cfg["paths"]["model_dir"])
    found = sorted(p.name for p in model_dir.iterdir()
                   if p.is_dir() and not p.name.startswith("_") and any(p.glob("v*"))) if model_dir.exists() else []
    return found


@st.cache_data(show_spinner="Loading price history…", ttl=600)
def get_inputs(ticker: str):
    from src.simulation.runner import load_inputs
    return load_inputs(ticker, cfg)


def parse_moves(text: str):
    tokens = [t for t in text.replace("%", " ").replace(",", " ").split() if t]
    return [float(t) for t in tokens]


# ───────────────────────── header ─────────────────────────
st.markdown(f"""
<div class="pagehead">
  <div><span class="title">DriftSense</span><span class="crumb">Simulation engine</span></div>
  <a class="pill" href="http://localhost:{SIM['dashboard_port']}" target="_self">← Monitoring dashboard</a>
</div>
<div class="note"><b>SIMULATED.</b> This is a test bench. You choose a stock and inject false data into
its real price window, and the real DriftSense pipeline reacts. Nothing here is a market event, and nothing is saved
to your real models, logs or predictions.</div>
""", unsafe_allow_html=True)

left, right = st.columns([1, 1.7], gap="medium")

# ───────────────────────── controls ─────────────────────────
with left:
    with st.container(border=True):
        st.markdown('<div class="eyebrow">1 · Choose a stock</div>', unsafe_allow_html=True)
        known = trained_tickers()
        choice = st.selectbox("Stock", known + ["Other (type any ticker)…"], label_visibility="collapsed")
        if choice.startswith("Other"):
            ticker = st.text_input("Ticker symbol", value="", placeholder="e.g. AMD, WIPRO.NS, TSLA").strip().upper()
        else:
            ticker = choice
        if not ticker:
            st.info("Type a ticker symbol to continue.")
            st.stop()

        try:
            live_df, snapshot_df = get_inputs(ticker)
        except Exception as e:
            st.error(f"Couldn't load price data for '{ticker}': {e}")
            st.stop()

        sigma = recent_sigma(live_df)
        has_model = ticker in known
        st.caption(f"{len(snapshot_df)} days of history · latest close {live_df['Close'].iloc[-1]:,.2f} · "
                   f"typical daily move (σ) {sigma:.1%} · "
                   + ("trained model ✓" if has_model else "no trained model (drift + classification only)"))

        st.markdown('<div class="sec"></div><div class="eyebrow">2 · Design the false data</div>', unsafe_allow_html=True)
        mode = st.radio("Type of event", MODES, label_visibility="collapsed")

        def pct(sigmas):  # what a size in σ means for THIS stock, capped like the engine caps it
            return min(sigmas * sigma, MAX_DAILY_MOVE)

        sim_cfg, custom_moves, parse_error = dict(SIM), None, None
        if mode == "Anomaly spike":
            direction = st.radio("Shock", ["drop", "jump"], horizontal=True, key="sp_dir")
            size = st.slider("Shock size (× this stock's normal daily move)", 4, 20, int(SIM["spike_sigmas"]), key="sp_size")
            st.caption(f"≈ {'−' if direction == 'drop' else '+'}{pct(size):.0%} in one day, reverting the next day")
            turb_days = st.slider("Turbulent days before the shock", 0, 60, int(SIM["spike_buildup_days"]), key="sp_td")
            turb_lvl = st.slider("Turbulence level (× normal)", 0.0, 4.0, float(SIM["spike_buildup_sigmas"]), 0.5, key="sp_tl")
            calm_before = st.slider("Calm days right before it", 0, 10, int(SIM["spike_calm_before_days"]), key="sp_cb")
            calm_after = st.slider("Calm days after it", 0, 10, int(SIM["spike_calm_days"]), key="sp_ca")
            sim_cfg.update(spike_direction=direction, spike_sigmas=size, spike_buildup_days=turb_days,
                           spike_buildup_sigmas=turb_lvl, spike_calm_before_days=calm_before, spike_calm_days=calm_after)
        elif mode == "Regime shift":
            direction = st.radio("First swing", ["drop", "jump"], horizontal=True, key="rg_dir")
            size = st.slider("First swing size (× normal daily move)", 4, 20, int(SIM["regime_first_swing_sigmas"]), key="rg_size")
            st.caption(f"≈ {'−' if direction == 'drop' else '+'}{pct(size):.0%} on day one, then it whipsaws and grows")
            swing_days = st.slider("Whipsaw days", 3, 10, int(SIM["regime_swing_days"]), key="rg_sd")
            growth = st.slider("Growth per swing", 1.0, 1.4, float(SIM["regime_swing_growth"]), 0.05, key="rg_gr")
            turb_days = st.slider("Turbulent days before", 0, 60, int(SIM["regime_buildup_days"]), key="rg_td")
            turb_lvl = st.slider("Turbulence level (× normal)", 0.0, 4.0, float(SIM["regime_buildup_sigmas"]), 0.5, key="rg_tl")
            calm_before = st.slider("Calm days right before the shock", 0, 12, int(SIM["regime_calm_before_days"]), key="rg_cb")
            sim_cfg.update(regime_first_direction=direction, regime_first_swing_sigmas=size, regime_swing_days=swing_days,
                           regime_swing_growth=growth, regime_buildup_days=turb_days, regime_buildup_sigmas=turb_lvl,
                           regime_calm_before_days=calm_before)
        else:
            raw = st.text_area("Daily moves in %, oldest first", value="-3, 2, -6, 8, -12, 15", height=90,
                               help=f"Appended after the last real day. Each move is capped at ±{MAX_DAILY_MOVE:.0%}.")
            try:
                custom_moves = parse_moves(raw)
                if not custom_moves:
                    parse_error = "Type at least one daily move."
            except ValueError:
                parse_error = "Couldn't read that — use numbers separated by commas, e.g. -3, 2, -6, 8"
            if parse_error:
                st.error(parse_error)
            else:
                st.caption(f"{len(custom_moves)} days · compare with this stock's normal ±{sigma:.1%}")

        with st.expander("Advanced"):
            seed = st.number_input("Random seed (for the turbulent days)", min_value=0, max_value=9999, value=int(SIM["seed"]))
            do_retrain = st.checkbox("If it's a Regime Shift, also run the retrain + promotion gate (about a minute)", value=True,
                                     disabled=not has_model)

# ───────────────────── build the false data (cheap, live preview) ─────────────────────
sim_df = meta = None
if not parse_error:
    if mode == "Anomaly spike":
        sim_df, meta = inject_anomaly_spike(live_df, sim_cfg, seed=int(seed))
    elif mode == "Regime shift":
        sim_df, meta = inject_regime_shift(live_df, sim_cfg, seed=int(seed))
    else:
        sim_df, meta = inject_custom_returns(live_df, custom_moves)
signature = repr((ticker, mode, sorted(sim_cfg.items()) if mode != MODES[2] else custom_moves, int(seed), do_retrain))


def price_chart(df: pd.DataFrame) -> go.Figure:
    real, synth = df[~df["synthetic"]], df[df["synthetic"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=real["Date"], y=real["Close"], mode="lines", name="real prices", line=dict(color=INK, width=1.6)))
    joined = pd.concat([real.tail(1), synth])  # keeps the line continuous at the join
    fig.add_trace(go.Scatter(x=joined["Date"], y=joined["Close"], mode="lines", name="SIMULATED",
                             line=dict(color=AMBER, width=1.8, dash="dot")))
    fig.add_shape(type="line", x0=synth["Date"].iloc[0], x1=synth["Date"].iloc[0], y0=0, y1=1, yref="paper",
                  line=dict(color=TEXT_MUTED, dash="dash", width=1))
    fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=340, margin=dict(l=0, r=0, t=10, b=0),
                      font=dict(color=TEXT_MUTED, size=13), legend=dict(orientation="h", y=1.12),
                      xaxis=dict(gridcolor=GRID), yaxis=dict(gridcolor=GRID))
    return fig


def render_result(res: dict):
    d, c, a, rt = res["drift"], res["classification"], res["attribution"], res["retraining"]
    label = c["classification"] if c else "—"
    label_color = {"Regime Shift": CORAL, "Anomaly Spike": AMBER}.get(label, TEXT_MUTED)

    st.markdown('<div class="eyebrow" style="margin-top:8px;">What DriftSense did</div>', unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3, gap="medium")
    k1.markdown(f"""
    <div class="ds-card"><div class="eyebrow">Drift score (FR-4)</div>
      <div class="ds-num">{d['drift_score']:.3f}</div>
      <div class="ds-metric-label">threshold {d['threshold']} &middot; {'DRIFT DETECTED' if d['detected'] else 'below threshold'}</div>
    </div>""", unsafe_allow_html=True)
    k2.markdown(f"""
    <div class="ds-card"><div class="eyebrow">Classification (FR-5)</div>
      <div class="ds-num" style="font-size:22px;color:{label_color};">{label}</div>
      <div class="ds-metric-label">{_html.escape(c['reasoning']) if c else 'Not classified — the drift gate was not tripped.'}</div>
    </div>""", unsafe_allow_html=True)
    k3.markdown(f"""
    <div class="ds-card"><div class="eyebrow">Attributed cause (FR-6)</div>
      <div class="ds-num" style="font-size:22px;">{a['top_feature'] if a else '—'}</div>
      <div class="ds-metric-label">{_html.escape(a['explanation_text']) if a else 'No attribution (nothing flagged, or no trained model).'}</div>
    </div>""", unsafe_allow_html=True)

    if rt:
        vc = TEAL if rt["promoted"] else CORAL
        st.markdown(f"""
        <div class="ds-card" style="margin-top:4px;border-color:{vc};">
          <div class="eyebrow">Retrain + promotion gate (FR-7 / FR-9) &middot; sandboxed</div>
          <div class="ds-metric-row">
            <div><div class="ds-metric-label">Candidate accuracy</div><div class="ds-num" style="font-size:22px;">{rt['accuracy_after']:.1%}</div></div>
            <div><div class="ds-metric-label">Current model</div><div class="ds-num" style="font-size:22px;">{rt['accuracy_before']:.1%}</div></div>
            <div><div class="ds-metric-label">Gate decision</div>
              <div class="ds-num" style="font-size:22px;color:{vc};">{'PROMOTED' if rt['promoted'] else 'REJECTED'}</div></div>
          </div>
          <div class="ds-metric-label" style="margin-top:10px;">Benchmarked on the simulated window; promoted only if it does not underperform the current model. This simulated model is discarded — your real model is unchanged.</div>
        </div>""", unsafe_allow_html=True)

    st.info(res["outcome"])
    st.caption(f"Ran in {res['seconds']}s" + (f" against the real v{res['model_version']} model" if res["has_model"] else "")
               + " — sandboxed, nothing saved.")


# ───────────────────────── preview + run ─────────────────────────
with right:
    run = False
    with st.container(border=True):
        st.markdown('<div class="eyebrow">Preview · what you are about to inject</div>', unsafe_allow_html=True)
        if sim_df is None:
            st.warning("Fix the moves on the left to see a preview.")
        else:
            st.caption(meta["description"])
            st.plotly_chart(price_chart(sim_df), width="stretch", config={"displayModeBar": False})
            synth = sim_df[sim_df["synthetic"]]["Close"]
            worst = sim_df["Close"].pct_change().iloc[-meta["n_synthetic"]:].abs().max()
            p1, p2, p3 = st.columns(3, gap="medium")
            p1.markdown(f'<div class="tile"><div class="ds-metric-label">Injected days</div>'
                        f'<div class="ds-num">{int(meta["n_synthetic"])}</div></div>', unsafe_allow_html=True)
            p2.markdown(f'<div class="tile"><div class="ds-metric-label">Biggest one-day move</div>'
                        f'<div class="ds-num">{worst:.0%}</div></div>', unsafe_allow_html=True)
            p3.markdown(f'<div class="tile"><div class="ds-metric-label">Simulated price range</div>'
                        f'<div class="ds-num">{synth.min():,.0f} – {synth.max():,.0f}</div></div>', unsafe_allow_html=True)
            run = st.button("▶  Run through DriftSense", type="primary", width="stretch")

    if sim_df is not None:
        if run:
            from src.simulation.runner import analyze_window
            retrain_note = " (and retraining a candidate — about a minute)" if do_retrain and has_model and mode != MODES[0] else ""
            with st.spinner("Running the real drift pipeline" + retrain_note + "…"):
                try:
                    res = analyze_window(ticker, sim_df, meta, snapshot_df, cfg, run_retraining=do_retrain)
                    res["signature"] = signature
                    st.session_state["engine_result"] = res
                except Exception as e:
                    st.session_state["engine_result"] = {"error": str(e), "signature": signature}

        res = st.session_state.get("engine_result")
        if res:
            if "error" in res:
                st.error(f"The simulation couldn't run: {res['error']}")
            else:
                if res["signature"] != signature:
                    st.warning("You've changed the settings since this run — press Run again to refresh the result.")
                render_result(res)

with st.expander("What do these results mean?"):
    st.markdown(
        "- **Drift score** compares this stock's recent daily moves to its whole history (KS test + PSI). "
        "Above the threshold means *something has changed* — no more.\n"
        "- **Anomaly spike**: a short, sharp shock that fades quickly. It's flagged and logged, but the model is "
        "**not** retrained — one freak day shouldn't rewrite what the model learned.\n"
        "- **Regime shift**: the shake-up lasts several days in a row. Only this triggers retraining.\n"
        "- **Attributed cause** (SHAP) names the input feature the model's behaviour is most tied to.\n"
        "- **Promotion gate**: a retrained candidate replaces the current model only if it does not do worse "
        "on the benchmark window.\n\n"
        "Honest caveat: the classifier is strict. A steady trend alone reads as *No Significant Event*; a lone spike in a "
        "calm market doesn't even trip the drift score. That's the detector behaving as designed, not a bug in the demo.")

st.caption("DriftSense is a course/portfolio project, not a financial product. Everything on this page is synthetic.")
