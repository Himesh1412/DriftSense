# src/simulation/scenarios.py
"""
Synthetic stress scenarios for demonstrating the drift pipeline (supports
SRS acceptance criterion: distinguish a regime shift from an anomaly spike).

Both generators take a REAL recent price window and append clearly-synthetic
days to it, trimming the oldest real days so the window length is unchanged.
Injected rows carry synthetic=True so nothing downstream (chart, report) can
mistake them for market data. Nothing here reads or writes real logs/models.

Sizes are multiples of the stock's own recent daily volatility (sigma), so one
config works for any ticker. The shapes are not arbitrary: the existing drift
classifier is strict (a 10-day z-score can't exceed 2.85, and a Regime Shift
needs five consecutive elevated days), so a steady trend reads as "No
Significant Event" — it takes an escalating whipsaw to count as a regime shift.
"""
import numpy as np
import pandas as pd


def _recent_sigma(df: pd.DataFrame, lookback: int = 60) -> float:
    return float(df["Close"].pct_change().dropna().tail(lookback).std())


MAX_DAILY_MOVE = 0.45  # no synthetic day moves a stock more than 45% (also keeps prices positive)


def _buildup(rng, days: int, sigma: float, calm_tail: int) -> list:
    """Choppy days, with the last `calm_tail` of them quiet — turbulence, then a lull, then the shock.
    The lull is what makes the shock stand out against its rolling window."""
    noise = list(rng.normal(0, sigma, days))
    for i in range(max(0, days - calm_tail), days):
        noise[i] *= 0.2
    return noise


def _append_synthetic(live_df: pd.DataFrame, returns: list, sigma: float) -> pd.DataFrame:
    """Walk the price forward by `returns`, tag the new rows, keep the window length."""
    returns = [float(np.clip(r, -MAX_DAILY_MOVE, MAX_DAILY_MOVE)) for r in returns]
    n = len(returns)
    last_close = float(live_df["Close"].iloc[-1])
    base_volume = float(np.median(live_df["Volume"].tail(30)))
    last_date = pd.to_datetime(live_df["Date"].iloc[-1])
    dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n)

    closes, volumes = [], []
    price = last_close
    for r in returns:
        price *= 1 + r
        closes.append(price)
        # bigger moves come with heavier volume — as they do in real markets
        volumes.append(base_volume * (1 + 0.5 * abs(r) / sigma))

    new_rows = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in dates],
        "Close": closes, "High": closes, "Low": closes, "Open": closes, "Volume": volumes,
    })
    real = live_df.copy()
    if "synthetic" not in real.columns:
        real["synthetic"] = False
    new_rows["synthetic"] = True
    combined = pd.concat([real, new_rows], ignore_index=True)
    return combined.tail(len(live_df)).reset_index(drop=True)


def inject_regime_shift(live_df: pd.DataFrame, sim_cfg: dict, seed: int = None):
    """
    Volatility regime shift: a stretch of choppier-than-normal days (the build-up
    that makes the drift score cross its threshold), then an escalating whipsaw
    — first swing is a drop, each next swing flips sign and grows by
    regime_swing_growth.
    """
    rng = np.random.default_rng(sim_cfg["seed"] if seed is None else seed)
    sigma = _recent_sigma(live_df)
    buildup = _buildup(rng, sim_cfg["regime_buildup_days"], sim_cfg["regime_buildup_sigmas"] * sigma,
                       sim_cfg.get("regime_calm_before_days", 0))
    swings = [-(1 if k % 2 == 0 else -1) * sim_cfg["regime_first_swing_sigmas"] * sigma
              * sim_cfg["regime_swing_growth"] ** k for k in range(sim_cfg["regime_swing_days"])]
    sim_df = _append_synthetic(live_df, buildup + swings, sigma)
    meta = {
        "scenario": "regime_shift", "sigma": sigma,
        "n_synthetic": int(sim_df["synthetic"].sum()),
        "description": (f"{sim_cfg['regime_buildup_days']} days of choppier trading "
                        f"({sim_cfg['regime_buildup_sigmas']}x normal volatility), then "
                        f"{sim_cfg['regime_swing_days']} escalating whipsaw days starting at a "
                        f"{sim_cfg['regime_first_swing_sigmas']}x-sigma drop."),
    }
    return sim_df, meta


def inject_anomaly_spike(live_df: pd.DataFrame, sim_cfg: dict, seed: int = None):
    """
    One-day shock that fully reverts the next day (a news-driven spike), then a
    few calm days.
    """
    rng = np.random.default_rng(sim_cfg["seed"] if seed is None else seed)
    sigma = _recent_sigma(live_df)
    buildup = _buildup(rng, sim_cfg["spike_buildup_days"], sim_cfg["spike_buildup_sigmas"] * sigma,
                       sim_cfg.get("spike_calm_before_days", 0))
    shock = [-sim_cfg["spike_sigmas"] * sigma, sim_cfg["spike_sigmas"] * sigma * 0.97]
    calm = list(rng.normal(0, 0.3 * sigma, sim_cfg["spike_calm_days"]))
    sim_df = _append_synthetic(live_df, buildup + shock + calm, sigma)
    meta = {
        "scenario": "anomaly_spike", "sigma": sigma,
        "n_synthetic": int(sim_df["synthetic"].sum()),
        "description": (f"{sim_cfg['spike_buildup_days']} days of choppier trading "
                        f"({sim_cfg['spike_buildup_sigmas']}x normal volatility), then a "
                        f"one-day {sim_cfg['spike_sigmas']}x-sigma drop that reverts the next day, "
                        f"followed by {sim_cfg['spike_calm_days']} calm days."),
    }
    return sim_df, meta
