# src/drift/benchmarks.py
"""
Drift Detector Performance Benchmarks (NFR-1, NFR-6).
Measures the two things the SRS explicitly requires be "measured and
reported, not assumed": detection latency on a single incoming data point,
and the detector's false-alarm rate.

False-alarm rate methodology: there's no ground-truth "this day had real
drift" label for historical market data, so counting how often
is_drift_detected fires on real history conflates genuine drift with false
alarms. Instead, this bootstraps many synthetic "definitely no drift" live
windows by resampling daily returns (with replacement) from the real
historical snapshot's own return distribution and rebuilding a synthetic
price series from them. Because each synthetic window is drawn from the
exact same distribution as the baseline by construction, any time
is_drift_detected fires on one is unambiguously a false alarm, not a real
regime change — that gives an honest, reproducible false-alarm rate.
"""
import time
import numpy as np
import pandas as pd

from src.drift.detector import compute_drift_score, is_drift_detected


def bootstrap_no_drift_window(baseline_returns: np.ndarray, n_days: int, seed: int) -> pd.DataFrame:
    """A synthetic live window with no real distributional shift from baseline_returns."""
    rng = np.random.default_rng(seed)
    sampled_returns = rng.choice(baseline_returns, size=n_days, replace=True)
    prices = 100 * np.cumprod(1 + sampled_returns)
    volume = rng.integers(1_000_000, 5_000_000, n_days)
    return pd.DataFrame({"Close": prices, "Volume": volume})


def measure_false_alarm_rate(baseline_df: pd.DataFrame, cfg: dict, window: int, n_trials: int = 200) -> dict:
    """NFR-6: fraction of known-no-drift synthetic windows the detector flags anyway."""
    baseline_returns = baseline_df["Close"].pct_change().dropna().values

    false_alarms = 0
    drift_scores = []
    for trial in range(n_trials):
        synthetic_live = bootstrap_no_drift_window(baseline_returns, window, seed=trial)
        result = compute_drift_score(baseline_df, synthetic_live)
        drift_scores.append(result["drift_score"])
        if is_drift_detected(result, cfg):
            false_alarms += 1

    return {
        "n_trials": n_trials,
        "false_alarms": false_alarms,
        "false_alarm_rate": round(false_alarms / n_trials, 4),
        "drift_score_mean": round(float(np.mean(drift_scores)), 4),
        "drift_score_max": round(float(np.max(drift_scores)), 4),
    }


def measure_latency(baseline_df: pd.DataFrame, cfg: dict, window: int, n_trials: int = 30) -> dict:
    """NFR-1: drift detection must complete within 5 seconds on a standard laptop CPU."""
    baseline_returns = baseline_df["Close"].pct_change().dropna().values

    latencies = []
    for trial in range(n_trials):
        live = bootstrap_no_drift_window(baseline_returns, window, seed=1000 + trial)
        start = time.perf_counter()
        result = compute_drift_score(baseline_df, live)
        is_drift_detected(result, cfg)
        latencies.append(time.perf_counter() - start)

    return {
        "n_trials": n_trials,
        "mean_seconds": round(float(np.mean(latencies)), 4),
        "max_seconds": round(float(np.max(latencies)), 4),
        "p95_seconds": round(float(np.percentile(latencies, 95)), 4),
        "meets_nfr1_5s_budget": bool(np.max(latencies) < 5.0),
    }
