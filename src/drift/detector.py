# src/drift/detector.py
"""
Drift Detection Module (FR-4).
Statistical (non-ML) comparison of the live-data window against the
training-data distribution using the Kolmogorov-Smirnov test and PSI.
Deliberately not a model — drift detection should stay cheap and interpretable.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config_loader import load_config


def population_stability_index(baseline: np.ndarray, live: np.ndarray, bins: int = 10) -> float:
    """Standard PSI calculation between two distributions."""
    edges = np.percentile(baseline, np.linspace(0, 100, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    base_pct = np.histogram(baseline, bins=edges)[0] / len(baseline)
    live_pct = np.histogram(live, bins=edges)[0] / len(live)

    base_pct = np.clip(base_pct, 1e-4, None)
    live_pct = np.clip(live_pct, 1e-4, None)

    return float(np.sum((live_pct - base_pct) * np.log(live_pct / base_pct)))


def compute_drift_score(baseline_df: pd.DataFrame, live_df: pd.DataFrame, feature: str = "Close") -> dict:
    """
    FR-4: compute a drift score comparing live vs training distribution.
    Returns both KS-test and PSI so the classifier (FR-5) has more signal to work with.
    """
    baseline_returns = baseline_df[feature].pct_change().dropna().values
    live_returns = live_df[feature].pct_change().dropna().values

    ks_stat, p_value = ks_2samp(baseline_returns, live_returns)
    psi = population_stability_index(baseline_returns, live_returns)

    # Normalize KS statistic (already 0-1) and PSI (typically 0-0.5+ for real drift)
    # into a single comparable drift_score in [0, 1].
    psi_norm = min(psi / 0.5, 1.0)
    drift_score = round(float(0.5 * ks_stat + 0.5 * psi_norm), 4)

    return {
        "drift_score": drift_score,
        "ks_statistic": round(float(ks_stat), 4),
        "ks_p_value": round(float(p_value), 4),
        "psi": round(float(psi), 4),
    }


def is_drift_detected(drift_result: dict, cfg: dict = None) -> bool:
    cfg = cfg or load_config()
    return drift_result["drift_score"] > cfg["drift"]["threshold"]