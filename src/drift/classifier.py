# src/drift/classifier.py
"""
Regime Shift vs Anomaly Spike Classification (FR-5).
Heuristic based on spike magnitude, duration, and reversion pattern —
explicitly NOT a verified fraud/scam detector, just a shape-based proxy.
See SRS section 2.7 (Assumptions) for the documented limitation.
"""
import numpy as np
import pandas as pd

from src.config_loader import load_config


def classify_drift(live_df: pd.DataFrame, cfg: dict = None, feature: str = "Close") -> dict:
    """
    Looks at how the live window's volatility/returns behave:
    - Large, short-lived, reverting spike  -> "Anomaly Spike"
    - Sustained shift lasting several days -> "Regime Shift"
    """
    cfg = cfg or load_config()
    returns = live_df[feature].pct_change().dropna()

    rolling_mean = returns.rolling(10).mean()
    rolling_std = returns.rolling(10).std()
    z_scores = (returns - rolling_mean) / rolling_std

    spike_threshold = cfg["drift"]["spike_magnitude_std"]
    spike_days = z_scores[abs(z_scores) > spike_threshold]

    if len(spike_days) == 0:
        return {"classification": "No Significant Event", "magnitude": 0.0, "duration_days": 0}

    magnitude = float(abs(z_scores).max())
    # duration: how many consecutive recent days stayed elevated after the spike
    recent = z_scores.tail(cfg["drift"]["regime_min_duration_days"])
    duration = int((abs(recent) > spike_threshold / 2).sum())

    max_anomaly_duration = cfg["drift"]["spike_max_duration_days"]
    min_regime_duration = cfg["drift"]["regime_min_duration_days"]

    if duration >= min_regime_duration:
        classification = "Regime Shift"
    elif duration <= max_anomaly_duration:
        classification = "Anomaly Spike"
    else:
        # ambiguous middle ground — default to the safer (non-retraining) label
        classification = "Anomaly Spike"

    return {
        "classification": classification,
        "magnitude": round(magnitude, 3),
        "duration_days": duration,
    }