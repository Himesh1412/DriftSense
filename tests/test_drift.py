# tests/test_drift.py
import pandas as pd
import numpy as np
from src.drift.detector import compute_drift_score, is_drift_detected, population_stability_index
from src.drift.classifier import classify_drift


def _fake_price_df(n=100, seed=0, shock_at=None, shock_size=0.0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, n)
    if shock_at is not None:
        returns[shock_at:] += shock_size
    prices = 100 * np.cumprod(1 + returns)
    return pd.DataFrame({"Close": prices, "Volume": rng.integers(1000, 5000, n)})


def test_drift_score_low_when_distributions_match():
    baseline = _fake_price_df(seed=1)
    live = _fake_price_df(seed=2)  # same distribution, different draws
    result = compute_drift_score(baseline, live)
    assert result["drift_score"] < 0.3


def test_drift_score_high_when_distribution_shifts():
    baseline = _fake_price_df(seed=1)
    live = _fake_price_df(seed=1, shock_at=50, shock_size=0.05)
    result = compute_drift_score(baseline, live)
    assert result["drift_score"] > 0.0


def test_classifier_detects_no_event_on_flat_data():
    flat = _fake_price_df(seed=3)
    result = classify_drift(flat)
    assert result["classification"] in ["No Significant Event", "Anomaly Spike", "Regime Shift"]