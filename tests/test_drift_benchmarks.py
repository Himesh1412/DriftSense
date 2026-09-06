# tests/test_drift_benchmarks.py
import numpy as np
import pandas as pd
from src.drift.benchmarks import measure_false_alarm_rate, measure_latency


def _fake_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame({"Close": prices, "Volume": rng.integers(1000, 5000, n)})


def test_false_alarm_rate_is_a_valid_fraction():
    cfg = {"drift": {"threshold": 0.30}}
    result = measure_false_alarm_rate(_fake_df(), cfg, window=90, n_trials=20)
    assert result["n_trials"] == 20
    assert 0.0 <= result["false_alarm_rate"] <= 1.0
    assert result["false_alarms"] <= result["n_trials"]


def test_latency_report_has_expected_keys_and_meets_nfr1_budget():
    cfg = {"drift": {"threshold": 0.30}}
    result = measure_latency(_fake_df(), cfg, window=90, n_trials=5)
    assert result["n_trials"] == 5
    assert "mean_seconds" in result and "p95_seconds" in result
    # NFR-1: drift detection must complete within 5 seconds on a standard laptop CPU
    assert result["meets_nfr1_5s_budget"] is True
    assert result["max_seconds"] < 5.0
