# tests/test_decision.py
import numpy as np
import pandas as pd
import pytest
import torch

from src.forecasting import train as train_module
from src.forecasting.confidence import predict_with_confidence
from src.forecasting.decision import (
    balance_report, calls_with_context, decide, fit_decision_offset, latest_reference,
    logit_gaps, rolling_reference,
)
from src.forecasting.model import DirectionLSTM
from src.forecasting.train import FEATURE_COLS, train_baseline


def _fake_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"Close": 100 * np.cumprod(1 + rng.normal(0, 0.01, n)),
                         "Volume": rng.integers(1000, 5000, n)})


# ── the arithmetic ──────────────────────────────────────────────────────────
def test_rolling_reference_only_looks_at_past_scores():
    gaps = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    ref = rolling_reference(gaps, start=2, n_ref=2)
    assert ref.tolist() == [1.5, 2.5, 3.5]   # i=2 -> median(1,2); i=3 -> median(2,3); i=4 -> median(3,4)
    # the huge value at the end never leaks into any reference
    assert ref.max() < 4


def test_latest_reference_is_the_median_of_the_scores_before_the_newest_window():
    assert latest_reference(np.array([1.0, 3.0, 5.0, 999.0]), n_ref=3) == 3.0
    assert latest_reference(np.array([7.0]), n_ref=3) == 0.0  # no history yet -> neutral


def test_decide_is_strictly_above_reference():
    assert decide(np.array([0.1, 0.0, -0.1]), 0.0).tolist() == [1, 0, 0]


def test_balance_report_flags_lopsided_calls():
    assert balance_report(np.array([1, 0, 1, 0]), 0.3, 0.7) == {"up_share": 0.5, "balanced": True}
    assert balance_report(np.array([0, 0, 0, 0, 0]), 0.3, 0.7)["balanced"] is False
    assert balance_report(np.array([1, 1, 1, 1, 1]), 0.3, 0.7)["balanced"] is False


# ── the actual bug: a constant lean makes one-sided calls ───────────────────
def test_a_constant_lean_is_one_sided_with_a_plain_sign_but_balanced_with_the_rolling_reference():
    """AAPL's real scores looked like this: average -0.012, spread 0.011."""
    rng = np.random.default_rng(0)
    gaps = -0.012 + rng.normal(0, 0.011, 400)

    plain_sign = decide(gaps, 0.0)
    recentered = decide(gaps[60:], rolling_reference(gaps, start=60, n_ref=60))

    assert balance_report(plain_sign, 0.3, 0.7)["balanced"] is False   # "DOWN every time"
    assert balance_report(recentered, 0.3, 0.7)["balanced"] is True


def test_rolling_reference_keeps_the_relative_signal():
    """Recentering must remove the lean, not the information: a clearly-higher score still means UP."""
    gaps = np.full(100, -0.5)
    gaps[-1] = -0.4   # still negative, but higher than usual
    ref = rolling_reference(gaps, start=99, n_ref=60)
    assert decide(gaps[99:], ref).tolist() == [1]


# ── model plumbing ──────────────────────────────────────────────────────────
def test_fit_decision_offset_makes_validation_calls_match_the_training_up_rate():
    model = DirectionLSTM(n_features=len(FEATURE_COLS), hidden_size=8, num_layers=1)
    X_val = np.random.default_rng(1).normal(size=(200, 20, len(FEATURE_COLS))).astype(np.float32)
    fit_decision_offset(model, X_val, train_up_rate=0.6)
    calls = decide(logit_gaps(model, X_val), float(model.decision_offset.item()))
    assert calls.mean() == pytest.approx(0.6, abs=0.03)


def test_calls_with_context_uses_history_before_the_evaluated_windows():
    model = DirectionLSTM(n_features=len(FEATURE_COLS), hidden_size=8, num_layers=1)
    X = np.random.default_rng(2).normal(size=(120, 20, len(FEATURE_COLS))).astype(np.float32)
    with_history = calls_with_context(model, X[:100], X[100:], n_ref=60)
    gaps = logit_gaps(model, X)
    expected = decide(gaps[100:], rolling_reference(gaps, start=100, n_ref=60))
    assert with_history.tolist() == expected.tolist()


def test_reference_decides_the_direction_of_a_live_call():
    cfg = {"confidence": {"mc_dropout_passes": 3}}
    model = DirectionLSTM(n_features=len(FEATURE_COLS), hidden_size=8, num_layers=1)
    x = np.random.default_rng(3).normal(size=(20, len(FEATURE_COLS))).astype(np.float32)
    gap = float(logit_gaps(model, x[None])[0])

    assert predict_with_confidence(model, x, cfg, reference=gap - 1.0)["direction"] == "UP"
    assert predict_with_confidence(model, x, cfg, reference=gap + 1.0)["direction"] == "DOWN"


# ── training-time balance check ─────────────────────────────────────────────
def test_training_reports_whether_its_calls_are_balanced():
    result = train_baseline(_fake_df(), {"data": {"sequence_length": 20}, "training": {"epochs": 2, "hidden_size": 8, "num_layers": 1}})
    assert "up_share" in result["metrics"] and "balanced" in result["metrics"]
    assert result["balance"]["up_share"] == result["metrics"]["up_share"]
    assert 1 <= result["attempts"] <= 3


def test_a_lopsided_run_is_retrained_and_a_balanced_retry_is_kept(monkeypatch):
    seen = {"n": 0}

    def fake_calls(model, before, evaluated, n_ref):
        seen["n"] += 1
        # first run: everything UP (lopsided). second run: alternating (balanced).
        return np.ones(len(evaluated), dtype=int) if seen["n"] == 1 else np.arange(len(evaluated)) % 2

    monkeypatch.setattr(train_module, "calls_with_context", fake_calls)
    cfg = {"data": {"sequence_length": 20}, "training": {"epochs": 1, "hidden_size": 8, "num_layers": 1}}
    result = train_baseline(_fake_df(n=400), cfg)
    assert result["attempts"] == 2 and result["balance"]["balanced"] is True


def test_a_model_that_stays_lopsided_is_flagged_after_the_retries_run_out(monkeypatch):
    monkeypatch.setattr(train_module, "calls_with_context", lambda m, before, ev, n: np.ones(len(ev), dtype=int))
    cfg = {"data": {"sequence_length": 20},
           "training": {"epochs": 1, "hidden_size": 8, "num_layers": 1, "balance_max_retries": 2}}
    result = train_baseline(_fake_df(n=400), cfg)
    assert result["attempts"] == 3                      # 1 try + 2 retries
    assert result["metrics"]["balanced"] is False       # honestly flagged, not hidden
    assert result["metrics"]["up_share"] == 1.0
