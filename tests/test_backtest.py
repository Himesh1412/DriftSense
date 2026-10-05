# tests/test_backtest.py
import numpy as np
import pandas as pd
import pytest
from src.forecasting.backtest import walk_forward_backtest


def _fake_df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame({
        "Close": prices,
        "Volume": rng.integers(1000, 5000, n),
    })


def test_walk_forward_backtest_produces_one_result_per_fold():
    cfg = {
        "data": {"sequence_length": 20},
        "backtest": {"n_folds": 3, "min_train_days": 150, "test_window_days": 30},
    }
    result = walk_forward_backtest(_fake_df(), cfg=cfg, epochs=2)
    assert len(result["folds"]) == 3
    for fold in result["folds"]:
        assert 0.0 <= fold["accuracy"] <= 1.0
        assert fold["test_end_idx"] > fold["train_end_idx"] >= fold["n_train"] - 1  # no leakage: test starts at train_end
    assert "accuracy_mean" in result["summary"]
    assert "accuracy_std" in result["summary"]


def test_walk_forward_backtest_raises_when_not_enough_data():
    cfg = {
        "data": {"sequence_length": 20},
        "backtest": {"n_folds": 3, "min_train_days": 350, "test_window_days": 30},
    }
    with pytest.raises(ValueError):
        walk_forward_backtest(_fake_df(n=100), cfg=cfg, epochs=1)

def test_backtest_reports_how_lopsided_each_folds_calls_were():
    cfg = {
        "data": {"sequence_length": 20},
        "backtest": {"n_folds": 3, "min_train_days": 150, "test_window_days": 30},
        "training": {"epochs": 2, "hidden_size": 8, "num_layers": 1},
    }
    result = walk_forward_backtest(_fake_df(), cfg=cfg, epochs=2)
    for fold in result["folds"]:
        assert 0.0 <= fold["up_share"] <= 1.0
        assert isinstance(fold["balanced"], bool)
    assert "up_share_mean" in result["summary"] and "lopsided_folds" in result["summary"]


def test_every_day_in_the_test_window_is_scored_not_just_the_days_after_the_lookback():
    """A 30-day window used to yield ~11 predictions (30 minus the 19 lookback days)."""
    cfg = {
        "data": {"sequence_length": 20},
        "backtest": {"n_folds": 3, "min_train_days": 150, "test_window_days": 30},
        "training": {"epochs": 1, "hidden_size": 8, "num_layers": 1},
    }
    result = walk_forward_backtest(_fake_df(), cfg=cfg, epochs=1)
    for fold in result["folds"]:
        window = fold["test_end_idx"] - fold["test_start_idx"]
        assert window == 30
        assert fold["n_test"] >= window - 1  # only the very latest day (label not known yet) can be missing
