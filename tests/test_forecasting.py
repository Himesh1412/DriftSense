# tests/test_forecasting.py
import numpy as np
import pandas as pd
import torch
from src.forecasting.model import DirectionLSTM
from src.forecasting.train import (
    engineer_features, make_sequences, FEATURE_COLS, fit_feature_scaler, train_baseline,
    calibrate_abstain_threshold, temporal_split,
)
from src.forecasting.confidence import predict_with_confidence
from src.config_loader import load_config


def _fake_df(n=150):
    rng = np.random.default_rng(0)
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame({
        "Close": prices,
        "Volume": rng.integers(1000, 5000, n),
    })


def test_engineer_features_produces_expected_columns():
    df = engineer_features(_fake_df())
    for col in FEATURE_COLS:
        assert col in df.columns
    assert "target" in df.columns


def test_make_sequences_shapes():
    df = engineer_features(_fake_df())
    X, y = make_sequences(df, seq_len=20)
    assert X.shape[1] == 20
    assert X.shape[2] == len(FEATURE_COLS)
    assert len(X) == len(y)


def test_predict_with_confidence_returns_expected_keys():
    cfg = load_config()
    model = DirectionLSTM(n_features=len(FEATURE_COLS))
    x = np.random.randn(cfg["data"]["sequence_length"], len(FEATURE_COLS)).astype(np.float32)
    result = predict_with_confidence(model, x, cfg)
    assert result["direction"] in ["UP", "DOWN"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["abstained"], bool)

def test_make_sequences_uses_every_window_and_aligns_label_to_window_end():
    df = engineer_features(_fake_df())
    seq_len = 20
    X, y = make_sequences(df, seq_len=seq_len)
    # one window per possible starting position now, not one fewer
    assert len(X) == len(df) - seq_len + 1
    # the last window's final row must be the engineered frame's actual last row
    np.testing.assert_allclose(X[-1][-1], df[FEATURE_COLS].values[-1].astype(np.float32))
    # its label must be that same last row's own target
    assert y[-1] == df["target"].values[-1]


def test_rsi_and_range_position_features_are_bounded():
    df = engineer_features(_fake_df())
    assert (df["rsi_14"] >= 0).all() and (df["rsi_14"] <= 100).all()
    assert (df["hl_position_20"] >= 0).all() and (df["hl_position_20"] <= 1).all()


def test_fit_feature_scaler_sets_nondefault_mean_and_std():
    model = DirectionLSTM(n_features=len(FEATURE_COLS))
    np.testing.assert_allclose(model.feature_mean.numpy(), np.zeros(len(FEATURE_COLS)))
    np.testing.assert_allclose(model.feature_std.numpy(), np.ones(len(FEATURE_COLS)))

    df = engineer_features(_fake_df())
    X, _ = make_sequences(df, seq_len=20)
    fit_feature_scaler(model, X)

    # at least one feature should now have a non-trivial mean or std — this
    # would fail silently (mean=0, std=1 forever) if the buffers weren't
    # actually being written to.
    assert not np.allclose(model.feature_std.numpy(), np.ones(len(FEATURE_COLS)))


def test_train_baseline_uses_config_driven_hyperparameters():
    df = _fake_df(n=250)
    cfg = {
        "data": {"sequence_length": 20},
        "training": {"epochs": 3, "hidden_size": 8, "num_layers": 1, "dropout": 0.1,
                     "learning_rate": 1e-3, "weight_decay": 0.0, "val_fraction": 0.15},
    }
    result = train_baseline(df, cfg)
    assert result["model"].lstm_layers[0].hidden_size == 8
    assert len(result["model"].lstm_layers) == 1
    assert "best_val_balanced_accuracy" in result
    assert "calibrated_abstain_threshold" in result


def test_temporal_split_leaves_a_gap_so_no_row_is_shared_between_slices():
    seq_len, n = 20, 500
    # X[i] encodes its own window start i, so we can check which raw rows each slice touches.
    X = np.arange(n, dtype=np.float32).reshape(n, 1, 1)
    y = np.zeros(n, dtype=np.int64)

    X_train, _, X_val, _, X_test, _ = temporal_split(X, y, seq_len, val_fraction=0.15)

    def rows_touched(Xs):  # window i uses rows i .. i+seq_len (label looks one day past the window)
        starts = Xs[:, 0, 0].astype(int)
        return starts.min(), starts.max() + seq_len

    train_lo, train_hi = rows_touched(X_train)
    val_lo, val_hi = rows_touched(X_val)
    test_lo, test_hi = rows_touched(X_test)

    assert train_hi < val_lo   # train never touches a day validation uses
    assert val_hi < test_lo    # validation never touches a day test uses
    assert len(X_train) > 0 and len(X_val) > 0 and len(X_test) > 0


def test_calibrate_abstain_threshold_sets_model_buffer_from_val_distribution():
    model = DirectionLSTM(n_features=len(FEATURE_COLS), hidden_size=8, num_layers=1)
    assert abs(float(model.abstain_threshold.item()) - 0.55) < 1e-5  # the un-calibrated default

    df = engineer_features(_fake_df(n=200))
    X, _ = make_sequences(df, seq_len=20)
    cfg = {"confidence": {"abstain_percentile": 25, "mc_dropout_passes": 5}}

    threshold = calibrate_abstain_threshold(model, X, cfg)

    assert threshold != 0.55  # actually recalculated, not left at the default
    assert abs(float(model.abstain_threshold.item()) - threshold) < 1e-5  # buffer matches the returned value


def test_predict_with_confidence_uses_the_models_own_calibrated_threshold():
    cfg = load_config()
    model = DirectionLSTM(n_features=len(FEATURE_COLS))
    x = np.random.randn(cfg["data"]["sequence_length"], len(FEATURE_COLS)).astype(np.float32)

    # Force the model's own threshold far above any reachable confidence —
    # every prediction must abstain, regardless of what cfg says.
    with torch.no_grad():
        model.abstain_threshold.fill_(0.999)
    result = predict_with_confidence(model, x, cfg)
    assert result["abstained"] is True

    # Force it far below — nothing should ever abstain now.
    with torch.no_grad():
        model.abstain_threshold.fill_(0.0)
    result = predict_with_confidence(model, x, cfg)
    assert result["abstained"] is False