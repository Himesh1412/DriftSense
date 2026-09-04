# tests/test_forecasting.py
import numpy as np
import pandas as pd
from src.forecasting.model import DirectionLSTM
from src.forecasting.train import engineer_features, make_sequences, FEATURE_COLS
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