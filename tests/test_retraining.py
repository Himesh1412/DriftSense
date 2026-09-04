# tests/test_retraining.py
import numpy as np
import pandas as pd
from src.forecasting.train import train_baseline
from src.retraining.validator import validate_candidate
from src.retraining.manager import assemble_training_data

def _fake_df(n=200):
    rng = np.random.default_rng(0)
    prices = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame({"Close": prices, "Volume": rng.integers(1000, 5000, n)})


def test_first_candidate_always_passes_when_no_current_model():
    df = _fake_df()
    result = train_baseline(df, epochs=2)
    validation = validate_candidate(result["model"], None, df.tail(60))
    assert validation["passed"] is True
    assert validation["current_metrics"] is None
    
def test_assemble_training_data_merges_and_prefers_live_on_overlap():
    historical = pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "Close": [100, 101, 102], "Volume": [1000, 1000, 1000],
    })
    live = pd.DataFrame({
        "Date": ["2024-01-03", "2024-01-04"],
        "Close": [102.5, 103], "Volume": [1000, 1000],
    })
    combined = assemble_training_data(historical, live)
    assert list(combined["Date"]) == ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    assert combined.loc[combined["Date"] == "2024-01-03", "Close"].iloc[0] == 102.5


def test_assemble_training_data_without_date_column_just_concatenates():
    historical = _fake_df(50)
    live = _fake_df(10)
    combined = assemble_training_data(historical, live)
    assert len(combined) == 60