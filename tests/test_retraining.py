# tests/test_retraining.py
import numpy as np
import pandas as pd
from src.forecasting.train import train_baseline, save_model
from src.retraining.validator import validate_candidate
from src.retraining.manager import assemble_training_data
from src.retraining.registry import load_current_model, get_model_history

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


def test_models_are_isolated_per_ticker(tmp_path):
    """
    Regression test for the bug found while adding multi-ticker support:
    all tickers used to share one flat models/v{n}/ sequence, so training
    a second ticker's model would silently become the "current" model for
    every other ticker too. Models must now be namespaced per ticker.
    """
    cfg = {"data": {"sequence_length": 20}, "paths": {"model_dir": str(tmp_path / "models")}}
    result = train_baseline(_fake_df(), cfg, epochs=2)

    save_model(result["model"], result["metrics"], "AAPL", cfg)
    save_model(result["model"], result["metrics"], "GOOG", cfg)
    save_model(result["model"], result["metrics"], "GOOG", cfg)  # GOOG's second version

    _, aapl_version = load_current_model("AAPL", cfg)
    _, goog_version = load_current_model("GOOG", cfg)

    assert aapl_version == 1
    assert goog_version == 2
    assert len(get_model_history("AAPL", cfg)) == 1
    assert len(get_model_history("GOOG", cfg)) == 2

def test_model_with_non_default_architecture_round_trips_through_the_registry(tmp_path):
    """Regression: load_model used to rebuild every model at the default size, so a checkpoint
    trained with any other hidden_size/num_layers couldn't be loaded back."""
    cfg = {
        "data": {"sequence_length": 20},
        "paths": {"model_dir": str(tmp_path / "models")},
        "training": {"epochs": 2, "hidden_size": 8, "num_layers": 1, "dropout": 0.1,
                     "learning_rate": 1e-3, "weight_decay": 0.0, "val_fraction": 0.15},
    }
    result = train_baseline(_fake_df(), cfg)
    save_model(result["model"], result["metrics"], "TINY", cfg)

    model, version = load_current_model("TINY", cfg)
    assert version == 1
    assert model.lstm_layers[0].hidden_size == 8 and len(model.lstm_layers) == 1


def test_checkpoints_saved_before_decision_offset_existed_still_load(tmp_path):
    """Models trained earlier have no decision_offset buffer; they must load with a neutral 0."""
    import json
    import torch
    cfg = {"data": {"sequence_length": 20}, "paths": {"model_dir": str(tmp_path / "models")}}
    result = train_baseline(_fake_df(), cfg, epochs=2)
    save_model(result["model"], result["metrics"], "OLD", cfg)

    ckpt = tmp_path / "models" / "OLD" / "v1" / "model.pt"
    state = torch.load(ckpt)
    del state["decision_offset"]            # make it look like an older checkpoint
    torch.save(state, ckpt)

    model, version = load_current_model("OLD", cfg)
    assert version == 1 and float(model.decision_offset.item()) == 0.0
