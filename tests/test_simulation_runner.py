# tests/test_simulation_runner.py
import copy

import numpy as np
import pandas as pd
import pytest

from src.config_loader import load_config
from src.forecasting.train import save_model, train_baseline
from src.simulation.runner import run_scenario


def _prices(n, sigma, seed, start):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, sigma, n))
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d")
    return pd.DataFrame({"Date": dates, "Close": close, "High": close, "Low": close, "Open": close,
                         "Volume": rng.integers(1_000_000, 3_000_000, n).astype(float)})


@pytest.fixture
def env(tmp_path):
    """A self-contained fake stock ('FAKE'): history, live window, and one trained tiny model."""
    cfg = copy.deepcopy(load_config())
    for key, name in [("snapshot_dir", "snapshots"), ("live_dir", "live"), ("model_dir", "models")]:
        cfg["paths"][key] = str(tmp_path / name)
    cfg["training"] = {"epochs": 2, "hidden_size": 8, "num_layers": 1, "dropout": 0.1,
                       "learning_rate": 1e-3, "weight_decay": 0.0, "val_fraction": 0.15}
    cfg["confidence"]["mc_dropout_passes"] = 3
    cfg["simulator"]["retrain_epochs"] = 2

    history = _prices(1500, 0.011, seed=2, start="2020-01-01")
    live = _prices(90, 0.015, seed=1, start="2026-05-01")
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "live").mkdir()
    history.to_csv(tmp_path / "snapshots" / "FAKE_historical.csv", index=False)
    live.to_csv(tmp_path / "live" / "FAKE_live.csv", index=False)

    trained = train_baseline(history, cfg)
    save_model(trained["model"], trained["metrics"], "FAKE", cfg)
    return cfg, tmp_path


def test_regime_shift_goes_all_the_way_through_retrain_and_promotion_gate(env):
    cfg, tmp = env
    result = run_scenario("FAKE", "regime_shift", cfg, seed=0)

    assert result["drift"]["detected"] is True
    assert result["classification"]["classification"] == "Regime Shift"
    assert result["attribution"]["top_feature"]
    assert result["retraining"] is not None
    assert isinstance(result["retraining"]["promoted"], bool)  # the gate made a real decision
    assert "Regime Shift" in result["outcome"]
    assert result["sim_df"]["synthetic"].any()


def test_anomaly_spike_is_flagged_but_never_retrains(env):
    cfg, _ = env
    result = run_scenario("FAKE", "anomaly_spike", cfg, seed=0)

    assert result["drift"]["detected"] is True
    assert result["classification"]["classification"] == "Anomaly Spike"
    assert result["retraining"] is None  # FR-7: only a Regime Shift retrains
    assert "NOT retrained" in result["outcome"]


def test_below_drift_threshold_means_no_classification_and_no_action(env):
    cfg, _ = env
    cfg["simulator"]["spike_buildup_days"] = 0  # lone spike in a calm market
    result = run_scenario("FAKE", "anomaly_spike", cfg, seed=0)

    assert result["drift"]["detected"] is False
    assert result["classification"] is None and result["attribution"] is None and result["retraining"] is None
    assert "no action" in result["outcome"]


def test_simulation_never_writes_to_the_real_model_registry_or_logs(env):
    cfg, tmp = env
    before = sorted(p.relative_to(tmp).as_posix() for p in tmp.rglob("*") if p.is_file())
    run_scenario("FAKE", "regime_shift", cfg, seed=0)
    after = sorted(p.relative_to(tmp).as_posix() for p in tmp.rglob("*") if p.is_file())

    assert before == after  # no new model versions, no log files, nothing changed
    assert [p.name for p in (tmp / "models" / "FAKE").glob("v*")] == ["v1"]


def test_unknown_scenario_is_rejected(env):
    cfg, _ = env
    with pytest.raises(ValueError):
        run_scenario("FAKE", "meteor_strike", cfg)


def test_custom_typed_moves_run_through_the_pipeline(env):
    cfg, _ = env
    result = run_scenario("FAKE", "custom", cfg, custom_moves_pct=[-8, 7, -9, 8, -10, 9, -12])
    assert result["scenario"] == "custom"
    assert result["meta"]["n_synthetic"] == 7
    assert "drift" in result and "outcome" in result


def test_user_supplied_sim_settings_override_the_config(env):
    cfg, _ = env
    gentle = dict(cfg["simulator"], spike_sigmas=0.5, spike_buildup_days=0)
    result = run_scenario("FAKE", "anomaly_spike", cfg, seed=0, sim_cfg=gentle)
    assert result["drift"]["detected"] is False  # a 0.5-sigma "spike" in a calm market is nothing


def test_stock_without_a_trained_model_still_gets_drift_and_classification(env):
    cfg, tmp = env
    for sub, name in [("snapshots", "historical"), ("live", "live")]:
        src = tmp / sub / f"FAKE_{name}.csv"
        (tmp / sub / f"NOMODEL_{name}.csv").write_text(src.read_text())

    result = run_scenario("NOMODEL", "regime_shift", cfg, seed=0)

    assert result["has_model"] is False and result["model_version"] is None
    assert result["drift"]["detected"] is True
    assert result["classification"]["classification"] == "Regime Shift"
    assert result["attribution"] is None and result["retraining"] is None
    assert "no trained model" in result["outcome"]
