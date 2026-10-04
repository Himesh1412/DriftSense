# tests/test_simulation.py
import numpy as np
import pandas as pd
import pytest

from src.config_loader import load_config
from src.drift.classifier import classify_drift
from src.drift.detector import compute_drift_score
from src.simulation.scenarios import inject_anomaly_spike, inject_regime_shift

CFG = load_config()
SIM = CFG["simulator"]


def _prices(n, sigma, seed, start="2025-01-01"):
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, sigma, n))
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d")
    return pd.DataFrame({"Date": dates, "Close": close, "High": close, "Low": close, "Open": close,
                         "Volume": rng.integers(1_000_000, 3_000_000, n).astype(float)})


@pytest.fixture
def live():
    return _prices(90, 0.015, seed=1, start="2026-05-01")


@pytest.fixture
def snapshot():
    return _prices(1500, 0.011, seed=2)


@pytest.mark.parametrize("inject", [inject_regime_shift, inject_anomaly_spike])
def test_window_length_unchanged_and_synthetic_rows_are_tagged_at_the_end(live, inject):
    sim_df, meta = inject(live, SIM, seed=0)
    assert len(sim_df) == len(live)
    assert sim_df["synthetic"].sum() == meta["n_synthetic"] > 0
    flags = sim_df["synthetic"].tolist()
    assert flags == sorted(flags)  # all real rows first, then all synthetic rows


@pytest.mark.parametrize("inject", [inject_regime_shift, inject_anomaly_spike])
def test_remaining_real_rows_are_untouched(live, inject):
    sim_df, _ = inject(live, SIM, seed=0)
    real_part = sim_df[~sim_df["synthetic"]]
    original_tail = live.tail(len(real_part)).reset_index(drop=True)
    pd.testing.assert_series_equal(real_part["Close"].reset_index(drop=True), original_tail["Close"])


@pytest.mark.parametrize("inject", [inject_regime_shift, inject_anomaly_spike])
def test_synthetic_dates_continue_after_the_last_real_date(live, inject):
    sim_df, _ = inject(live, SIM, seed=0)
    synth_dates = pd.to_datetime(sim_df.loc[sim_df["synthetic"], "Date"])
    assert synth_dates.min() > pd.to_datetime(live["Date"].iloc[-1])
    assert synth_dates.is_monotonic_increasing


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_regime_shift_trips_both_gates_and_is_classified_regime_shift(live, snapshot, seed):
    sim_df, _ = inject_regime_shift(live, SIM, seed=seed)
    assert compute_drift_score(snapshot, sim_df)["drift_score"] > CFG["drift"]["threshold"]
    assert classify_drift(sim_df, CFG)["classification"] == "Regime Shift"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_anomaly_spike_trips_drift_gate_but_is_classified_anomaly_spike(live, snapshot, seed):
    sim_df, _ = inject_anomaly_spike(live, SIM, seed=seed)
    assert compute_drift_score(snapshot, sim_df)["drift_score"] > CFG["drift"]["threshold"]
    result = classify_drift(sim_df, CFG)
    assert result["classification"] == "Anomaly Spike"
    assert result["reasoning"]  # the plain-English "why" is present


def test_a_lone_spike_in_a_calm_market_does_not_trip_the_drift_gate(live, snapshot):
    """Honest behavior worth pinning down: no background turbulence -> the pipeline ignores it."""
    calm = dict(SIM, spike_buildup_days=0)
    sim_df, _ = inject_anomaly_spike(live, calm, seed=0)
    assert compute_drift_score(snapshot, sim_df)["drift_score"] < CFG["drift"]["threshold"]


def test_spike_reverts_to_roughly_the_pre_shock_price(live):
    sim_df, meta = inject_anomaly_spike(live, dict(SIM, spike_buildup_days=0, spike_calm_days=0), seed=0)
    synthetic = sim_df[sim_df["synthetic"]]["Close"].tolist()
    pre_shock = float(live["Close"].iloc[-1])
    assert abs(synthetic[-1] / pre_shock - 1) < 0.05  # shock of ~16 sigma, reverted to within 5%


def test_custom_hand_typed_moves_are_appended_exactly(live):
    from src.simulation.scenarios import inject_custom_returns
    sim_df, meta = inject_custom_returns(live, [-10, 5, -20])
    synth = sim_df[sim_df["synthetic"]]["Close"].tolist()
    last_real = float(live["Close"].iloc[-1])
    assert len(synth) == 3 and meta["n_synthetic"] == 3
    assert synth[0] == pytest.approx(last_real * 0.90)
    assert synth[1] == pytest.approx(last_real * 0.90 * 1.05)
    assert synth[2] == pytest.approx(last_real * 0.90 * 1.05 * 0.80)


def test_custom_moves_are_capped_so_a_typo_cannot_make_a_negative_price(live):
    from src.simulation.scenarios import inject_custom_returns
    sim_df, _ = inject_custom_returns(live, [-500, 900])
    assert (sim_df["Close"] > 0).all()
    assert sim_df[sim_df["synthetic"]]["Close"].iloc[0] == pytest.approx(float(live["Close"].iloc[-1]) * 0.55)


def test_custom_mode_requires_at_least_one_move(live):
    from src.simulation.scenarios import inject_custom_returns
    with pytest.raises(ValueError):
        inject_custom_returns(live, [])


def test_shock_direction_can_be_a_jump_instead_of_a_drop(live):
    drop, _ = inject_anomaly_spike(live, dict(SIM, spike_buildup_days=0, spike_calm_days=0, spike_direction="drop"), seed=0)
    jump, _ = inject_anomaly_spike(live, dict(SIM, spike_buildup_days=0, spike_calm_days=0, spike_direction="jump"), seed=0)
    last = float(live["Close"].iloc[-1])
    assert drop[drop["synthetic"]]["Close"].iloc[0] < last < jump[jump["synthetic"]]["Close"].iloc[0]
