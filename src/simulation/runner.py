# src/simulation/runner.py
"""
Runs a synthetic scenario through the REAL DriftSense pipeline, in a sandbox:

    inject false data -> drift score (FR-4) -> drift gate -> classify (FR-5)
    -> SHAP attribution (FR-6) -> [Regime Shift only] retrain + promotion
    gate (FR-7 / FR-9)

It reuses the same detector, classifier, attribution and retraining code the
daily cycle uses — nothing is mocked — but all retraining output (new model
versions, the retrain log) goes into a throwaway temp directory seeded with a
COPY of the ticker's current model. The real models/, logs/ and prediction log
are never written to.

Works for any stock Yahoo Finance has: if it hasn't been onboarded (no saved
data on disk) its history is fetched in memory only and nothing is saved. A
stock with no trained model still gets the drift score and classification;
SHAP attribution and the retrain demo need a trained model and are skipped
with a clear note.
"""
import copy
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd

from src.config_loader import load_config, resolve_path
from src.drift.classifier import classify_drift
from src.drift.detector import compute_drift_score, is_drift_detected
from src.forecasting.train import engineer_features, make_sequences
from src.ingestion.yahoo_ingestor import fetch_historical, fetch_live_window
from src.retraining.manager import run_retraining_cycle
from src.retraining.registry import load_current_model
from src.simulation.scenarios import inject_anomaly_spike, inject_custom_returns, inject_regime_shift

SCENARIOS = {"regime_shift": inject_regime_shift, "anomaly_spike": inject_anomaly_spike}


def load_inputs(ticker: str, cfg: dict):
    """(live_window, full_history) from disk if the stock was onboarded, else fetched in memory only."""
    live_path = resolve_path(cfg["paths"]["live_dir"]) / f"{ticker}_live.csv"
    snap_path = resolve_path(cfg["paths"]["snapshot_dir"]) / f"{ticker}_historical.csv"
    live_df = pd.read_csv(live_path) if live_path.exists() else fetch_live_window(ticker, cfg=cfg)
    snapshot_df = pd.read_csv(snap_path) if snap_path.exists() else fetch_historical(ticker, cfg["data"]["historical_start"])
    for df in (live_df, snapshot_df):
        df["Date"] = df["Date"].astype(str).str[:10]
    if len(live_df) < 30 or len(snapshot_df) < 250:
        raise RuntimeError(f"Not enough price history for '{ticker}' (live {len(live_df)} rows, "
                           f"history {len(snapshot_df)} rows). Check the ticker symbol.")
    return live_df, snapshot_df


def _outcome_text(drift: dict, classification, retraining, has_model: bool) -> str:
    if not drift["detected"]:
        return (f"Drift score {drift['drift_score']} is below the {drift['threshold']} threshold, "
                f"so the pipeline takes no action.")
    label = classification["classification"]
    if label == "Anomaly Spike":
        return ("Drift flagged and classified as an Anomaly Spike: it is logged and flagged, but the "
                "model is NOT retrained (by design — a short-lived spike shouldn't be learned from).")
    if label == "Regime Shift" and retraining is not None:
        verdict = "PROMOTED to current model" if retraining["promoted"] else "REJECTED (current model kept)"
        return (f"Drift flagged and classified as a Regime Shift: a candidate was retrained on history plus "
                f"the new regime and benchmarked against the current model "
                f"(candidate {retraining['accuracy_after']} vs current {retraining['accuracy_before']}) "
                f"-> {verdict} by the promotion gate.")
    if label == "Regime Shift" and not has_model:
        return ("Drift flagged and classified as a Regime Shift. This stock has no trained model yet, so the "
                "retrain-and-promote step was skipped (run scripts.onboard_tickers to enable it).")
    if label == "Regime Shift":
        return "Drift flagged and classified as a Regime Shift (retraining step not requested)."
    return "Drift flagged, but the classifier found no qualifying event, so the model is not retrained."


def analyze_window(ticker: str, sim_df: pd.DataFrame, meta: dict, snapshot_df: pd.DataFrame,
                   cfg: dict, run_retraining: bool = True) -> dict:
    """Push an already-injected window through the real pipeline (see module docstring)."""
    t0 = time.time()
    model, version = load_current_model(ticker, cfg)

    drift_result = compute_drift_score(snapshot_df, sim_df)
    drift = {**drift_result, "threshold": cfg["drift"]["threshold"],
             "detected": is_drift_detected(drift_result, cfg)}

    classification = attribution = retraining = None
    if drift["detected"]:
        classification = classify_drift(sim_df, cfg)

        if model is not None:
            from src.drift.attribution import explain_drift  # lazy: SHAP is slow to import
            engineered = engineer_features(sim_df)
            X, _ = make_sequences(engineered, cfg["data"]["sequence_length"], labeled_only=False)
            if len(X) > 0:
                attribution = explain_drift(model, X[-10:])

            if run_retraining and classification["classification"] == "Regime Shift":
                retraining = _sandboxed_retrain(ticker, snapshot_df, sim_df, cfg)

    return {
        "ticker": ticker, "scenario": meta["scenario"], "model_version": version, "has_model": model is not None,
        "sim_df": sim_df, "meta": meta, "drift": drift,
        "classification": classification, "attribution": attribution, "retraining": retraining,
        "outcome": _outcome_text(drift, classification, retraining, model is not None),
        "seconds": round(time.time() - t0, 1),
    }


def run_scenario(ticker: str, scenario: str, cfg: dict = None, seed: int = None,
                 run_retraining: bool = True, sim_cfg: dict = None, custom_moves_pct: list = None) -> dict:
    """
    Inject + analyze in one call. scenario is 'regime_shift', 'anomaly_spike', or 'custom'
    (with custom_moves_pct, e.g. [-3, 2, -6, 8]). sim_cfg overrides cfg['simulator'].
    """
    if scenario not in (*SCENARIOS, "custom"):
        raise ValueError(f"unknown scenario '{scenario}' — choose from {[*SCENARIOS, 'custom']}")
    cfg = cfg or load_config()
    sim_cfg = sim_cfg or cfg["simulator"]

    live_df, snapshot_df = load_inputs(ticker, cfg)
    if scenario == "custom":
        sim_df, meta = inject_custom_returns(live_df, custom_moves_pct, sim_cfg, seed=seed)
    else:
        sim_df, meta = SCENARIOS[scenario](live_df, sim_cfg, seed=seed)
    return analyze_window(ticker, sim_df, meta, snapshot_df, cfg, run_retraining)


def _sandboxed_retrain(ticker: str, snapshot_df: pd.DataFrame, sim_df: pd.DataFrame, cfg: dict) -> dict:
    """Retrain + promotion gate against a COPY of the current model, in a temp dir."""
    with tempfile.TemporaryDirectory(prefix="driftsense_sim_") as tmp:
        tmp = Path(tmp)
        sandbox = copy.deepcopy(cfg)
        sandbox["paths"]["model_dir"] = str(tmp / "models")
        sandbox["paths"]["retrain_log"] = str(tmp / "retrain_log.jsonl")
        sandbox["registry"] = {"git_tag_versions": False}  # a simulated model must never tag the real repo
        sandbox["training"] = {**cfg.get("training", {}), "epochs": cfg["simulator"]["retrain_epochs"]}

        real_ticker_dir = resolve_path(cfg["paths"]["model_dir"]) / ticker
        shutil.copytree(real_ticker_dir, tmp / "models" / ticker)

        benchmark_df = sim_df.tail(cfg["retraining"]["benchmark_window_days"])
        entry = run_retraining_cycle(snapshot_df, sim_df, benchmark_df, ticker,
                                     trigger_reason="SIMULATED Regime Shift", cfg=sandbox)
        return entry
