# src/simulation/runner.py
"""
Runs a synthetic scenario through the REAL DriftSense pipeline, in a sandbox:

    inject scenario -> drift score (FR-4) -> drift gate -> classify (FR-5)
    -> SHAP attribution (FR-6) -> [Regime Shift only] retrain + promotion
    gate (FR-7 / FR-9)

It reuses the same detector, classifier, attribution and retraining code the
daily cycle uses — nothing is mocked — but all retraining output (new model
versions, the retrain log) goes into a throwaway temp directory seeded with a
COPY of the ticker's current model. The real models/, logs/ and prediction log
are never written to, so a demo can't pollute real history.
"""
import copy
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd

from src.config_loader import load_config, resolve_path
from src.drift.attribution import explain_drift
from src.drift.classifier import classify_drift
from src.drift.detector import compute_drift_score, is_drift_detected
from src.forecasting.train import engineer_features, make_sequences
from src.ingestion.yahoo_ingestor import load_live_window
from src.retraining.manager import run_retraining_cycle
from src.retraining.registry import load_current_model
from src.simulation.scenarios import inject_anomaly_spike, inject_regime_shift

SCENARIOS = {"regime_shift": inject_regime_shift, "anomaly_spike": inject_anomaly_spike}


def _outcome_text(drift: dict, classification, retraining) -> str:
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
    return "Drift flagged, but the classifier found no qualifying event, so the model is not retrained."


def run_scenario(ticker: str, scenario: str, cfg: dict = None, seed: int = None,
                 run_retraining: bool = True) -> dict:
    """
    Returns a dict with: sim_df (window incl. synthetic rows, `synthetic` column),
    meta, drift {...detected, threshold}, classification | None, attribution | None,
    retraining | None, outcome (plain-English), seconds.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario '{scenario}' — choose from {list(SCENARIOS)}")
    cfg = cfg or load_config()
    t0 = time.time()

    live_df = load_live_window(ticker, cfg)
    snapshot_df = pd.read_csv(resolve_path(cfg["paths"]["snapshot_dir"]) / f"{ticker}_historical.csv")
    model, version = load_current_model(ticker, cfg)
    if model is None:
        raise RuntimeError(f"No trained model for {ticker} — run scripts.onboard_tickers first.")

    sim_df, meta = SCENARIOS[scenario](live_df, cfg["simulator"], seed=seed)

    drift_result = compute_drift_score(snapshot_df, sim_df)
    drift = {**drift_result, "threshold": cfg["drift"]["threshold"],
             "detected": is_drift_detected(drift_result, cfg)}

    classification = attribution = retraining = None
    if drift["detected"]:
        classification = classify_drift(sim_df, cfg)

        engineered = engineer_features(sim_df)
        X, _ = make_sequences(engineered, cfg["data"]["sequence_length"], labeled_only=False)
        if len(X) > 0:
            attribution = explain_drift(model, X[-10:])

        if run_retraining and classification["classification"] == "Regime Shift":
            retraining = _sandboxed_retrain(ticker, snapshot_df, sim_df, cfg)

    return {
        "ticker": ticker, "scenario": scenario, "model_version": version,
        "sim_df": sim_df, "meta": meta, "drift": drift,
        "classification": classification, "attribution": attribution, "retraining": retraining,
        "outcome": _outcome_text(drift, classification, retraining),
        "seconds": round(time.time() - t0, 1),
    }


def _sandboxed_retrain(ticker: str, snapshot_df: pd.DataFrame, sim_df: pd.DataFrame, cfg: dict) -> dict:
    """Retrain + promotion gate against a COPY of the current model, in a temp dir."""
    with tempfile.TemporaryDirectory(prefix="driftsense_sim_") as tmp:
        tmp = Path(tmp)
        sandbox = copy.deepcopy(cfg)
        sandbox["paths"]["model_dir"] = str(tmp / "models")
        sandbox["paths"]["retrain_log"] = str(tmp / "retrain_log.jsonl")
        sandbox["training"] = {**cfg.get("training", {}), "epochs": cfg["simulator"]["retrain_epochs"]}

        real_ticker_dir = resolve_path(cfg["paths"]["model_dir"]) / ticker
        shutil.copytree(real_ticker_dir, tmp / "models" / ticker)

        benchmark_df = sim_df.tail(cfg["retraining"]["benchmark_window_days"])
        entry = run_retraining_cycle(snapshot_df, sim_df, benchmark_df, ticker,
                                     trigger_reason="SIMULATED Regime Shift", cfg=sandbox)
        return entry
