# src/retraining/manager.py
"""
Retraining Manager (FR-7).
Orchestrates: assemble training data -> train candidate -> validate against
benchmark -> promote or reject. Logs every attempt (promoted or not) to the
retrain changelog.
"""
import json
import pandas as pd
from datetime import datetime

from src.config_loader import load_config, resolve_path
from src.forecasting.train import train_baseline, save_model
from src.retraining.registry import load_current_model
from src.retraining.validator import validate_candidate


def append_retrain_log(entry: dict, cfg: dict = None):
    cfg = cfg or load_config()
    log_path = resolve_path(cfg["paths"]["retrain_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def assemble_training_data(historical_df: pd.DataFrame, live_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fold the live window that triggered this retraining cycle back into the
    training corpus, so the candidate model actually learns from the data
    that drifted instead of retraining on the same frozen historical snapshot
    every time. When both frames carry a Date column, overlapping dates keep
    the live-window value (it's the freshest source for those rows).
    """
    combined = pd.concat([historical_df, live_df], ignore_index=True)
    if "Date" in combined.columns:
        combined = combined.drop_duplicates(subset="Date", keep="last").sort_values("Date")
    return combined.reset_index(drop=True)


def run_retraining_cycle(historical_df, live_df, benchmark_df, ticker: str, trigger_reason: str, cfg: dict = None) -> dict:
    """
    FR-7/FR-9: the full retrain-validate-promote loop, triggered only when
    the drift classifier has already determined this is a Regime Shift.
    """
    cfg = cfg or load_config()

    current_model, current_version = load_current_model(cfg)
    training_df = assemble_training_data(historical_df, live_df)
    result = train_baseline(training_df, cfg)
    candidate_model = result["model"]

    validation = validate_candidate(candidate_model, current_model, benchmark_df, cfg)
    promoted = validation["passed"]

    new_version = save_model(candidate_model, result["metrics"], ticker, cfg, promoted=promoted)

    log_entry = {
        "version": new_version,
        "trigger_reason": trigger_reason,
        "n_training_rows": len(training_df),
        "accuracy_before": validation["current_metrics"]["accuracy"] if validation["current_metrics"] else None,
        "accuracy_after": validation["candidate_metrics"]["accuracy"],
        "promoted": promoted,
        "timestamp": datetime.now().isoformat(),
    }
    append_retrain_log(log_entry, cfg)

    return log_entry