# scripts/run_daily_cycle.py
"""
The scheduled entry point (FR-1 through FR-12, tied together).
This is the direct code implementation of the sequence diagram:
Scheduler -> Ingestion -> Drift Detection -> Classification ->
[Regime Shift -> Retraining]  |  [Anomaly -> skip retrain]
-> Attribution -> Forecasting + Confidence -> Dashboard logs
"""
import json
import numpy as np
import pandas as pd
from datetime import datetime

from src.config_loader import load_config, resolve_path
from src.ingestion.yahoo_ingestor import update_live_window, load_live_window
from src.drift.detector import compute_drift_score, is_drift_detected
from src.drift.classifier import classify_drift
from src.drift.attribution import explain_drift
from src.retraining.manager import run_retraining_cycle
from src.retraining.registry import load_current_model
from src.forecasting.train import engineer_features, make_sequences
from src.forecasting.confidence import predict_with_confidence
from src.forecasting.decision import latest_reference, logit_gaps
from src.retraining.manager import assemble_training_data


def append_drift_log(entry: dict, cfg: dict):
    path = resolve_path(cfg["paths"]["drift_log"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def append_prediction_log(entry: dict, cfg: dict):
    """
    FR-11/FR-12: persist every prediction (or abstained/no-model/no-data
    state) so the dashboard can display it, not just this script's console
    output. Every cycle appends exactly one entry here, regardless of which
    branch below produced it — the dashboard reads the last line.
    """
    path = resolve_path(cfg["paths"]["prediction_log"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def run_daily_cycle():
    cfg = load_config()
    ticker = cfg["ticker"]

    # 1-3: trigger + fetch + pass data window
    update_live_window(ticker, cfg)
    live_df = load_live_window(ticker, cfg)
    snapshot_df = pd.read_csv(resolve_path(cfg["paths"]["snapshot_dir"]) / f"{ticker}_historical.csv")

    # 4: compute drift score
    drift_result = compute_drift_score(snapshot_df, live_df)
    print(f"Drift score: {drift_result['drift_score']}")

    if is_drift_detected(drift_result, cfg):
        # 5-6: classify
        classification = classify_drift(live_df, cfg)
        print(f"Classification: {classification}")

        current_model, _ = load_current_model(ticker, cfg)

        if classification["classification"] == "Regime Shift" and current_model is not None:
            # 7-8, 10-12, 14: retrain (on historical + live data), validate, promote if passed
            benchmark_df = live_df.tail(cfg["retraining"]["benchmark_window_days"])
            retrain_result = run_retraining_cycle(
                snapshot_df, live_df, benchmark_df, ticker, trigger_reason="Regime Shift", cfg=cfg
            )
            print(f"Retraining result: {retrain_result}")

        # 9/13: attribution always runs for any flagged drift
        if current_model is not None:
            engineered = engineer_features(live_df)
            X, _ = make_sequences(engineered, cfg["data"]["sequence_length"], labeled_only=False)
            if len(X) > 0:
                attribution = explain_drift(current_model, X[-10:])
            else:
                attribution = {"explanation_text": "Not enough live data yet for attribution."}
        else:
            attribution = {"explanation_text": "No trained model yet — skipping attribution."}

        # 15: publish drift explanation
        append_drift_log({
            "date": datetime.now().isoformat(),
            "classification": classification["classification"],
            "drift_score": drift_result["drift_score"],
            # why the classifier called it Regime Shift vs Anomaly Spike
            # (the magnitude/duration heuristic's own reasoning) — distinct
            # from "cause"/"explanation" below, which is SHAP explaining
            # which *feature* the model's behavior is most associated with.
            "classification_reason": classification.get("reasoning", ""),
            "cause": attribution.get("top_feature", "n/a"),
            "explanation": attribution["explanation_text"],
        }, cfg)

    # 16-18: load current model, read recent window, generate prediction + confidence
    current_model, version = load_current_model(ticker, cfg)
    if current_model is not None:
        # The live window alone is too short to know the model's recent typical score, so build
        # the windows from full history + live window (see forecasting/decision.py).
        engineered = engineer_features(assemble_training_data(snapshot_df, live_df))
        X, _ = make_sequences(engineered, cfg["data"]["sequence_length"], labeled_only=False)
        if len(X) > 0:
            n_ref = cfg.get("training", {}).get("balance_reference_window", 60)
            reference = latest_reference(logit_gaps(current_model, X[-(n_ref + 1):]), n_ref)
            result = predict_with_confidence(current_model, X[-1], cfg, reference=reference)
            # 19-20: publish prediction or abstained state
            print(f"Prediction: {result}")
            append_prediction_log({
                "date": datetime.now().isoformat(),
                "ticker": ticker,
                "model_version": version,
                "status": "ok",
                # the actual trading day this prediction's input window ends
                # on — NOT necessarily today, if there's a weekend/holiday/
                # ingestion lag. This is what "actual vs predicted" must
                # anchor to, not the timestamp the cycle happened to run at.
                "as_of_date": str(live_df["Date"].iloc[-1]),
                **result,
            }, cfg)
        else:
            print("Not enough live data yet to generate a prediction.")
            append_prediction_log({
                "date": datetime.now().isoformat(),
                "ticker": ticker,
                "model_version": version,
                "status": "insufficient_data",
            }, cfg)
    else:
        print("No trained model exists yet — run scripts/backfill_history.py then src/forecasting/train.py first.")
        append_prediction_log({
            "date": datetime.now().isoformat(),
            "ticker": ticker,
            "model_version": None,
            "status": "no_model",
        }, cfg)


if __name__ == "__main__":
    run_daily_cycle()