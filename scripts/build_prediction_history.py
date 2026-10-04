# scripts/build_prediction_history.py
"""
Backfills prediction_log.jsonl with historical predictions (FR-10/FR-11),
so the dashboard's "Actual vs. Predicted" table has more than a single
day's live snapshot to show. Each entry uses only data available up to its
own as_of_date (no lookahead) — the exact same predict_with_confidence()
call run_daily_cycle.py uses for a live prediction, just applied to past
windows whose outcomes are already known from real history.

Every entry is tagged "source": "backfill" so it's never confused with a
real live daily-cycle run — this is recent real history being surfaced for
evaluation, not the pipeline claiming to have predicted N days in a row
live.
"""
import json
from datetime import datetime

import pandas as pd

from src.config_loader import load_config, resolve_path
from src.forecasting.train import engineer_features, make_sequences
from src.forecasting.confidence import predict_with_confidence
from src.retraining.registry import load_current_model


def build_history(ticker: str, cfg: dict, n_days: int = 60) -> list:
    model, version = load_current_model(ticker, cfg)
    if model is None:
        raise RuntimeError(f"No trained model for {ticker} yet — train one first.")

    hist = pd.read_csv(resolve_path(cfg["paths"]["snapshot_dir"]) / f"{ticker}_historical.csv")
    engineered = engineer_features(hist)
    seq_len = cfg["data"]["sequence_length"]
    X, _ = make_sequences(engineered, seq_len)

    start = max(0, len(X) - n_days)
    entries = []
    for i in range(start, len(X)):
        result = predict_with_confidence(model, X[i], cfg)
        as_of_date = str(engineered.iloc[i + seq_len - 1]["Date"])
        entries.append({
            "date": datetime.now().isoformat(),
            "ticker": ticker,
            "model_version": version,
            "status": "ok",
            "as_of_date": as_of_date,
            "source": "backfill",
            **result,
        })
    return entries


if __name__ == "__main__":
    import sys
    cfg = load_config()
    ticker = sys.argv[1] if len(sys.argv) > 1 else cfg["ticker"]
    n_days = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    entries = build_history(ticker, cfg, n_days)

    path = resolve_path(cfg["paths"]["prediction_log"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    n_abstained = sum(1 for e in entries if e["abstained"])
    print(f"Backfilled {len(entries)} predictions for {ticker} "
          f"({len(entries) - n_abstained} calls made, {n_abstained} abstained). "
          f"Appended to {path}")
