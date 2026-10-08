# scripts/onboard_tickers.py
"""
One command to make any stock(s) ready to show: backfill history -> train the
per-ticker model -> walk-forward backtest -> backfilled prediction history.

    python -m scripts.onboard_tickers                # every ticker in config tickers.*
    python -m scripts.onboard_tickers TSLA INFY.NS   # just these

Each ticker is isolated: a failure (e.g. no data from Yahoo) is reported and the
rest continue. Results land in logs/onboarding/<first ticker>.json.
"""
import json
import os
import sys
import time

import torch

from src.config_loader import load_config, resolve_path
from src.ingestion.yahoo_ingestor import fetch_historical, save_snapshot, update_live_window
from src.forecasting.train import train_baseline, save_model
from src.forecasting.backtest import walk_forward_backtest
from scripts.build_prediction_history import build_history

MIN_ROWS = 400  # enough for train + validation + test + backtest folds


def onboard(ticker: str, cfg: dict) -> dict:
    t0 = time.time()
    hist = fetch_historical(ticker, cfg["data"]["historical_start"])
    if len(hist) < MIN_ROWS:
        raise RuntimeError(f"only {len(hist)} rows of history from Yahoo (need {MIN_ROWS}+)")
    save_snapshot(hist, ticker, cfg)
    update_live_window(ticker, cfg)  # so the dashboard can show this ticker straight away

    trained = train_baseline(hist, cfg)
    version = save_model(trained["model"], trained["metrics"], ticker, cfg, training_data=trained["training_data"])

    backtest = walk_forward_backtest(hist, cfg)
    out_dir = resolve_path(cfg["paths"]["backtest_log_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{ticker}.json").write_text(json.dumps(backtest, indent=2))

    entries = build_history(ticker, cfg, n_days=60)
    log_path = resolve_path(cfg["paths"]["prediction_log"])
    with open(log_path, "a") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    resolved_calls = [e for e in entries if not e["abstained"]]
    return {
        "ticker": ticker, "rows": len(hist), "model_version": version,
        "single_split_metrics": trained["metrics"],
        "backtest_summary": backtest["summary"],
        "calibrated_abstain_threshold": trained["calibrated_abstain_threshold"],
        "history_calls_made": len(resolved_calls), "history_abstained": len(entries) - len(resolved_calls),
        "seconds": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    if os.environ.get("DRIFTSENSE_THREADS"):  # lets several shards run side by side
        torch.set_num_threads(int(os.environ["DRIFTSENSE_THREADS"]))
    cfg = load_config()
    tickers = sys.argv[1:] or [t for group in cfg["tickers"].values() for t in group]

    summary = []
    for t in tickers:
        print(f"=== {t} ===", flush=True)
        try:
            r = onboard(t, cfg)
            summary.append(r)
            print(f"  v{r['model_version']}  backtest acc={r['backtest_summary'].get('accuracy_mean')}  "
                  f"split acc={r['single_split_metrics']['accuracy']}  ({r['seconds']}s)", flush=True)
        except Exception as e:
            summary.append({"ticker": t, "error": str(e)})
            print(f"  FAILED: {e}", flush=True)

    out = resolve_path(f"logs/onboarding/{tickers[0]}.json")  # one file per shard, no write clashes
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {out}")
