# scripts/run_backtest.py
"""
Runs the walk-forward backtest (src/forecasting/backtest.py) over the full
historical snapshot and writes fold-by-fold + summary metrics to
logs/backtest_results/{ticker}.json for later inspection (e.g. in the
exploration notebook or the dashboard). Namespaced per ticker — a single
shared logs/backtest_results.json would silently overwrite one stock's
results with another's, the same class of bug fixed for the model registry
(src/retraining/registry.py) when multi-ticker support was added.

    python -m scripts.run_backtest                 # the default ticker in config
    python -m scripts.run_backtest AAPL INFY.NS    # just these (re-measures; no model is retrained)

DRIFTSENSE_THREADS=2 caps torch threads so several runs can share one machine.
"""
import json
import os
import sys

import pandas as pd
import torch

from src.config_loader import load_config, resolve_path
from src.forecasting.backtest import walk_forward_backtest


def backtest_ticker(ticker: str, cfg: dict) -> dict:
    snapshot = resolve_path(cfg["paths"]["snapshot_dir"]) / f"{ticker}_historical.csv"
    result = walk_forward_backtest(pd.read_csv(snapshot), cfg)

    out_dir = resolve_path(cfg["paths"]["backtest_log_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    if os.environ.get("DRIFTSENSE_THREADS"):
        torch.set_num_threads(int(os.environ["DRIFTSENSE_THREADS"]))
    cfg = load_config()
    for ticker in sys.argv[1:] or [cfg["ticker"]]:
        result = backtest_ticker(ticker, cfg)
        print(f"{ticker}: ran {len(result['folds'])} folds. Summary: {result['summary']}", flush=True)
