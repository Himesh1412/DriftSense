# scripts/run_backtest.py
"""
Runs the walk-forward backtest (src/forecasting/backtest.py) over the full
historical snapshot and writes fold-by-fold + summary metrics to
logs/backtest_results/{ticker}.json for later inspection (e.g. in the
exploration notebook or the dashboard). Namespaced per ticker — a single
shared logs/backtest_results.json would silently overwrite one stock's
results with another's, the same class of bug fixed for the model registry
(src/retraining/registry.py) when multi-ticker support was added.
"""
import json
import pandas as pd

from src.config_loader import load_config, resolve_path
from src.forecasting.backtest import walk_forward_backtest

if __name__ == "__main__":
    cfg = load_config()
    ticker = cfg["ticker"]
    snapshot = resolve_path(cfg["paths"]["snapshot_dir"]) / f"{ticker}_historical.csv"
    df = pd.read_csv(snapshot)

    result = walk_forward_backtest(df, cfg)

    out_dir = resolve_path(cfg["paths"]["backtest_log_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ticker}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Ran {len(result['folds'])} folds. Summary: {result['summary']}")
    print(f"Full results written to {out_path}")
