# scripts/measure_drift_performance.py
"""
Runs the drift detector performance benchmarks (src/drift/benchmarks.py) and
writes a report satisfying NFR-1 (latency) and NFR-6 (false-alarm rate) —
the SRS requires both be "measured and reported, not assumed."
"""
import json
import pandas as pd

from src.config_loader import load_config, resolve_path
from src.drift.benchmarks import measure_false_alarm_rate, measure_latency

if __name__ == "__main__":
    cfg = load_config()
    ticker = cfg["ticker"]
    window = cfg["data"]["live_window_days"]
    baseline_df = pd.read_csv(resolve_path(cfg["paths"]["snapshot_dir"]) / f"{ticker}_historical.csv")

    latency_report = measure_latency(baseline_df, cfg, window)
    false_alarm_report = measure_false_alarm_rate(baseline_df, cfg, window)

    report = {
        "ticker": ticker,
        "nfr1_latency": latency_report,
        "nfr6_false_alarm_rate": false_alarm_report,
    }

    out_path = resolve_path(cfg["paths"]["drift_performance_log"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("NFR-1 (latency):", latency_report)
    print("NFR-6 (false-alarm rate):", false_alarm_report)
    print(f"Report written to {out_path}")
