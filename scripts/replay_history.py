# scripts/replay_history.py
"""
Replays the drift detector and classifier over each stock's REAL history, one 90-day window at a time,
exactly as the daily cycle would have seen it on each past day (SRS section 7: the classifier should be
shown separating regime shifts from anomaly spikes, with documented reasoning).

    python -m scripts.replay_history              # every configured stock
    python -m scripts.replay_history AAPL TSLA    # just these

Nothing is simulated. Writes logs/history_replay.json: per-class counts, distinct episodes (runs of
consecutive flagged windows merged into one), and each episode's date, stock, drift score and the
classifier's own reasoning. A stock's first 250 trading days are skipped so there is a baseline to compare to.
"""
import json
import sys
from collections import Counter

import pandas as pd

from src.config_loader import load_config, resolve_path
from src.drift.classifier import classify_drift
from src.drift.detector import compute_drift_score

STEP = 3          # test every 3rd trading day (windows overlap heavily, so every day adds nothing)
WARMUP = 250      # trading days of history required before the first window


def replay_ticker(ticker: str, cfg: dict) -> list:
    window = cfg["data"]["live_window_days"]
    df = pd.read_csv(resolve_path(cfg["paths"]["snapshot_dir"]) / f"{ticker}_historical.csv")
    flags = []
    for end in range(WARMUP, len(df), STEP):
        live, baseline = df.iloc[end - window:end].reset_index(drop=True), df.iloc[:end - window]
        drift = compute_drift_score(baseline, live)
        if drift["drift_score"] > cfg["drift"]["threshold"]:
            c = classify_drift(live, cfg)
            flags.append({"ticker": ticker, "date": str(df["Date"].iloc[end - 1])[:10], "end": end,
                          "drift_score": drift["drift_score"], "classification": c["classification"],
                          "reasoning": c.get("reasoning", "")})
    return flags


def to_episodes(flags: list) -> list:
    """Merge consecutive flagged windows of the same stock and class into one episode."""
    episodes = []
    for f in flags:
        last = episodes[-1] if episodes else None
        if last and last["ticker"] == f["ticker"] and last["classification"] == f["classification"] \
                and f["end"] - last["_end"] <= 2 * STEP:
            last["_end"], last["last_date"] = f["end"], f["date"]
            last["peak_drift_score"] = max(last["peak_drift_score"], f["drift_score"])
            last["windows"] += 1
        else:
            episodes.append({"ticker": f["ticker"], "first_date": f["date"], "last_date": f["date"], "_end": f["end"],
                             "classification": f["classification"], "peak_drift_score": f["drift_score"],
                             "windows": 1, "reasoning": f["reasoning"]})
    for e in episodes:
        e.pop("_end")
    return episodes


if __name__ == "__main__":
    cfg = load_config()
    tickers = sys.argv[1:] or [t for g in cfg["tickers"].values() for t in g]
    all_flags = []
    for t in tickers:
        all_flags += replay_ticker(t, cfg)
        print(f"{t}: done", flush=True)
    episodes = to_episodes(all_flags)
    report = {
        "stocks": tickers, "window_days": cfg["data"]["live_window_days"], "drift_threshold": cfg["drift"]["threshold"],
        "flagged_windows_by_class": dict(Counter(f["classification"] for f in all_flags)),
        "episodes_by_class": dict(Counter(e["classification"] for e in episodes)),
        "episodes": episodes,
    }
    out = resolve_path("logs/history_replay.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print("\nFlagged windows:", report["flagged_windows_by_class"])
    print("Distinct episodes:", report["episodes_by_class"])
    print(f"Written to {out}")
