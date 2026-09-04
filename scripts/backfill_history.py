# scripts/backfill_history.py
"""
One-time historical data pull (populates data/snapshots/ for first training run).
"""
from src.config_loader import load_config
from src.ingestion.yahoo_ingestor import fetch_historical, save_snapshot

if __name__ == "__main__":
    cfg = load_config()
    ticker = cfg["ticker"]
    df = fetch_historical(ticker, cfg["data"]["historical_start"])
    path = save_snapshot(df, ticker, cfg)
    print(f"Saved {len(df)} rows of historical data to {path}")