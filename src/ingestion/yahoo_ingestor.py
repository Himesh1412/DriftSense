# src/ingestion/yahoo_ingestor.py
"""
Data Ingestion Module (FR-1, FR-3)
Fetches historical and live OHLCV data via the Yahoo Finance API and
maintains the rolling live-data window (D1).
"""
import pandas as pd
import yfinance as yf
from pathlib import Path
from src.config_loader import load_config, resolve_path


def fetch_historical(ticker: str, start: str, end: str = None) -> pd.DataFrame:
    """FR-1: pull historical daily OHLCV data for a ticker."""
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    df = df.reset_index()
    df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
    return df


def save_snapshot(df: pd.DataFrame, ticker: str, cfg: dict = None) -> Path:
    """Freeze a reproducible CSV snapshot of historical data."""
    cfg = cfg or load_config()
    out_dir = resolve_path(cfg["paths"]["snapshot_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker}_historical.csv"
    df.to_csv(path, index=False)
    return path


def fetch_live_window(ticker: str, days: int = None, cfg: dict = None) -> pd.DataFrame:
    """FR-3: pull the most recent `days` of daily data as the rolling live window."""
    cfg = cfg or load_config()
    days = days or cfg["data"]["live_window_days"]
    df = yf.download(ticker, period=f"{days}d", interval="1d", progress=False, auto_adjust=True)
    df = df.reset_index()
    df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
    return df


def update_live_window(ticker: str, cfg: dict = None) -> Path:
    """Fetch the latest live window and persist it to data/live/ (D1)."""
    cfg = cfg or load_config()
    df = fetch_live_window(ticker, cfg=cfg)
    live_dir = resolve_path(cfg["paths"]["live_dir"])
    live_dir.mkdir(parents=True, exist_ok=True)
    path = live_dir / f"{ticker}_live.csv"
    df.to_csv(path, index=False)
    return path


def load_live_window(ticker: str, cfg: dict = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    path = resolve_path(cfg["paths"]["live_dir"]) / f"{ticker}_live.csv"
    if not path.exists():
        update_live_window(ticker, cfg)
    return pd.read_csv(path)


if __name__ == "__main__":
    cfg = load_config()
    ticker = cfg["ticker"]
    hist = fetch_historical(ticker, cfg["data"]["historical_start"])
    save_snapshot(hist, ticker, cfg)
    update_live_window(ticker, cfg)
    print(f"Ingested {len(hist)} historical rows and refreshed the live window for {ticker}.")