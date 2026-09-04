# tests/test_ingestion.py
import pandas as pd
from unittest.mock import patch
from src.ingestion.yahoo_ingestor import fetch_historical


def _fake_yf_download(*args, **kwargs):
    """Stand-in for yf.download so the test doesn't hit the network."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "Open": range(10), "High": range(10), "Low": range(10),
        "Close": range(10), "Volume": range(1000, 1010),
    }, index=dates)
    df.index.name = "Date"
    return df


def test_fetch_historical_returns_expected_columns():
    with patch("src.ingestion.yahoo_ingestor.yf.download", side_effect=_fake_yf_download):
        df = fetch_historical("AAPL", start="2024-01-01", end="2024-02-01")
    for col in ["Date", "Open", "High", "Low", "Close", "Volume"]:
        assert col in df.columns
    assert len(df) == 10