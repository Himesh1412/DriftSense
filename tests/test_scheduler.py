# tests/test_scheduler.py
from unittest.mock import patch
from scripts.scheduler import run_scheduled_cycle

TWO_STOCKS = {"tickers": {"nasdaq": ["AAPL"], "india": ["INFY.NS"]}}


def test_run_scheduled_cycle_does_not_raise_when_daily_cycle_fails():
    """SRS §5.1 Reliability: a failed cycle must be caught and logged, not crash the scheduler loop."""
    with patch("scripts.scheduler.load_config", return_value=TWO_STOCKS),          patch("scripts.scheduler.run_daily_cycle", side_effect=RuntimeError("Yahoo Finance is down")):
        run_scheduled_cycle()  # must not raise


def test_run_scheduled_cycle_runs_every_configured_stock():
    with patch("scripts.scheduler.load_config", return_value=TWO_STOCKS),          patch("scripts.scheduler.run_daily_cycle") as mock_run:
        run_scheduled_cycle()
        assert [c.args[0] for c in mock_run.call_args_list] == ["AAPL", "INFY.NS"]


def test_one_stock_failing_does_not_stop_the_others():
    def flaky(ticker):
        if ticker == "AAPL":
            raise RuntimeError("no data")
    with patch("scripts.scheduler.load_config", return_value=TWO_STOCKS),          patch("scripts.scheduler.run_daily_cycle", side_effect=flaky) as mock_run:
        run_scheduled_cycle()
        assert mock_run.call_count == 2
