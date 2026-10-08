# tests/test_scheduler.py
import json
from unittest.mock import patch

from scripts.scheduler import run_scheduled_cycle

CFG = {"tickers": {"nasdaq": ["AAPL"], "india": ["INFY.NS"]}, "schedule": {"retry_delay_seconds": 0},
       "paths": {"cycle_error_log": "logs/cycle_errors.jsonl"}}


def _run(tmp_path, side_effect=None):
    cfg = {**CFG, "paths": {"cycle_error_log": str(tmp_path / "errors.jsonl")}}
    with patch("scripts.scheduler.load_config", return_value=cfg), \
         patch("scripts.scheduler.time.sleep"), \
         patch("scripts.scheduler.run_daily_cycle", side_effect=side_effect) as mock_run:
        run_scheduled_cycle()
    return mock_run


def test_run_scheduled_cycle_does_not_raise_when_daily_cycle_fails(tmp_path):
    """SRS 5 Reliability: a failed cycle must be caught and logged, not crash the scheduler loop."""
    _run(tmp_path, side_effect=RuntimeError("Yahoo Finance is down"))  # must not raise


def test_run_scheduled_cycle_runs_every_configured_stock(tmp_path):
    mock_run = _run(tmp_path)
    assert [c.args[0] for c in mock_run.call_args_list] == ["AAPL", "INFY.NS"]


def test_one_stock_failing_does_not_stop_the_others(tmp_path):
    def flaky(ticker):
        if ticker == "AAPL":
            raise RuntimeError("no data")
    mock_run = _run(tmp_path, side_effect=flaky)
    assert "INFY.NS" in [c.args[0] for c in mock_run.call_args_list]


def test_failures_are_logged_to_a_file_and_retried_once(tmp_path):
    mock_run = _run(tmp_path, side_effect=RuntimeError("down"))
    assert [c.args[0] for c in mock_run.call_args_list] == ["AAPL", "INFY.NS", "AAPL", "INFY.NS"]  # one retry pass
    rows = [json.loads(l) for l in (tmp_path / "errors.jsonl").read_text().splitlines()]
    assert [r["will_retry"] for r in rows] == [True, True, False, False]
    assert {r["ticker"] for r in rows} == {"AAPL", "INFY.NS"}


def test_a_stock_that_recovers_on_retry_is_not_logged_as_final(tmp_path):
    seen = {"AAPL": 0}

    def recovers(ticker):
        if ticker == "AAPL":
            seen["AAPL"] += 1
            if seen["AAPL"] == 1:
                raise RuntimeError("blip")
    _run(tmp_path, side_effect=recovers)
    rows = [json.loads(l) for l in (tmp_path / "errors.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["will_retry"] is True
