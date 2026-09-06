# tests/test_scheduler.py
from unittest.mock import patch
from scripts.scheduler import run_scheduled_cycle


def test_run_scheduled_cycle_does_not_raise_when_daily_cycle_fails():
    """SRS §5.1 Reliability: a failed cycle must be caught and logged, not crash the scheduler loop."""
    with patch("scripts.scheduler.run_daily_cycle", side_effect=RuntimeError("Yahoo Finance is down")):
        run_scheduled_cycle()  # must not raise


def test_run_scheduled_cycle_calls_run_daily_cycle_once_on_success():
    with patch("scripts.scheduler.run_daily_cycle") as mock_run:
        run_scheduled_cycle()
        mock_run.assert_called_once()
