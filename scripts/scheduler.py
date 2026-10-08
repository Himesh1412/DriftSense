# scripts/scheduler.py
"""
Automated scheduler (FR-3): runs scripts.run_daily_cycle on a recurring
schedule, so polling/drift-detection/retraining happens without a human
manually invoking each cycle.

This is an in-process scheduler — it must be left running (foreground
terminal, tmux, screen, or similar) for the schedule to fire. It does not
register anything with the OS. If you want it to survive a reboot and run
even when you're not watching a terminal, register it as a real background
task yourself (see README.md's "Scheduling" section for the exact
Windows Task Scheduler / cron commands) — that's a persistent system change
so it's left as something you opt into deliberately, not something this
script does on its own.

A failed cycle (e.g. Yahoo Finance API downtime) is caught and logged
rather than crashing the scheduler loop (SRS §5.1 Reliability) — it simply
waits for the next scheduled run instead of taking the whole process down.
"""
import json
import time
from datetime import datetime

import schedule

from src.config_loader import load_config, resolve_path
from scripts.run_daily_cycle import run_daily_cycle


def log_cycle_error(ticker: str, error: Exception, cfg: dict, final: bool):
    """SRS 5 Reliability: failures are logged (to a file, not just the console), then retried."""
    path = resolve_path(cfg["paths"]["cycle_error_log"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(), "ticker": ticker, "error": str(error),
                            "will_retry": not final}) + "\n")


def run_scheduled_cycle():
    """
    Run the daily cycle for every configured stock. One stock failing never stops the others: failures
    are logged to a file and retried once after schedule.retry_delay_seconds; anything still failing is
    logged as final and left for the next scheduled run.
    """
    cfg = load_config()
    tickers = [t for group in cfg["tickers"].values() for t in group]
    print(f"[{datetime.now().isoformat()}] Running scheduled daily cycle for {len(tickers)} stocks...")
    failed = []
    for ticker in tickers:
        try:
            run_daily_cycle(ticker)
        except Exception as e:
            print(f"[{datetime.now().isoformat()}] {ticker} failed: {e}")
            log_cycle_error(ticker, e, cfg, final=False)
            failed.append(ticker)
    if failed:
        delay = cfg.get("schedule", {}).get("retry_delay_seconds", 300)
        print(f"[{datetime.now().isoformat()}] Retrying {len(failed)} failed stock(s) in {delay}s...")
        time.sleep(delay)
        for ticker in failed:
            try:
                run_daily_cycle(ticker)
            except Exception as e:
                print(f"[{datetime.now().isoformat()}] {ticker} failed again: {e}. Left for the next scheduled run.")
                log_cycle_error(ticker, e, cfg, final=True)
    print(f"[{datetime.now().isoformat()}] Daily cycle finished.")


if __name__ == "__main__":
    cfg = load_config()
    run_time = cfg.get("schedule", {}).get("daily_run_time", "17:30")

    schedule.every().day.at(run_time).do(run_scheduled_cycle)

    print(f"Scheduler started — will run the daily cycle every day at {run_time} (local time).")
    print("Leave this process running. Press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(30)
