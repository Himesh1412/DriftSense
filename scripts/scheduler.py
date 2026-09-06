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
import time
from datetime import datetime

import schedule

from src.config_loader import load_config
from scripts.run_daily_cycle import run_daily_cycle


def run_scheduled_cycle():
    print(f"[{datetime.now().isoformat()}] Running scheduled daily cycle...")
    try:
        run_daily_cycle()
        print(f"[{datetime.now().isoformat()}] Daily cycle completed.")
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] Daily cycle failed: {e}. Will retry at the next scheduled run.")


if __name__ == "__main__":
    cfg = load_config()
    run_time = cfg.get("schedule", {}).get("daily_run_time", "17:30")

    schedule.every().day.at(run_time).do(run_scheduled_cycle)

    print(f"Scheduler started — will run the daily cycle every day at {run_time} (local time).")
    print("Leave this process running. Press Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(30)
