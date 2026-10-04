# README.md
# DriftSense

An explainable, self-retraining ML pipeline with confidence-aware forecasting
for real-time stock market prediction.

> **Disclaimer:** DriftSense is a course/portfolio project, not a financial
> product. It does not provide financial advice and must not be used to make
> real trading or investment decisions. Predictions may abstain, may be
> wrong, and the underlying model has not been validated for production
> trading use. (SRS §5.5 Safety.)

## Setup

    pip install -r requirements.txt

## First-time run order

1. **Backfill historical data** (creates `data/snapshots/`):

       python -m scripts.backfill_history

2. **Train the baseline model** (creates `models/{ticker}/v1/` — models are
   namespaced per ticker, so training a second stock never overwrites or
   gets confused with another's):

       python -m src.forecasting.train

3. **Run the daily cycle** (ingestion → drift detection → classification →
   retraining if needed → attribution → prediction):

       python -m scripts.run_daily_cycle

4. **Run the walk-forward backtest** (writes `logs/backtest_results/{ticker}.json`):

       python -m scripts.run_backtest

5. **Measure drift-detector performance** (NFR-1 latency, NFR-6 false-alarm
   rate — writes `logs/drift_performance_report.json`):

       python -m scripts.measure_drift_performance

6. **Launch the dashboard**:

       streamlit run src/dashboard/app.py

## Scheduling (FR-3)

Nothing runs on its own until you set up a schedule — by default you have
to invoke each script yourself.

**Option A — simplest, run in a terminal you leave open:**

    python -m scripts.scheduler

This runs the daily cycle automatically every day at the time set by
`schedule.daily_run_time` in `config/settings.yaml` (default `17:30`), for
as long as this process keeps running. Closing the terminal stops it.

**Option B — a real background task that survives reboots and closed
terminals.** This registers a persistent OS-level task, so it's left as a
manual step for you to run yourself rather than something set up
automatically:

*Windows (Task Scheduler), run once from an elevated PowerShell prompt:*

    schtasks /create /tn "DriftSense Daily Cycle" /tr "python -m scripts.run_daily_cycle" /sc daily /st 17:30 /sd (Get-Date).ToString('MM/dd/yyyy')

To remove it later: `schtasks /delete /tn "DriftSense Daily Cycle" /f`

*Linux/macOS (cron), add a line via `crontab -e`:*

    30 17 * * * cd /path/to/driftsense && python -m scripts.run_daily_cycle

## Configuration

All thresholds (drift, confidence, retraining) live in `config/settings.yaml` —
edit that file rather than hardcoding values in code.

## Design Rationale

- **Direction, not price.** The forecaster is a binary up/down classifier,
  not a price regressor — daily price regression targets are noisy and give
  a weak R², while direction gives clean precision/recall/F1 (see
  `src/forecasting/model.py` and `src/forecasting/train.py`).
- **Statistical drift detection, not ML.** Drift is detected with KS-test /
  PSI, not another model — deliberately cheap and interpretable
  (`src/drift/detector.py`).
- **Heuristic regime-vs-anomaly classifier.** `src/drift/classifier.py` is
  an explicit magnitude/duration/reversion heuristic, not a verified
  detector — see its own docstring for the documented limitation.
- **Promotion gate.** A retrained candidate is only promoted if it doesn't
  underperform the current model on a held-out benchmark
  (`src/retraining/validator.py`) — this gate is never bypassed.
- **Per-model calibrated confidence threshold, not one fixed global number.**
  MC-Dropout confidence turned out to cluster in a narrow, model-specific
  band (e.g. 0.49-0.52) regardless of a model's real backtested accuracy —
  a fixed `abstain_below: 0.55` was unreachable for every ticker tried,
  including ones with genuinely good backtest accuracy (META: 60%+ across 6
  folds), so it abstained on 100% of live predictions. `train.py`'s
  `calibrate_abstain_threshold()` now sets each model's own
  `abstain_threshold` (a buffer saved inside the checkpoint itself) from
  that model's own validation-set confidence distribution — abstaining on
  the bottom `confidence.abstain_percentile` (default 25%) least-confident
  predictions, rather than comparing to a magnitude that assumed a scale
  this architecture doesn't actually produce. This is framed as "flag the
  relatively least-sure calls", not "the kept tier is proven more
  accurate" — at this data scale, confidence magnitude and correctness were
  only weakly/noisily correlated, so the dashboard's Backtest Evidence
  section is the more honest place to look for whether a model actually
  works, not any single day's live reading.

## Exploration notebook

`notebooks/exploration.ipynb` sweeps `drift.threshold`,
`spike_magnitude_std`/`regime_min_duration_days`, and
`confidence.abstain_below` against real historical data to justify the
values in `config/settings.yaml` empirically rather than by feel.

## Running tests

    pytest tests/

## Module-to-requirement map

| Module | Requirement |
|---|---|
| `src/ingestion/` | FR-1, FR-3 |
| `scripts/scheduler.py` | FR-3 |
| `src/forecasting/train.py` | FR-2 |
| `src/forecasting/confidence.py` | FR-10, FR-11 |
| `src/drift/detector.py` | FR-4 |
| `src/drift/benchmarks.py` | NFR-1, NFR-6 |
| `src/drift/classifier.py` | FR-5 |
| `src/drift/attribution.py` | FR-6 |
| `src/retraining/manager.py` | FR-7 |
| `src/retraining/registry.py` | FR-8 (per-ticker namespaced) |
| `src/retraining/validator.py` | FR-9 |
| `src/dashboard/app.py` | FR-12 |