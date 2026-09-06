# README.md
# DriftSense

An explainable, self-retraining ML pipeline with confidence-aware forecasting
for real-time stock market prediction.

## Setup

    pip install -r requirements.txt

## First-time run order

1. **Backfill historical data** (creates `data/snapshots/`):

       python -m scripts.backfill_history

2. **Train the baseline model** (creates `models/v1/`):

       python -m src.forecasting.train

3. **Run the daily cycle** (ingestion → drift detection → classification →
   retraining if needed → attribution → prediction):

       python -m scripts.run_daily_cycle

4. **Run the walk-forward backtest** (writes `logs/backtest_results.json`):

       python -m scripts.run_backtest

5. **Measure drift-detector performance** (NFR-1 latency, NFR-6 false-alarm
   rate — writes `logs/drift_performance_report.json`):

       python -m scripts.measure_drift_performance

6. **Launch the dashboard**:

       streamlit run src/dashboard/app.py

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
| `src/forecasting/train.py` | FR-2 |
| `src/forecasting/confidence.py` | FR-10, FR-11 |
| `src/drift/detector.py` | FR-4 |
| `src/drift/benchmarks.py` | NFR-1, NFR-6 |
| `src/drift/classifier.py` | FR-5 |
| `src/drift/attribution.py` | FR-6 |
| `src/retraining/manager.py` | FR-7 |
| `src/retraining/registry.py` | FR-8 |
| `src/retraining/validator.py` | FR-9 |
| `src/dashboard/app.py` | FR-12 |