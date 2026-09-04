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

4. **Launch the dashboard**:

       streamlit run src/dashboard/app.py

## Configuration

All thresholds (drift, confidence, retraining) live in `config/settings.yaml` —
edit that file rather than hardcoding values in code.

## Running tests

    pytest tests/

## Module-to-requirement map

| Module | Requirement |
|---|---|
| `src/ingestion/` | FR-1, FR-3 |
| `src/forecasting/train.py` | FR-2 |
| `src/forecasting/confidence.py` | FR-10, FR-11 |
| `src/drift/detector.py` | FR-4 |
| `src/drift/classifier.py` | FR-5 |
| `src/drift/attribution.py` | FR-6 |
| `src/retraining/manager.py` | FR-7 |
| `src/retraining/registry.py` | FR-8 |
| `src/retraining/validator.py` | FR-9 |
| `src/dashboard/app.py` | FR-12 |