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

## Adding more stocks

One command backfills history, trains that stock's own model, runs the
walk-forward backtest, and backfills a labelled prediction history so the
dashboard has something to show:

    python -m scripts.onboard_tickers              # every ticker in config tickers.*
    python -m scripts.onboard_tickers TSLA INFY.NS # just these

`config/settings.yaml` lists the defaults: the top 10 Nasdaq stocks and 10
large Indian stocks (NSE listings — Yahoo's BSE `.BO` feed returns a single
row for most of them). Each stock is trained independently; a failure on one
is reported and the rest continue.

## Simulation Engine (separate tool, for demos)

Real markets rarely produce a clean regime shift on demand, so the simulation
engine is a standalone test bench — deliberately *not* part of the monitoring
dashboard. From the dashboard, use **Simulation Engine** in the sidebar: it shows
**Open Simulation Engine** (new tab) when the tool is running, or **Start
Simulation Engine** if it isn't. The tool has a "← Monitoring dashboard" link back.
To run it by hand:

    streamlit run src/simulation/app.py --server.port 8502

Pick any stock (a trained one from the list, or type any Yahoo ticker), design
the false data — an **anomaly spike**, a **regime shift** (size, direction,
number of days, turbulence all adjustable), or **type your own daily moves** —
preview exactly what will be injected, then run it through the **real** drift
pipeline (drift score -> classifier with reasoning -> SHAP -> retrain +
promotion gate).

Safety: everything is stamped SIMULATED, injected rows carry a `synthetic`
flag, retraining happens in a throwaway temp folder seeded with a copy of the
model, and a stock that was never onboarded is fetched in memory only. Your
real models, logs and predictions are never written to. A stock without a
trained model still gets the drift score and classification; SHAP and the
retrain step need a model and are skipped with a note.

There is also a command-line version:

    python -m scripts.run_simulation META regime_shift
    python -m scripts.run_simulation INFY.NS anomaly_spike

Defaults live under `simulator:` in `config/settings.yaml`, sized relative to
each stock's own volatility. Two honest findings from building it: the drift
gate only trips on a *sustained* change, so a lone spike in a calm market is
correctly ignored; and the classifier is strict — a steady trend reads as "No
Significant Event", while a Regime Shift takes a lull followed by an
escalating whipsaw.

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
  including ones with the best backtest accuracy of the set, so it abstained on 100% of live predictions. `train.py`'s
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

- **Re-centred UP/DOWN calls plus a balance check.** The LSTM's up-vs-down
  score barely moves from day to day, so a plain "score > 0" rule made some
  stocks call DOWN (or UP) almost every day. `src/forecasting/decision.py`
  calls UP when today's score is above the median of the model's own previous
  60 scores (past data only), and training/backtesting report the share of UP
  calls. A model whose held-out calls fall outside 30-70% UP is retrained from
  a fresh start (up to twice) and flagged if it stays lopsided.

## Honest accuracy

Walk-forward backtest, 6 folds x 100 test days (~600 predictions per stock),
all 20 configured stocks, trained with the same routine production uses:

| Group | Direction accuracy |
|---|---|
| Nasdaq top 10 | 50.9% |
| NSE top 10 | 49.9% |
| All 20 | 50.4% (11 stocks above 50%, 9 below; range 46.1-54.1%) |

That is a coin flip. One stock's score carries about +/-2 points of sampling
noise, so the best results (COST 54.1%, AVGO 53.3%) are about what luck gives
among 20 stocks. Daily direction of liquid large caps is close to efficiently
priced, and these 7 price/volume features do not change that. The project's
value is the pipeline around the model: drift detection, explainable
regime-vs-anomaly classification, gated retraining, confidence-aware
abstention and the simulation engine - not a trading edge. Earlier figures of
57-62% came from label leakage, a last-day label bug and picking the best
stock, and have been removed. A first version of the backtest scored only ~11
days per 30-day fold (the lookback days were being discarded), which made
individual stocks swing between 41% and 63%; that is fixed. Reproduce with
`python -m scripts.run_backtest AAPL MSFT ...`.

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