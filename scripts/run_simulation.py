# scripts/run_simulation.py
"""
Runs a SYNTHETIC drift scenario through the real pipeline, in a sandbox that
never touches your real models or logs (see src/simulation/runner.py).

    python -m scripts.run_simulation                       # regime_shift on the configured ticker
    python -m scripts.run_simulation META anomaly_spike
    python -m scripts.run_simulation INFY.NS regime_shift 3   # optional seed

Everything it prints is simulated, not market data.
"""
import sys

from src.config_loader import load_config
from src.simulation.runner import SCENARIOS, run_scenario


def print_report(r: dict):
    d, c, a, rt = r["drift"], r["classification"], r["attribution"], r["retraining"]
    print("\n" + "=" * 72)
    print(f"  SIMULATION (synthetic data, not real market data)  -  {r['ticker']}  -  {r['scenario']}")
    print("=" * 72)
    print(f"Scenario : {r['meta']['description']}")
    print(f"Window   : {len(r['sim_df'])} days, {r['meta']['n_synthetic']} of them synthetic")
    print(f"\n[FR-4] Drift score {d['drift_score']}  (KS {d['ks_statistic']}, PSI {d['psi']})  "
          f"threshold {d['threshold']}  ->  {'DRIFT DETECTED' if d['detected'] else 'below threshold'}")
    if c:
        print(f"[FR-5] Classified: {c['classification']}  (magnitude {c['magnitude']}, duration {c['duration_days']} days)")
        print(f"       Why: {c['reasoning']}")
    if a:
        print(f"[FR-6] SHAP: {a['explanation_text']}")
    if rt:
        print(f"[FR-7/9] Retrained v{rt['version']} on {rt['n_training_rows']} rows: "
              f"candidate acc {rt['accuracy_after']} vs current {rt['accuracy_before']} "
              f"-> {'PROMOTED' if rt['promoted'] else 'REJECTED'}")
    print(f"\nOutcome  : {r['outcome']}")
    print(f"(took {r['seconds']}s; sandboxed — no real models, logs or predictions were changed)\n")


if __name__ == "__main__":
    cfg = load_config()
    ticker = sys.argv[1] if len(sys.argv) > 1 else cfg["ticker"]
    scenario = sys.argv[2] if len(sys.argv) > 2 else "regime_shift"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else None
    if scenario not in SCENARIOS:
        sys.exit(f"scenario must be one of {list(SCENARIOS)}")
    print_report(run_scenario(ticker, scenario, cfg, seed=seed))
