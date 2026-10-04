# src/retraining/validator.py
"""
Regression Benchmark Gate (FR-9).
A retrained candidate is only promoted if it does not underperform the
currently promoted model on a shared, held-out benchmark window.
"""
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.config_loader import load_config
from src.forecasting.decision import decide, logit_gaps
from src.forecasting.train import engineer_features, make_sequences


def evaluate_on_benchmark(model, benchmark_df, cfg: dict = None) -> dict:
    cfg = cfg or load_config()
    seq_len = cfg["data"]["sequence_length"]

    engineered = engineer_features(benchmark_df)
    X, y = make_sequences(engineered, seq_len)
    if len(X) == 0:
        return {"accuracy": 0.0, "f1": 0.0}

    # A 60-day benchmark is too short to build a recent-score history, so each model is judged
    # with its own stored decision offset (see forecasting/decision.py) — the same treatment for
    # candidate and current model, so the comparison stays like-for-like.
    preds = decide(logit_gaps(model, X), float(model.decision_offset.item()))

    return {
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
    }


def validate_candidate(candidate_model, current_model, benchmark_df, cfg: dict = None) -> dict:
    """
    FR-9: candidate must not underperform current model (min_improvement, default 0.0)
    on accuracy over the shared benchmark window.
    """
    cfg = cfg or load_config()
    min_improvement = cfg["retraining"]["min_improvement"]

    candidate_metrics = evaluate_on_benchmark(candidate_model, benchmark_df, cfg)

    if current_model is None:
        # no existing model to beat — first-ever training always passes
        return {"passed": True, "candidate_metrics": candidate_metrics, "current_metrics": None}

    current_metrics = evaluate_on_benchmark(current_model, benchmark_df, cfg)
    passed = (candidate_metrics["accuracy"] - current_metrics["accuracy"]) >= min_improvement

    return {
        "passed": passed,
        "candidate_metrics": candidate_metrics,
        "current_metrics": current_metrics,
    }