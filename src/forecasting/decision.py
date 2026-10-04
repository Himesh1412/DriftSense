# src/forecasting/decision.py
"""
Turning the model's raw output into an UP/DOWN call, and checking that the calls
aren't lopsided.

Why this exists: the LSTM's UP-vs-DOWN score (logit_up - logit_down) turned out to
be almost constant — e.g. AAPL's score had a spread of 0.011 around an average of
-0.012 — so which side it landed on was decided by a tiny constant lean, not by the
data. That produced "DOWN every time" (AAPL) or "UP every time" (AVGO), and an
"accuracy" that was really just how often the market went that way recently.

The fix is to call UP when today's score is above the model's OWN recent typical
score (the median of its previous N scores), not above zero. That is causal (only
past scores are used), keeps whatever relative signal the model has, removes the
constant lean, and tracks slow drift in the score's level.
"""
import numpy as np
import torch


def logit_gaps(model, X: np.ndarray) -> np.ndarray:
    """Deterministic (dropout off) score per window: logit_up - logit_down. Positive leans UP."""
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
    return (logits[:, 1] - logits[:, 0]).numpy()


def rolling_reference(gaps: np.ndarray, start: int, n_ref: int) -> np.ndarray:
    """For each index i in [start, len(gaps)): the median of the n_ref scores BEFORE i."""
    return np.array([np.median(gaps[max(0, i - n_ref):i]) if i > 0 else 0.0
                     for i in range(start, len(gaps))])


def latest_reference(gaps: np.ndarray, n_ref: int) -> float:
    """Reference for the newest window: the median of the n_ref scores before it."""
    if len(gaps) < 2:
        return 0.0
    return float(np.median(gaps[max(0, len(gaps) - 1 - n_ref):len(gaps) - 1]))


def decide(gaps: np.ndarray, reference) -> np.ndarray:
    """1 (UP) where the score is above its reference, else 0 (DOWN)."""
    return ((np.asarray(gaps) - reference) > 0).astype(int)


def fit_decision_offset(model, X_val: np.ndarray, train_up_rate: float) -> float:
    """
    Static fallback reference, stored on the model: the validation-score quantile that makes
    the model call UP about as often as UP days occurred in training. Used where no recent
    score history exists (e.g. a short benchmark window). Fit on validation data only.
    """
    offset = 0.0
    if len(X_val) > 0:
        offset = float(np.quantile(logit_gaps(model, X_val), 1 - train_up_rate))
    with torch.no_grad():
        model.decision_offset.fill_(offset)
    return offset


def balance_report(preds: np.ndarray, lo: float, hi: float) -> dict:
    """Label-free check that a set of calls isn't lopsided: the share called UP must sit in [lo, hi]."""
    share = float(np.mean(preds)) if len(preds) else 0.5
    return {"up_share": round(share, 4), "balanced": bool(lo <= share <= hi)}


def calls_with_context(model, X_before: np.ndarray, X_eval: np.ndarray, n_ref: int) -> np.ndarray:
    """
    UP/DOWN calls for X_eval, each judged against the median of the model's previous n_ref
    scores. X_before = the windows that immediately precede X_eval in time, so the first
    evaluated windows have history to compare against (scores only, no labels, only the past).
    """
    before = logit_gaps(model, X_before[-n_ref:]) if len(X_before) else np.array([])
    gaps = np.concatenate([before, logit_gaps(model, X_eval)])
    return decide(gaps[len(before):], rolling_reference(gaps, start=len(before), n_ref=n_ref))
