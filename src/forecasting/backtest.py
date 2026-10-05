# src/forecasting/backtest.py
"""
Walk-Forward Backtest (FR-2, evaluation).
Trains and evaluates a fresh candidate over multiple rolling folds instead
of a single 80/20 split, to check whether the direction classifier holds up
across different time periods rather than one lucky/unlucky split. This is
a reporting/evaluation tool — it does not replace train_baseline, which is
still what retraining/manager.py uses for the fast production retrain cycle.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.config_loader import load_config
from src.forecasting.decision import balance_report, calls_with_context
from src.forecasting.train import engineer_features, make_sequences, fit_model, split_train_val


def _train_one_fold(X_train, y_train, epochs: int, cfg: dict = None):
    """Train a fold with the SAME routine production uses (train.fit_model), so the backtest
    measures what actually ships — including checkpoint selection and the decision offset."""
    cfg = cfg or {}
    seq_len = cfg.get("data", {}).get("sequence_length", 20)
    X_tr, y_tr, X_val, y_val = split_train_val(X_train, y_train, seq_len, cfg.get("training", {}).get("val_fraction", 0.15))
    model, _ = fit_model(X_tr, y_tr, X_val, y_val, cfg, epochs, calibrate_abstain=False)
    return model


def _summarize_folds(fold_results: list) -> dict:
    if not fold_results:
        return {}
    summary = {}
    for m in ["accuracy", "precision", "recall", "f1"]:
        vals = [f[m] for f in fold_results]
        summary[f"{m}_mean"] = round(float(np.mean(vals)), 4)
        summary[f"{m}_std"] = round(float(np.std(vals)), 4)
    summary["up_share_mean"] = round(float(np.mean([f["up_share"] for f in fold_results])), 4)
    summary["lopsided_folds"] = int(sum(not f["balanced"] for f in fold_results))
    return summary


def walk_forward_backtest(df, cfg: dict = None, epochs: int = None) -> dict:
    """
    Expanding-window walk-forward backtest: fold k trains on rows [0, train_end_k)
    and evaluates on the following `test_window_days` rows, with train_end_k spaced
    evenly between min_train_days and the last point that leaves a full test window.
    Every labelled day in the test window gets scored (~test_window_days predictions per fold):
    each test window borrows its seq_len-1 lookback days from the rows just before the test
    period. That is only past INPUT data a live call would also have; every training label is
    for a day before the test period, so nothing is leaked. (Slicing the test rows on their
    own used to throw away the first seq_len-1 days, leaving ~11 predictions per 30-day fold.)
    """
    cfg = cfg or load_config()
    bt_cfg = cfg["backtest"]
    seq_len = cfg["data"]["sequence_length"]
    epochs = epochs if epochs is not None else cfg.get("training", {}).get("epochs", 50)

    engineered = engineer_features(df)
    n = len(engineered)

    min_train = bt_cfg["min_train_days"]
    test_window = bt_cfg["test_window_days"]
    n_folds = bt_cfg["n_folds"]

    max_start = n - test_window
    if max_start <= min_train:
        raise ValueError(
            f"Not enough data for backtest config: need > {min_train + test_window} rows, got {n}."
        )

    train_ends = np.linspace(min_train, max_start, num=n_folds, dtype=int)

    fold_results = []
    for fold_idx, train_end in enumerate(train_ends):
        train_end = int(train_end)
        test_end = min(train_end + test_window, n)
        if test_end - train_end < 1:
            continue

        train_slice = engineered.iloc[:train_end].reset_index(drop=True)
        test_slice = engineered.iloc[train_end - seq_len + 1:test_end].reset_index(drop=True)  # lookback + test days

        X_train, y_train = make_sequences(train_slice, seq_len)
        X_test, y_test = make_sequences(test_slice, seq_len)
        if len(X_train) == 0 or len(X_test) == 0:
            continue

        model = _train_one_fold(X_train, y_train, epochs, cfg)
        tcfg = cfg.get("training", {})
        preds = calls_with_context(model, X_train, X_test, tcfg.get("balance_reference_window", 60))
        balance = balance_report(preds, tcfg.get("balance_min_up_share", 0.30), tcfg.get("balance_max_up_share", 0.70))

        fold_results.append({
            "fold": fold_idx,
            "train_end_idx": train_end,
            "test_start_idx": train_end,
            "test_end_idx": test_end,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "accuracy": round(float(accuracy_score(y_test, preds)), 4),
            "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
            "up_share": balance["up_share"],
            "balanced": balance["balanced"],
        })

    return {"folds": fold_results, "summary": _summarize_folds(fold_results)}