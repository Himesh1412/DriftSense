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
from src.forecasting.model import DirectionLSTM
from src.forecasting.train import (
    FEATURE_COLS, engineer_features, make_sequences, compute_class_weights, fit_feature_scaler,
)


def _train_one_fold(X_train, y_train, epochs: int, cfg: dict = None) -> DirectionLSTM:
    tcfg = (cfg or {}).get("training", {})
    model = DirectionLSTM(n_features=len(FEATURE_COLS), hidden_size=tcfg.get("hidden_size", 64),
                           num_layers=tcfg.get("num_layers", 2), dropout=tcfg.get("dropout", 0.3))
    fit_feature_scaler(model, X_train)

    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.get("learning_rate", 1e-3),
                                  weight_decay=tcfg.get("weight_decay", 1e-4))
    loss_fn = nn.CrossEntropyLoss(weight=compute_class_weights(y_train))

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    return model


def _summarize_folds(fold_results: list) -> dict:
    if not fold_results:
        return {}
    summary = {}
    for m in ["accuracy", "precision", "recall", "f1"]:
        vals = [f[m] for f in fold_results]
        summary[f"{m}_mean"] = round(float(np.mean(vals)), 4)
        summary[f"{m}_std"] = round(float(np.std(vals)), 4)
    return summary


def walk_forward_backtest(df, cfg: dict = None, epochs: int = None) -> dict:
    """
    Expanding-window walk-forward backtest: fold k trains on rows [0, train_end_k)
    and evaluates on the following `test_window_days` rows, with train_end_k spaced
    evenly between min_train_days and the last point that leaves a full test window.
    Test windows never overlap training rows for that fold, so there's no leakage.
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
        if test_end - train_end < seq_len:
            continue  # not enough rows in this fold's test window for even one sequence

        train_slice = engineered.iloc[:train_end].reset_index(drop=True)
        test_slice = engineered.iloc[train_end:test_end].reset_index(drop=True)

        X_train, y_train = make_sequences(train_slice, seq_len)
        X_test, y_test = make_sequences(test_slice, seq_len)
        if len(X_train) == 0 or len(X_test) == 0:
            continue

        model = _train_one_fold(X_train, y_train, epochs, cfg)
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test)).argmax(dim=1).numpy()

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
        })

    return {"folds": fold_results, "summary": _summarize_folds(fold_results)}