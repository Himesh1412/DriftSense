# src/forecasting/train.py
"""
Baseline model training (FR-2).
Feature engineering: returns, moving averages, volatility (not raw prices).
Trains a direction classifier and reports precision/recall/F1.
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from src.config_loader import load_config, resolve_path
from src.forecasting.model import DirectionLSTM

FEATURE_COLS = ["return_1d", "ma_5", "ma_20", "volatility_10d", "volume_z"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw OHLCV into model-ready features (never feed raw price directly)."""
    df = df.copy()
    df["return_1d"] = df["Close"].pct_change()
    df["ma_5"] = df["Close"].rolling(5).mean() / df["Close"] - 1
    df["ma_20"] = df["Close"].rolling(20).mean() / df["Close"] - 1
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volume_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)  # next-day direction
    return df.dropna().reset_index(drop=True)


def make_sequences(df: pd.DataFrame, seq_len: int):
    """
    Build (window, label) pairs. The label for a window ending at row r is
    targets[r] itself — target[r] = whether Close[r+1] > Close[r] — so the
    label is exactly the next-day direction following the last day the model
    actually observed. (Previously this used targets[i+seq_len], which skipped
    a day between the window and the label it was asked to predict, and also
    meant the most recent window never included the newest available row.)
    """
    feats = df[FEATURE_COLS].values.astype(np.float32)
    targets = df["target"].values.astype(np.int64)
    X, y = [], []
    for i in range(len(df) - seq_len + 1):
        window_end = i + seq_len  # exclusive
        X.append(feats[i:window_end])
        y.append(targets[window_end - 1])
    return np.array(X), np.array(y)

def train_baseline(df: pd.DataFrame, cfg: dict = None, epochs: int = 15) -> dict:
    cfg = cfg or load_config()
    seq_len = cfg["data"]["sequence_length"]

    engineered = engineer_features(df)
    X, y = make_sequences(engineered, seq_len)

    split = int(len(X) * 0.8)  # temporal split — never shuffle time series (FR-2 note)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = DirectionLSTM(n_features=len(FEATURE_COLS))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test)).argmax(dim=1).numpy()

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
    }
    return {"model": model, "metrics": metrics, "n_train": len(X_train), "n_test": len(X_test)}


def save_model(model, metrics: dict, ticker: str, cfg: dict = None, promoted: bool = True) -> int:
    cfg = cfg or load_config()
    model_dir = resolve_path(cfg["paths"]["model_dir"])
    existing = sorted([int(p.name[1:]) for p in model_dir.glob("v*") if p.name[1:].isdigit()])
    version = (existing[-1] + 1) if existing else 1
    vdir = model_dir / f"v{version}"
    vdir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), vdir / "model.pt")
    metadata = {
        "version": version,
        "ticker": ticker,
        "trained_at": datetime.now().isoformat(),
        "metrics": metrics,
        "promoted": promoted,
        "feature_cols": FEATURE_COLS,
    }
    with open(vdir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return version


if __name__ == "__main__":
    cfg = load_config()
    ticker = cfg["ticker"]
    snapshot = resolve_path(cfg["paths"]["snapshot_dir"]) / f"{ticker}_historical.csv"
    df = pd.read_csv(snapshot)
    result = train_baseline(df, cfg)
    version = save_model(result["model"], result["metrics"], ticker, cfg)
    print(f"Trained v{version} — metrics: {result['metrics']}")