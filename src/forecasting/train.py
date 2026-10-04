# src/forecasting/train.py
"""
Baseline model training (FR-2).
Feature engineering: returns, moving averages, volatility, momentum, and
range-position (not raw prices). Trains a direction classifier and reports
precision/recall/F1.
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score

from src.config_loader import load_config, resolve_path
from src.forecasting.model import DirectionLSTM
from src.forecasting.confidence import mc_dropout_pass

FEATURE_COLS = ["return_1d", "ma_5", "ma_20", "volatility_10d", "volume_z", "rsi_14", "hl_position_20"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw OHLCV into model-ready features (never feed raw price directly)."""
    df = df.copy()
    df["return_1d"] = df["Close"].pct_change()
    df["ma_5"] = df["Close"].rolling(5).mean() / df["Close"] - 1
    df["ma_20"] = df["Close"].rolling(20).mean() / df["Close"] - 1
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volume_z"] = (df["Volume"] - df["Volume"].rolling(20).mean()) / df["Volume"].rolling(20).std()

    # RSI(14) — standard momentum oscillator, bounded [0, 100]; 100 when
    # there were zero losses in the window (the textbook limiting case).
    delta = df["Close"].diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi_14"] = (100 - (100 / (1 + rs))).replace([np.inf, -np.inf], 100)

    # Where today's close sits within the last 20 days' [min, max] range —
    # a stochastic-%K-style positioning feature, bounded [0, 1].
    roll_min_20 = df["Close"].rolling(20).min()
    roll_max_20 = df["Close"].rolling(20).max()
    df["hl_position_20"] = (df["Close"] - roll_min_20) / (roll_max_20 - roll_min_20).replace(0, np.nan)

    # next-day direction. The most recent day has no "next day" yet, so its
    # label is genuinely unknown (NaN) — NOT "down". (A plain `shift(-1) > Close`
    # comparison turns that missing value into False, i.e. a fake DOWN label.)
    next_close = df["Close"].shift(-1)
    df["target"] = (next_close > df["Close"]).astype(float).where(next_close.notna())

    # Drop rows whose *features* are incomplete (indicator warm-up), but keep the
    # latest day even though its label is unknown: live prediction needs it.
    return df.dropna(subset=FEATURE_COLS).reset_index(drop=True)


def make_sequences(df: pd.DataFrame, seq_len: int, labeled_only: bool = True):
    """
    Build (window, label) pairs. The label for a window ending at row r is
    targets[r] itself — target[r] = whether Close[r+1] > Close[r] — so the
    label is exactly the next-day direction following the last day the model
    actually observed.

    labeled_only=True (default — training, validation, backtesting, scoring):
    windows whose label is still unknown (the latest day) are left out, so a
    model is never trained or scored on a label that doesn't exist yet.
    labeled_only=False (live prediction / history backfill): every window is
    kept, including the newest one, and unknown labels come back as -1.
    """
    feats = df[FEATURE_COLS].values.astype(np.float32)
    targets = df["target"].values.astype(np.float64)
    X, y = [], []
    for i in range(len(df) - seq_len + 1):
        window_end = i + seq_len  # exclusive
        label = targets[window_end - 1]
        if np.isnan(label):
            if labeled_only:
                continue
            label = -1
        X.append(feats[i:window_end])
        y.append(int(label))
    return np.array(X), np.array(y, dtype=np.int64)


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """
    Inverse-frequency class weights for CrossEntropyLoss. Without this, if
    "UP" days outnumber "DOWN" days (plausible over a multi-year uptrend),
    the model can minimize loss by always predicting the majority class —
    which is exactly what was happening (recall=1.0, precision=accuracy,
    in both the single-split run and every walk-forward backtest fold).
    """
    counts = np.bincount(y, minlength=2).astype(np.float32)
    counts = np.clip(counts, 1, None)  # guard divide-by-zero if a class is absent
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def fit_feature_scaler(model: DirectionLSTM, X_train: np.ndarray):
    """
    Sets the model's internal feature_mean/feature_std buffers from the
    TRAINING split only (never test/validation data), so every input the
    model ever sees afterwards — held-out testing, backtesting, or live
    inference — gets standardized the exact same way it was trained on.
    """
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std < 1e-6] = 1.0  # guard against a near-constant feature
    with torch.no_grad():
        model.feature_mean.copy_(torch.tensor(mean, dtype=torch.float32))
        model.feature_std.copy_(torch.tensor(std, dtype=torch.float32))


def calibrate_abstain_threshold(model: DirectionLSTM, X_val: np.ndarray, cfg: dict) -> float:
    """
    Sets the model's own abstain_threshold buffer (FR-11) from its own
    validation-set MC-Dropout confidence distribution, instead of comparing
    against one fixed global magnitude (0.55) that assumes a certain output
    scale. That assumption turned out to be wrong in practice: confidence
    clustered in a narrow band (e.g. 0.49-0.52) regardless of a model's real
    backtested accuracy, so a fixed 0.55 made every model abstain 100% of
    the time, including ones with genuinely good backtest accuracy (e.g.
    META at 60%+ across 6 folds).

    Calibration abstains only on the least-confident tail of THIS model's
    own distribution (bottom `abstain_percentile`), guaranteeing meaningful,
    non-zero coverage on every ticker regardless of its absolute confidence
    scale. This is deliberately framed as "flag the relatively least-sure
    predictions", not "the kept tier is proven more accurate" — at this
    data scale, confidence magnitude and correctness turned out to be only
    weakly/noisily correlated, so claiming more than that would overstate
    what the evidence supports.

    Uses X_val (the internal slice already carved out of the TRAIN split
    for checkpoint selection), never X_test, so the held-out test metrics
    stay completely uncontaminated by this calibration step.
    """
    percentile = cfg.get("confidence", {}).get("abstain_percentile", 25)
    n_passes = cfg.get("confidence", {}).get("mc_dropout_passes", 20)
    fallback = cfg.get("confidence", {}).get("abstain_below", 0.55)

    if len(X_val) == 0:
        return fallback

    confidences = [mc_dropout_pass(model, x, n_passes)["confidence"] for x in X_val]
    threshold = float(np.percentile(confidences, percentile))
    with torch.no_grad():
        model.abstain_threshold.fill_(threshold)
    return threshold


def temporal_split(X: np.ndarray, y: np.ndarray, seq_len: int, val_fraction: float = 0.15):
    """
    Chronological train / validation / test split (never shuffled — FR-2 note)
    with a gap of seq_len samples between each pair of slices.

    Each sample is a seq_len-day window whose label looks one day further
    ahead, so a window starting at row i touches rows i .. i+seq_len. Without
    a gap, the last ~seq_len windows of an earlier slice share days with the
    first windows of the next one — mild train/test leakage. Dropping seq_len
    samples between slices guarantees no row is used by two slices.

    Returns X_train, y_train, X_val, y_val, X_test, y_test.
    """
    gap = seq_len
    split = int(len(X) * 0.8)
    X_train_full, y_train_full = X[:split], y[:split]
    X_test, y_test = X[split + gap:], y[split + gap:]

    val_split = max(1, int(len(X_train_full) * (1 - val_fraction)))
    X_train, y_train = X_train_full[:val_split], y_train_full[:val_split]
    X_val, y_val = X_train_full[val_split + gap:], y_train_full[val_split + gap:]
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_baseline(df: pd.DataFrame, cfg: dict = None, epochs: int = None) -> dict:
    cfg = cfg or load_config()
    seq_len = cfg["data"]["sequence_length"]
    tcfg = cfg.get("training", {})
    epochs = epochs if epochs is not None else tcfg.get("epochs", 60)
    hidden_size = tcfg.get("hidden_size", 64)
    num_layers = tcfg.get("num_layers", 2)
    dropout = tcfg.get("dropout", 0.3)
    lr = tcfg.get("learning_rate", 1e-3)
    weight_decay = tcfg.get("weight_decay", 1e-4)
    val_fraction = tcfg.get("val_fraction", 0.15)

    engineered = engineer_features(df)
    X, y = make_sequences(engineered, seq_len)

    # Temporal split with a seq_len gap between train/val/test (see
    # temporal_split). The validation slice is carved out of the TRAIN side
    # only, to pick the best-performing epoch's weights and calibrate the
    # abstain threshold — "epochs" is an upper bound, and the checkpoint
    # that's kept is chosen by real validation performance.
    X_train, y_train, X_val, y_val, X_test, y_test = temporal_split(X, y, seq_len, val_fraction)

    model = DirectionLSTM(n_features=len(FEATURE_COLS), hidden_size=hidden_size,
                           num_layers=num_layers, dropout=dropout)
    fit_feature_scaler(model, X_train)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=compute_class_weights(y_train))

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val)

    best_val_bal_acc = -1.0
    best_state = None
    loss_history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        loss_history.append(round(float(loss.item()), 4))

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t).argmax(dim=1).numpy() if len(y_val) > 0 else np.array([])
        # Balanced accuracy (average per-class recall), not raw accuracy —
        # raw accuracy on a small validation slice is exactly as gameable by
        # a degenerate "always predict the majority class" model as the
        # original training-collapse bug was. Balanced accuracy caps a
        # single-class predictor at 0.5, so checkpoint selection can no
        # longer reward collapsing back to it.
        val_bal_acc = float(balanced_accuracy_score(y_val, val_preds)) if len(y_val) > 0 else 0.0
        if val_bal_acc >= best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    calibrated_threshold = calibrate_abstain_threshold(model, X_val, cfg)

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test)).argmax(dim=1).numpy()

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
    }
    return {"model": model, "metrics": metrics, "n_train": len(X_train), "n_test": len(X_test),
            "loss_history": loss_history, "best_val_balanced_accuracy": round(best_val_bal_acc, 4),
            "calibrated_abstain_threshold": round(calibrated_threshold, 4)}


def save_model(model, metrics: dict, ticker: str, cfg: dict = None, promoted: bool = True) -> int:
    """Models are namespaced per ticker (models/{ticker}/v{n}/) so training
    one stock's model can never collide with or silently replace another's
    — see src/retraining/registry.py for the read side of this layout."""
    cfg = cfg or load_config()
    model_dir = resolve_path(cfg["paths"]["model_dir"]) / ticker
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
        # the shape this checkpoint was built with, so the registry can rebuild exactly
        # it later (training hyperparameters are config-driven and may change over time)
        "architecture": {
            "hidden_size": model.lstm_layers[0].hidden_size,
            "num_layers": len(model.lstm_layers),
            "dropout": model.dropout_layers[0].p,
        },
        # already baked into the checkpoint via model.abstain_threshold —
        # duplicated here in plain JSON for easy display (dashboard/report)
        # without having to load the model to read it.
        "calibrated_abstain_threshold": round(float(model.abstain_threshold.item()), 4),
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
