# src/retraining/registry.py
"""
Model Registry (FR-8, D2).
Versions trained models and keeps a queryable changelog, namespaced per
ticker (models/{ticker}/v{n}/) so training a second stock's model can never
silently become the "current" model for a different one — before this fix,
every ticker shared one flat models/v{n}/ sequence with no isolation.
"""
import json
import torch
from pathlib import Path

from src.config_loader import load_config, resolve_path
from src.forecasting.model import DirectionLSTM
from src.forecasting.train import FEATURE_COLS


def _ticker_model_dir(ticker: str, cfg: dict) -> Path:
    return resolve_path(cfg["paths"]["model_dir"]) / ticker


def get_current_version(ticker: str, cfg: dict = None) -> int:
    cfg = cfg or load_config()
    model_dir = _ticker_model_dir(ticker, cfg)
    promoted = []
    for vdir in model_dir.glob("v*"):
        meta_path = vdir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("promoted"):
                promoted.append(meta["version"])
    return max(promoted) if promoted else 0


def load_model(ticker: str, version: int, cfg: dict = None) -> DirectionLSTM:
    cfg = cfg or load_config()
    vdir = _ticker_model_dir(ticker, cfg) / f"v{version}"
    # feature count comes from this specific checkpoint's own metadata, not
    # today's FEATURE_COLS — so an older model trained before a feature was
    # added still loads correctly instead of a size-mismatch crash.
    meta_path = vdir / "metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    feature_cols = meta.get("feature_cols", FEATURE_COLS)
    # older checkpoints predate "architecture" and were all built with the defaults
    model = DirectionLSTM(n_features=len(feature_cols), **meta.get("architecture", {}))
    model.load_state_dict(torch.load(vdir / "model.pt"))
    return model


def load_current_model(ticker: str, cfg: dict = None):
    cfg = cfg or load_config()
    version = get_current_version(ticker, cfg)
    if version == 0:
        return None, None
    return load_model(ticker, version, cfg), version


def get_model_history(ticker: str, cfg: dict = None) -> list:
    """Used by the dashboard (FR-12) to show retraining/version history."""
    cfg = cfg or load_config()
    model_dir = _ticker_model_dir(ticker, cfg)
    history = []
    for vdir in sorted(model_dir.glob("v*")):
        meta_path = vdir / "metadata.json"
        if meta_path.exists():
            history.append(json.loads(meta_path.read_text()))
    return sorted(history, key=lambda m: m["version"], reverse=True)
