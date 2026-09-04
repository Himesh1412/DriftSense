# src/retraining/registry.py
"""
Model Registry (FR-8, D2).
Versions trained models and keeps a queryable changelog.
"""
import json
import torch
from pathlib import Path

from src.config_loader import load_config, resolve_path
from src.forecasting.model import DirectionLSTM
from src.forecasting.train import FEATURE_COLS


def get_current_version(cfg: dict = None) -> int:
    cfg = cfg or load_config()
    model_dir = resolve_path(cfg["paths"]["model_dir"])
    promoted = []
    for vdir in model_dir.glob("v*"):
        meta_path = vdir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("promoted"):
                promoted.append(meta["version"])
    return max(promoted) if promoted else 0


def load_model(version: int, cfg: dict = None) -> DirectionLSTM:
    cfg = cfg or load_config()
    vdir = resolve_path(cfg["paths"]["model_dir"]) / f"v{version}"
    model = DirectionLSTM(n_features=len(FEATURE_COLS))
    model.load_state_dict(torch.load(vdir / "model.pt"))
    return model


def load_current_model(cfg: dict = None):
    cfg = cfg or load_config()
    version = get_current_version(cfg)
    if version == 0:
        return None, None
    return load_model(version, cfg), version


def get_model_history(cfg: dict = None) -> list:
    """Used by the dashboard (FR-12) to show retraining/version history."""
    cfg = cfg or load_config()
    model_dir = resolve_path(cfg["paths"]["model_dir"])
    history = []
    for vdir in sorted(model_dir.glob("v*")):
        meta_path = vdir / "metadata.json"
        if meta_path.exists():
            history.append(json.loads(meta_path.read_text()))
    return sorted(history, key=lambda m: m["version"], reverse=True)