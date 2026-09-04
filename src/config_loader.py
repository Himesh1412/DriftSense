# src/config_loader.py
"""Shared config loader used by every module."""
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str = None) -> dict:
    path = path or ROOT / "config" / "settings.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(relative: str) -> Path:
    """Resolve a path from settings.yaml relative to the project root."""
    return ROOT / relative