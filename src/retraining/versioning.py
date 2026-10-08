# src/retraining/versioning.py
"""
Git-tag versioning for promoted models (SRS 3.3: "versioning via Git tags or DVC").

Model files themselves are big binaries and stay out of Git (see .gitignore), so a tag cannot
contain them. Instead each promoted model gets an annotated tag `model/<ticker>/v<n>` on the
commit it was trained from; the tag message carries the model's SHA-256, its metrics and its
training-data range, so any model file can be matched to the exact code that produced it.

Best effort and opt-in (registry.git_tag_versions in config): outside a Git checkout, or if Git
refuses, training simply carries on.
"""
import hashlib
import json
import subprocess
from pathlib import Path

from src.config_loader import ROOT


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tag_model_version(ticker: str, version: int, metadata: dict, cfg: dict) -> bool:
    """Create the annotated tag. Returns True if a new tag was made."""
    if not cfg.get("registry", {}).get("git_tag_versions", False):
        return False
    name = f"model/{ticker}/v{version}"
    message = f"{ticker} model v{version}\n" + json.dumps(
        {k: metadata.get(k) for k in ("model_sha256", "metrics", "training_data", "trained_at", "promoted")}, indent=2)
    try:
        exists = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "-q", "--verify", f"refs/tags/{name}"],
                                capture_output=True, timeout=15).returncode == 0
        if exists:
            return False
        done = subprocess.run(["git", "-C", str(ROOT), "tag", "-a", name, "-m", message],
                              capture_output=True, timeout=15)
        return done.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
