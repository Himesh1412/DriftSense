# scripts/export_onnx.py
"""
Interoperability (SRS 5): export each stock's current model to ONNX so it can be loaded outside this
project, with no DriftSense source code and no PyTorch, e.g. with onnxruntime.

    python -m scripts.export_onnx                 # every stock that has a trained model
    python -m scripts.export_onnx META INFY.NS    # just these

Writes models/<ticker>/v<n>/model.onnx next to model.pt. The exported graph takes a batch of
(20 days x 7 features) raw windows and returns two scores [down, up]. Feature scaling is baked into the
graph. UP/DOWN is then decided from those scores the way src/forecasting/decision.py does it (UP when the
up-minus-down score is above the model's own recent typical score). The export is checked: onnxruntime
must reproduce PyTorch's scores before a file is kept.
"""
import json
import sys

import numpy as np
import torch

from src.config_loader import load_config, resolve_path
from src.retraining.registry import get_current_version, load_model

TOLERANCE = 1e-4


def export_model(ticker: str, cfg: dict) -> dict:
    import onnxruntime as ort

    version = get_current_version(ticker, cfg)
    if version == 0:
        raise RuntimeError("no promoted model to export")
    model = load_model(ticker, version, cfg).eval()
    seq_len = cfg["data"]["sequence_length"]
    n_features = model.feature_mean.numel()
    path = resolve_path(cfg["paths"]["model_dir"]) / ticker / f"v{version}" / "model.onnx"

    example = torch.randn(1, seq_len, n_features)
    torch.onnx.export(model, example, str(path), input_names=["window"], output_names=["scores"],
                      dynamic_axes={"window": {0: "batch"}, "scores": {0: "batch"}}, dynamo=False)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    probe = torch.randn(1, seq_len, n_features)
    with torch.no_grad():
        expected = model(probe).numpy()
    got = session.run(None, {"window": probe.numpy()})[0]
    worst = float(np.abs(expected - got).max())
    if worst > TOLERANCE:
        path.unlink()
        raise RuntimeError(f"ONNX output differs from PyTorch by {worst:.2e} (limit {TOLERANCE:.0e}); file removed")
    return {"ticker": ticker, "version": version, "file": str(path), "max_abs_difference": worst,
            "size_kb": round(path.stat().st_size / 1024, 1)}


if __name__ == "__main__":
    cfg = load_config()
    tickers = sys.argv[1:] or [t for g in cfg["tickers"].values() for t in g]
    done = []
    for t in tickers:
        try:
            r = export_model(t, cfg)
            done.append(r)
            print(f"{t}: v{r['version']} -> {r['file']}  ({r['size_kb']} KB, matches PyTorch to {r['max_abs_difference']:.1e})", flush=True)
        except Exception as e:
            print(f"{t}: FAILED - {e}", flush=True)
    print(f"\nExported {len(done)} of {len(tickers)} models.")
