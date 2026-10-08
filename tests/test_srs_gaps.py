# tests/test_srs_gaps.py
"""Tests for the SRS items added late: model metadata (3.4), Git-tag versioning (3.3),
retraining justification (NFR-3), request throttling and retries (NFR-2, Reliability)."""
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.forecasting.train import describe_training_data, save_model, train_baseline
from src.ingestion import yahoo_ingestor
from src.retraining import versioning
from src.retraining.manager import run_retraining_cycle


def _df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"Date": pd.date_range("2022-01-03", periods=n, freq="B").astype(str),
                         "Close": 100 * np.cumprod(1 + rng.normal(0, 0.01, n)),
                         "Volume": rng.integers(1000, 5000, n)})


def _cfg(tmp_path, **extra):
    return {"data": {"sequence_length": 20},
            "paths": {"model_dir": str(tmp_path / "models"), "retrain_log": str(tmp_path / "retrain.jsonl")},
            "training": {"epochs": 2, "hidden_size": 8, "num_layers": 1},
            "retraining": {"min_improvement": 0.0, "benchmark_window_days": 60}, **extra}


# ── SRS 3.4: artifacts carry training-data range, metrics, promotion decision ──
def test_describe_training_data_reports_the_date_range_and_size():
    info = describe_training_data(_df(100))
    assert info["n_rows"] == 100 and info["start"] == "2022-01-03" and info["end"] > info["start"]


def test_saved_model_metadata_has_data_range_metrics_promotion_and_hash(tmp_path):
    cfg = _cfg(tmp_path)
    result = train_baseline(_df(), cfg)
    save_model(result["model"], result["metrics"], "T", cfg, promoted=False, training_data=result["training_data"])
    meta = json.loads((tmp_path / "models" / "T" / "v1" / "metadata.json").read_text())
    assert meta["training_data"]["start"] == "2022-01-03" and meta["training_data"]["n_rows"] == 400
    assert "accuracy" in meta["metrics"] and meta["promoted"] is False
    assert len(meta["model_sha256"]) == 64


# ── SRS 3.3: Git-tag versioning ──
def test_tagging_is_off_unless_the_config_turns_it_on():
    with patch("src.retraining.versioning.subprocess.run") as run:
        assert versioning.tag_model_version("T", 1, {}, {}) is False
        run.assert_not_called()


def test_a_promoted_model_gets_an_annotated_tag_carrying_its_hash_and_metrics():
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return type("R", (), {"returncode": 1 if "rev-parse" in cmd else 0})()  # tag does not exist yet

    with patch("src.retraining.versioning.subprocess.run", side_effect=fake_run):
        made = versioning.tag_model_version("AAPL", 3, {"model_sha256": "abc", "metrics": {"accuracy": 0.5}},
                                            {"registry": {"git_tag_versions": True}})
    tag_cmd = next(c for c in calls if "tag" in c and "-a" in c)
    assert made and "model/AAPL/v3" in tag_cmd and "abc" in tag_cmd[-1]


def test_an_existing_tag_is_not_recreated():
    ok = type("R", (), {"returncode": 0})()
    with patch("src.retraining.versioning.subprocess.run", return_value=ok) as run:
        assert versioning.tag_model_version("T", 1, {}, {"registry": {"git_tag_versions": True}}) is False
        assert run.call_count == 1  # only the existence check


def test_git_being_unavailable_never_breaks_training():
    with patch("src.retraining.versioning.subprocess.run", side_effect=OSError("no git")):
        assert versioning.tag_model_version("T", 1, {}, {"registry": {"git_tag_versions": True}}) is False


def test_rejected_models_are_not_tagged(tmp_path):
    cfg = _cfg(tmp_path, registry={"git_tag_versions": True})
    result = train_baseline(_df(), cfg)
    with patch("src.forecasting.train.tag_model_version") as tag:
        save_model(result["model"], result["metrics"], "T", cfg, promoted=False)
        tag.assert_not_called()


# ── NFR-3: every retraining decision carries its justification ──
def test_the_retrain_log_says_why_a_candidate_was_promoted_or_rejected(tmp_path):
    cfg = _cfg(tmp_path)
    first = train_baseline(_df(), cfg)
    save_model(first["model"], first["metrics"], "T", cfg, training_data=first["training_data"])
    df = _df(seed=1)
    entry = run_retraining_cycle(df.iloc[:300], df.iloc[300:], df.tail(60), "T", "Regime Shift", cfg)
    assert entry["justification"].startswith(("Promoted:", "Rejected:"))
    assert entry["training_data"]["n_rows"] == 400
    logged = json.loads((tmp_path / "retrain.jsonl").read_text().splitlines()[-1])
    assert logged["justification"] == entry["justification"]


# ── NFR-2 / Reliability: spaced-out, retried requests ──
def test_requests_are_spaced_by_the_configured_minimum(monkeypatch):
    sleeps, clock = [], [100.0]
    monkeypatch.setattr(yahoo_ingestor.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(yahoo_ingestor.time, "sleep", lambda s: (sleeps.append(s), clock.__setitem__(0, clock[0] + s)))
    monkeypatch.setattr(yahoo_ingestor, "_last_request", [clock[0]])
    with patch("src.ingestion.yahoo_ingestor.yf.download", return_value=pd.DataFrame()):
        yahoo_ingestor._download({"ingestion": {"min_seconds_between_requests": 2.0}}, tickers="X")
    assert sleeps and sleeps[0] == pytest.approx(2.0)


def test_a_failed_request_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(yahoo_ingestor.time, "sleep", lambda s: None)
    attempts = {"n": 0}

    def flaky(**kw):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("Yahoo hiccup")
        return pd.DataFrame({"Close": [1.0]})

    with patch("src.ingestion.yahoo_ingestor.yf.download", side_effect=flaky):
        out = yahoo_ingestor._download({"ingestion": {"retries": 3}}, tickers="X")
    assert attempts["n"] == 3 and len(out) == 1


def test_a_request_that_keeps_failing_eventually_raises(monkeypatch):
    monkeypatch.setattr(yahoo_ingestor.time, "sleep", lambda s: None)
    with patch("src.ingestion.yahoo_ingestor.yf.download", side_effect=ConnectionError("down")) as dl:
        with pytest.raises(ConnectionError):
            yahoo_ingestor._download({"ingestion": {"retries": 2}}, tickers="X")
        assert dl.call_count == 2


# ── SRS 5 Interoperability: ONNX export must reproduce PyTorch ──
def test_onnx_export_matches_pytorch_and_accepts_any_batch_size(tmp_path):
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort
    import torch
    from scripts.export_onnx import export_model
    from src.retraining.registry import load_model

    cfg = _cfg(tmp_path)
    result = train_baseline(_df(), cfg)
    save_model(result["model"], result["metrics"], "T", cfg, training_data=result["training_data"])
    info = export_model("T", cfg)
    assert info["max_abs_difference"] < 1e-4

    session = ort.InferenceSession(info["file"], providers=["CPUExecutionProvider"])
    batch = np.random.default_rng(0).normal(size=(5, 20, 7)).astype(np.float32)
    got = session.run(None, {"window": batch})[0]
    with torch.no_grad():
        expected = load_model("T", 1, cfg).eval()(torch.tensor(batch)).numpy()
    assert got.shape == (5, 2) and np.abs(got - expected).max() < 1e-4
