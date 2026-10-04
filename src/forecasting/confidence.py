# src/forecasting/confidence.py
"""
Confidence-aware forecasting (FR-10, FR-11).
Runs multiple stochastic forward passes (MC Dropout) and uses the
variance/agreement across passes to produce a confidence score.
Abstains from a prediction when confidence falls below threshold — the
threshold itself comes from the model's own calibrated abstain_threshold
buffer (see model.py / train.py's calibrate_abstain_threshold), falling
back to cfg["confidence"]["abstain_below"] only for a model that predates
calibration or was constructed directly (e.g. in tests).
"""
import numpy as np
import torch
import torch.nn.functional as F

from src.config_loader import load_config
from src.forecasting.model import DirectionLSTM, enable_mc_dropout


def mc_dropout_pass(model: DirectionLSTM, x: np.ndarray, n_passes: int, reference: float = None) -> dict:
    """
    Direction + MC-Dropout confidence for one window (no abstain decision).

    Direction: UP when the model's deterministic score is above `reference` (its own recent
    typical score — see decision.py), falling back to the model's stored decision_offset.
    Confidence: the same offset is subtracted from every stochastic pass, and confidence is
    the mean probability of the chosen side minus how much the passes disagreed. If the
    stochastic passes mostly disagree with the deterministic call, confidence falls under
    0.5 and the model abstains.
    Shared by predict_with_confidence (one live call) and calibrate_abstain_threshold
    (many validation windows) so the two never compute confidence differently.
    """
    ref = float(model.decision_offset.item()) if reference is None else float(reference)
    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, n_features)

    model.eval()
    with torch.no_grad():
        det = model(x_t)
    gap = float(det[0, 1] - det[0, 0]) - ref
    predicted_class = 1 if gap > 0 else 0
    direction = "UP" if predicted_class == 1 else "DOWN"

    enable_mc_dropout(model)  # keep dropout stochastic for the confidence estimate
    probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(x_t).clone()
            logits[:, 1] -= ref
            probs.append(F.softmax(logits, dim=1).numpy()[0])
    probs = np.array(probs)  # (n_passes, 2)

    mean_confidence = float(probs.mean(axis=0)[predicted_class])
    disagreement = float(probs[:, predicted_class].std())
    confidence = max(0.0, mean_confidence - disagreement)

    return {
        "direction": direction,
        "confidence": round(confidence, 4),
        "raw_mean_prob": round(mean_confidence, 4),
        "disagreement": round(disagreement, 4),
    }


def predict_with_confidence(model: DirectionLSTM, x: np.ndarray, cfg: dict = None, reference: float = None) -> dict:
    """
    x: a single sequence, shape (sequence_length, n_features)
    reference: the model's recent typical score to call UP/DOWN against (decision.latest_reference);
               omit it to use the model's stored decision_offset.
    Returns predicted direction, confidence score, and whether to abstain.
    """
    cfg = cfg or load_config()
    n_passes = cfg["confidence"]["mc_dropout_passes"]
    abstain_below = float(model.abstain_threshold.item()) if hasattr(model, "abstain_threshold") \
        else cfg["confidence"]["abstain_below"]

    result = mc_dropout_pass(model, x, n_passes, reference)
    result["abstained"] = result["confidence"] < abstain_below
    return result
