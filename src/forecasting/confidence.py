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


def mc_dropout_pass(model: DirectionLSTM, x: np.ndarray, n_passes: int) -> dict:
    """
    Runs the stochastic forward passes and returns the raw MC-Dropout
    signal (direction, confidence, disagreement) with no abstain decision —
    shared by predict_with_confidence (single live prediction) and
    calibrate_abstain_threshold (calibrating against many validation
    examples at once) so the two never compute confidence differently.
    """
    model.eval()
    enable_mc_dropout(model)  # keep dropout stochastic even in eval mode

    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, seq_len, n_features)

    probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(x_t)
            probs.append(F.softmax(logits, dim=1).numpy()[0])
    probs = np.array(probs)  # (n_passes, 2)

    mean_probs = probs.mean(axis=0)
    predicted_class = int(mean_probs.argmax())
    direction = "UP" if predicted_class == 1 else "DOWN"

    # Confidence = mean probability of the predicted class, discounted by
    # how much the passes disagreed with each other (epistemic uncertainty).
    mean_confidence = float(mean_probs[predicted_class])
    disagreement = float(probs[:, predicted_class].std())
    confidence = max(0.0, mean_confidence - disagreement)

    return {
        "direction": direction,
        "confidence": round(confidence, 4),
        "raw_mean_prob": round(mean_confidence, 4),
        "disagreement": round(disagreement, 4),
    }


def predict_with_confidence(model: DirectionLSTM, x: np.ndarray, cfg: dict = None) -> dict:
    """
    x: a single sequence, shape (sequence_length, n_features)
    Returns predicted direction, confidence score, and whether to abstain.
    """
    cfg = cfg or load_config()
    n_passes = cfg["confidence"]["mc_dropout_passes"]
    abstain_below = float(model.abstain_threshold.item()) if hasattr(model, "abstain_threshold") \
        else cfg["confidence"]["abstain_below"]

    result = mc_dropout_pass(model, x, n_passes)
    result["abstained"] = result["confidence"] < abstain_below
    return result
