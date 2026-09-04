# src/drift/attribution.py
"""
Drift Attribution Module (FR-6).
Uses SHAP to explain which input feature most influenced the model's
recent predictions during a flagged drift event — this is what lets the
dashboard say *why*, not just *that*, drift occurred.
"""
import numpy as np
import shap
import torch

from src.forecasting.train import FEATURE_COLS


def explain_drift(model, recent_sequences: np.ndarray) -> dict:
    """
    recent_sequences: array of shape (n_samples, sequence_length, n_features)
    covering the window where drift was detected.
    """
    model.eval()

    def predict_fn(flat_x):
        # SHAP's KernelExplainer works on flat feature vectors; reshape back
        # to (batch, seq_len, n_features) for the LSTM.
        seq_len = recent_sequences.shape[1]
        n_feat = recent_sequences.shape[2]
        x = flat_x.reshape(-1, seq_len, n_feat)
        with torch.no_grad():
            logits = model(torch.tensor(x, dtype=torch.float32))
        return torch.softmax(logits, dim=1).numpy()[:, 1]  # P(up)

    flat_samples = recent_sequences.reshape(recent_sequences.shape[0], -1)
    background = flat_samples[: min(20, len(flat_samples))]

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(flat_samples[-1:], nsamples=100)

    seq_len = recent_sequences.shape[1]
    n_feat = recent_sequences.shape[2]
    shap_grid = np.array(shap_values).reshape(seq_len, n_feat)

    # Aggregate SHAP magnitude per feature across the lookback window
    feature_importance = np.abs(shap_grid).mean(axis=0)
    top_idx = int(np.argmax(feature_importance))
    top_feature = FEATURE_COLS[top_idx]
    top_value = float(feature_importance[top_idx])

    explanation = f"Drift primarily attributed to '{top_feature}' (mean |SHAP| = {top_value:.4f})."

    return {
        "top_feature": top_feature,
        "shap_value": round(top_value, 4),
        "explanation_text": explanation,
        "all_features": {f: round(float(v), 4) for f, v in zip(FEATURE_COLS, feature_importance)},
    }