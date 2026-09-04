# src/forecasting/model.py
"""
Forecasting model architecture (FR-2).
Binary direction classifier (up/down) so we get clean precision/recall/F1
metrics rather than a noisy price-regression R^2 — see project notes.
Dropout is kept active at inference time to support MC Dropout (FR-10).
"""
import torch
import torch.nn as nn


class DirectionLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 2)  # logits: [down, up]

    def forward(self, x):
        # x: (batch, sequence_length, n_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]          # final timestep's hidden state
        last = self.dropout(last)     # dropout stays active even in eval() via enable_mc_dropout
        return self.head(last)


def enable_mc_dropout(model: nn.Module):
    """Keep dropout layers active during inference for MC Dropout confidence estimation."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()