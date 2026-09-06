# src/forecasting/model.py
"""
Forecasting model architecture (FR-2).
Binary direction classifier (up/down) so we get clean precision/recall/F1
metrics rather than a noisy price-regression R^2 — see project notes.
Dropout is kept active at inference time to support MC Dropout (FR-10).
Built as explicit per-layer LSTM + Dropout modules (rather than nn.LSTM's
built-in inter-layer dropout) specifically so enable_mc_dropout can actually
reactivate every dropout layer — nn.LSTM's internal recurrent dropout isn't
a discoverable nn.Dropout submodule, so it used to stay off in eval mode
regardless of enable_mc_dropout, leaving only the head dropout stochastic.
"""
import torch
import torch.nn as nn


class DirectionLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        for i in range(num_layers):
            in_size = n_features if i == 0 else hidden_size
            self.lstm_layers.append(nn.LSTM(in_size, hidden_size, num_layers=1, batch_first=True))
            self.dropout_layers.append(nn.Dropout(dropout))
        self.head = nn.Linear(hidden_size, 2)  # logits: [down, up]

    def forward(self, x):
        # x: (batch, sequence_length, n_features)
        out = x
        for lstm, drop in zip(self.lstm_layers, self.dropout_layers):
            out, _ = lstm(out)
            out = drop(out)  # a real nn.Dropout after every layer, reactivatable below
        last = out[:, -1, :]  # final timestep's hidden state
        return self.head(last)


def enable_mc_dropout(model: nn.Module):
    """Keep dropout layers active during inference for MC Dropout confidence estimation."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
