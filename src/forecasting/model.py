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

Feature standardization (feature_mean/feature_std) is baked into the model
itself as non-trainable buffers, rather than a separate scaler artifact
saved alongside it. train.py's fit_feature_scaler() sets these from the
training split only; every other caller (confidence.py, run_daily_cycle.py,
backtest.py, registry.py) keeps calling model(x) on raw engineered features
exactly as before — saving/loading model.pt saves/loads these buffers
automatically via state_dict(), so no plumbing changes were needed anywhere
else in the pipeline. Left un-fit (mean=0, std=1), the model behaves exactly
as before (a no-op), which is what a freshly constructed, untrained model in
the test suite gets.

abstain_threshold is the same pattern applied to FR-11's confidence gate.
MC-Dropout confidence turned out to cluster in a narrow, model-specific band
(e.g. 0.49-0.52) regardless of a model's real backtested accuracy — a fixed
global threshold (0.55) was unreachable for every ticker tried, including
ones with genuinely good backtest accuracy, so it abstained 100% of the
time. train.py's calibrate_abstain_threshold() sets this per-model from
that model's own validation-set confidence distribution instead.
"""
import torch
import torch.nn as nn


class DirectionLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.register_buffer("feature_mean", torch.zeros(n_features))
        self.register_buffer("feature_std", torch.ones(n_features))
        self.register_buffer("abstain_threshold", torch.tensor(0.55))
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        for i in range(num_layers):
            in_size = n_features if i == 0 else hidden_size
            self.lstm_layers.append(nn.LSTM(in_size, hidden_size, num_layers=1, batch_first=True))
            self.dropout_layers.append(nn.Dropout(dropout))
        self.head = nn.Linear(hidden_size, 2)  # logits: [down, up]

    def forward(self, x):
        # x: (batch, sequence_length, n_features)
        out = (x - self.feature_mean) / self.feature_std
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
