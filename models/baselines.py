import math

import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    def __init__(self, seq_len, input_features, hidden_dim=128, dropout=0.2):
        super().__init__()
        flat_dim = seq_len * input_features
        self.mlp = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        batch_size, seq_len, num_nodes, num_features = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len * num_features)
        y = self.mlp(x)
        return y.reshape(batch_size, num_nodes)


class LSTMRegressor(nn.Module):
    def __init__(self, input_features, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        batch_size, seq_len, num_nodes, num_features = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, num_features)
        _, (hidden, _) = self.encoder(x)
        last_hidden = hidden[-1]
        y = self.head(last_hidden)
        return y.reshape(batch_size, num_nodes)


class ALSTMRegressor(nn.Module):
    def __init__(self, input_features, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.attn_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_score = nn.Linear(hidden_dim, 1, bias=False)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        batch_size, seq_len, num_nodes, num_features = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, num_features)
        hidden_states, _ = self.encoder(x)
        attn_logits = self.attn_score(torch.tanh(self.attn_proj(hidden_states)))
        attn_weights = torch.softmax(attn_logits, dim=1)
        context = torch.sum(attn_weights * hidden_states, dim=1)
        y = self.head(context)
        return y.reshape(batch_size, num_nodes)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerRegressor(nn.Module):
    def __init__(self, seq_len, input_features, hidden_dim=128, num_layers=3, num_heads=8, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_features, hidden_dim)
        self.positional = PositionalEncoding(hidden_dim, max_len=seq_len + 4)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        batch_size, seq_len, num_nodes, num_features = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size * num_nodes, seq_len, num_features)
        x = self.input_proj(x)
        x = self.positional(x)
        x = self.encoder(x)
        pooled = x.mean(dim=1)
        y = self.head(pooled)
        return y.reshape(batch_size, num_nodes)

