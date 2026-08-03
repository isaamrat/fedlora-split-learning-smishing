"""
models/cnn_lstm.py — Hybrid CNN-LSTM classifier for SMS classification.

Architecture:
  Embedding → Conv1d (local n-gram features) → MaxPool → LSTM (sequential context)
            → Final hidden state → Dropout → Linear

The CNN acts as a local feature extractor (n-gram patterns), feeding a compressed
sequence into the LSTM, which captures order-dependent patterns.

Split learning support — two natural split points:

  Split A (after embedding):
    Client:  Embedding → [batch, seq_len, embed_dim]
    Server:  CNN + LSTM + Classifier

  Split B (after CNN) ⭐ recommended:
    Client:  Embedding + Conv1d + Tanh + MaxPool → compressed sequence
    Server:  LSTM + Classifier
    Reason:  CNN and LSTM play distinct roles; separating them is most meaningful.
             Also reduces communication: CNN output is [batch, cnn_seq_len, num_filters],
             which is typically smaller than the raw embedding sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Full model ─────────────────────────────────────────────────────────────────

class CNNLSTMClassifier(nn.Module):
    """
    Hybrid CNN-LSTM classifier.

    Args:
        vocab_size:    vocabulary size
        embed_dim:     embedding dimension
        num_filters:   number of CNN filters (output channels)
        kernel_size:   CNN kernel size (n-gram window)
        hidden_dim:    LSTM hidden state dimension
        num_layers:    number of stacked LSTM layers
        num_classes:   number of output classes
        dropout:       dropout before classifier
        embed_weights: optional pretrained embedding tensor
        freeze_embed:  freeze embedding weights if True
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        num_filters: int = 128,
        kernel_size: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_classes: int = 3,
        dropout: float = 0.3,
        embed_weights=None,
        freeze_embed: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embed_weights is not None:
            self.embedding.weight = nn.Parameter(embed_weights.clone().float())
            if freeze_embed:
                self.embedding.weight.requires_grad = False

        # CNN: extract local features; output [batch, num_filters, reduced_seq]
        self.conv     = nn.Conv1d(embed_dim, num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pool     = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM: model sequential patterns in compressed feature sequence
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size  = num_filters,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = lstm_dropout,
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)              # [batch, seq_len, embed_dim]
        x = x.permute(0, 2, 1)                    # [batch, embed_dim, seq_len]
        x = torch.tanh(self.conv(x))              # [batch, num_filters, seq_len]
        x = self.pool(x)                           # [batch, num_filters, seq_len//2]
        x = x.permute(0, 2, 1)                    # [batch, seq_len//2, num_filters]
        _, (h_n, _) = self.lstm(x)                # h_n: [num_layers, batch, hidden_dim]
        x = self.dropout(h_n[-1])                 # [batch, hidden_dim]
        return self.classifier(x)                 # [batch, num_classes]


# ── Split A: after embedding ───────────────────────────────────────────────────

class CNNLSTMClientA(nn.Module):
    """
    Client-side (split A): embedding only.
    Outputs raw embeddings [batch, seq_len, embed_dim].
    Minimal client computation; server does all heavy work.
    """

    def __init__(self, vocab_size, embed_dim=100, embed_weights=None, freeze_embed=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embed_weights is not None:
            self.embedding.weight = nn.Parameter(embed_weights.clone().float())
            if freeze_embed:
                self.embedding.weight.requires_grad = False

    def forward(self, input_ids):
        return self.embedding(input_ids)           # [batch, seq_len, embed_dim]


class CNNLSTMServerA(nn.Module):
    """
    Server-side (split A): CNN + LSTM + Classifier.
    Receives raw embeddings [batch, seq_len, embed_dim] from client.
    """

    def __init__(self, embed_dim=100, num_filters=128, kernel_size=3,
                 hidden_dim=128, num_layers=1, num_classes=3, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size = num_filters, hidden_size = hidden_dim,
            num_layers = num_layers, batch_first = True, dropout = lstm_dropout,
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, hidden):
        # hidden: [batch, seq_len, embed_dim]
        x = hidden.permute(0, 2, 1)              # [batch, embed_dim, seq_len]
        x = torch.tanh(self.conv(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)                  # [batch, seq_len//2, num_filters]
        _, (h_n, _) = self.lstm(x)
        return self.classifier(self.dropout(h_n[-1]))


# ── Split B: after CNN ─────────────────────────────────────────────────────────

class CNNLSTMClientB(nn.Module):
    """
    Client-side (split B): Embedding + Conv + Tanh + MaxPool.
    Outputs compressed feature sequence [batch, seq_len//2, num_filters].
    Semantically meaningful: client extracts local n-gram features.
    Communication-efficient: smaller than raw embeddings after pooling.
    """

    def __init__(self, vocab_size, embed_dim=100, num_filters=128, kernel_size=3,
                 embed_weights=None, freeze_embed=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embed_weights is not None:
            self.embedding.weight = nn.Parameter(embed_weights.clone().float())
            if freeze_embed:
                self.embedding.weight.requires_grad = False

        self.conv = nn.Conv1d(embed_dim, num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, input_ids):
        x = self.embedding(input_ids)            # [batch, seq_len, embed_dim]
        x = x.permute(0, 2, 1)                  # [batch, embed_dim, seq_len]
        x = torch.tanh(self.conv(x))            # [batch, num_filters, seq_len]
        x = self.pool(x)                         # [batch, num_filters, seq_len//2]
        return x.permute(0, 2, 1)               # [batch, seq_len//2, num_filters]


class CNNLSTMServerB(nn.Module):
    """
    Server-side (split B): LSTM + Classifier.
    Receives compressed CNN features [batch, seq_len//2, num_filters] from client.
    """

    def __init__(self, num_filters=128, hidden_dim=128, num_layers=1, num_classes=3, dropout=0.3):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size = num_filters, hidden_size = hidden_dim,
            num_layers = num_layers, batch_first = True, dropout = lstm_dropout,
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, hidden):
        # hidden: [batch, seq_len//2, num_filters]
        _, (h_n, _) = self.lstm(hidden)
        return self.classifier(self.dropout(h_n[-1]))


# ── Factory helpers ───────────────────────────────────────────────────────────

def make_cnn_lstm(vocab_size, embed_dim=100, num_filters=128, hidden_dim=128,
                  num_classes=3, embed_weights=None):
    return CNNLSTMClassifier(
        vocab_size, embed_dim=embed_dim, num_filters=num_filters,
        hidden_dim=hidden_dim, num_classes=num_classes, embed_weights=embed_weights,
    )


def make_cnn_lstm_split(vocab_size, split_point="B", embed_dim=100, num_filters=128,
                        kernel_size=3, hidden_dim=128, num_classes=3, embed_weights=None):
    """
    Return (client, server) pair for CNN-LSTM split learning.

    split_point:
      'A' — split after embedding (minimal client)
      'B' — split after CNN     (recommended; semantically meaningful)
    """
    if split_point.upper() == "A":
        client = CNNLSTMClientA(vocab_size, embed_dim=embed_dim, embed_weights=embed_weights)
        server = CNNLSTMServerA(embed_dim=embed_dim, num_filters=num_filters,
                                kernel_size=kernel_size, hidden_dim=hidden_dim,
                                num_classes=num_classes)
    else:  # B
        client = CNNLSTMClientB(vocab_size, embed_dim=embed_dim, num_filters=num_filters,
                                kernel_size=kernel_size, embed_weights=embed_weights)
        server = CNNLSTMServerB(num_filters=num_filters, hidden_dim=hidden_dim,
                                num_classes=num_classes)
    return client, server
