"""
models/lstm.py — LSTM and BiLSTM classifiers for SMS classification.

Architecture (LSTM):
  Embedding → LSTM (1 or 2 layers) → Last hidden state → Dropout → Linear

Architecture (BiLSTM):
  Embedding → Bidirectional LSTM → Concat forward+backward final states → Dropout → Linear

Split learning support (split point A — after embedding):
  {LSTM,BiLSTM}Client: Embedding → outputs [batch, seq_len, embed_dim]
  {LSTM,BiLSTM}Server: LSTM/BiLSTM + Classifier → receives hidden, returns logits

For 2-layer variants, split point B (after layer 1) is also available:
  {LSTM,BiLSTM}ClientB: Embedding + LSTM layer 1
  {LSTM,BiLSTM}ServerB: LSTM layer 2 + Classifier
"""

import torch
import torch.nn as nn


# ── Full models ────────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """
    Unidirectional LSTM classifier.

    Args:
        vocab_size:    vocabulary size
        embed_dim:     embedding dimension
        hidden_dim:    LSTM hidden state dimension
        num_layers:    number of stacked LSTM layers (1 or 2)
        num_classes:   number of output classes
        dropout:       dropout between layers and before classifier
        embed_weights: optional pretrained embedding tensor
        freeze_embed:  freeze embedding weights if True
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 256,
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

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size  = embed_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = lstm_dropout,
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids)          # [batch, seq_len, embed_dim]
        _, (h_n, _) = self.lstm(x)             # h_n: [num_layers, batch, hidden_dim]
        x = h_n[-1]                            # take last layer's hidden state
        x = self.dropout(x)
        return self.classifier(x)              # [batch, num_classes]


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier.
    Forward and backward final hidden states are concatenated → [batch, 2*hidden_dim].
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
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

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size   = embed_dim,
            hidden_size  = hidden_dim,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional = True,
            dropout      = lstm_dropout,
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bi-directional

    def forward(self, input_ids):
        x = self.embedding(input_ids)          # [batch, seq_len, embed_dim]
        _, (h_n, _) = self.lstm(x)             # h_n: [num_layers*2, batch, hidden_dim]
        # Concat last-layer forward and backward
        fwd = h_n[-2]                          # forward  final state
        bwd = h_n[-1]                          # backward final state
        x   = torch.cat([fwd, bwd], dim=1)    # [batch, hidden_dim*2]
        x   = self.dropout(x)
        return self.classifier(x)


# ── Split learning — Split point A (after embedding) ─────────────────────────

class LSTMClient(nn.Module):
    """Client: embedding only → [batch, seq_len, embed_dim]"""

    def __init__(self, vocab_size, embed_dim=100, embed_weights=None, freeze_embed=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embed_weights is not None:
            self.embedding.weight = nn.Parameter(embed_weights.clone().float())
            if freeze_embed:
                self.embedding.weight.requires_grad = False

    def forward(self, input_ids):
        return self.embedding(input_ids)


class LSTMServer(nn.Module):
    """Server: LSTM + Classifier → receives [batch, seq_len, embed_dim], returns logits."""

    def __init__(self, embed_dim=100, hidden_dim=256, num_layers=1, num_classes=3, dropout=0.3):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size = embed_dim, hidden_size = hidden_dim,
            num_layers = num_layers, batch_first = True, dropout = lstm_dropout,
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, hidden):
        _, (h_n, _) = self.lstm(hidden)
        x = self.dropout(h_n[-1])
        return self.classifier(x)


class BiLSTMClient(nn.Module):
    """Client: embedding only → [batch, seq_len, embed_dim]"""

    def __init__(self, vocab_size, embed_dim=100, embed_weights=None, freeze_embed=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embed_weights is not None:
            self.embedding.weight = nn.Parameter(embed_weights.clone().float())
            if freeze_embed:
                self.embedding.weight.requires_grad = False

    def forward(self, input_ids):
        return self.embedding(input_ids)


class BiLSTMServer(nn.Module):
    """Server: BiLSTM + Classifier → receives [batch, seq_len, embed_dim], returns logits."""

    def __init__(self, embed_dim=100, hidden_dim=128, num_layers=1, num_classes=3, dropout=0.3):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size = embed_dim, hidden_size = hidden_dim,
            num_layers = num_layers, batch_first = True,
            bidirectional = True, dropout = lstm_dropout,
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, hidden):
        _, (h_n, _) = self.lstm(hidden)
        fwd = h_n[-2]; bwd = h_n[-1]
        x   = torch.cat([fwd, bwd], dim=1)
        return self.classifier(self.dropout(x))


# ── Factory helpers ───────────────────────────────────────────────────────────

def make_lstm(vocab_size, embed_dim=100, hidden_dim=256, num_classes=3,
              bidirectional=False, embed_weights=None):
    if bidirectional:
        return BiLSTMClassifier(vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim,
                                num_classes=num_classes, embed_weights=embed_weights)
    return LSTMClassifier(vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim,
                          num_classes=num_classes, embed_weights=embed_weights)


def make_lstm_split(vocab_size, embed_dim=100, hidden_dim=256, num_classes=3,
                    bidirectional=False, embed_weights=None):
    """Return (client, server) split pair for LSTM or BiLSTM."""
    if bidirectional:
        client = BiLSTMClient(vocab_size, embed_dim=embed_dim, embed_weights=embed_weights)
        server = BiLSTMServer(embed_dim=embed_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    else:
        client = LSTMClient(vocab_size, embed_dim=embed_dim, embed_weights=embed_weights)
        server = LSTMServer(embed_dim=embed_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    return client, server
