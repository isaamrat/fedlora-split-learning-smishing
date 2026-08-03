"""
models/textcnn.py — TextCNN for SMS classification (Kim 2014).

Architecture:
  Embedding → parallel Conv1d filters (sizes 2, 3, 4) → MaxPool → Concat → Dropout → Linear

Split learning support (split point A — after embedding):
  TextCNNClient: Embedding layer only → outputs [batch, seq_len, embed_dim]
  TextCNNServer: Conv + Pool + Classifier → receives hidden, returns logits

This is the natural single split point for TextCNN since the convolutional
filters are the core feature extractor and should stay together on the server.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    """
    Full TextCNN model.

    Args:
        vocab_size:    size of the vocabulary
        embed_dim:     embedding dimension (100 for GloVe-100d)
        num_classes:   number of output classes (3: ham/spam/smishing)
        filter_sizes:  list of convolution kernel sizes
        num_filters:   number of filters per kernel size
        dropout:       dropout probability before classifier
        embed_weights: optional pretrained embedding matrix [vocab_size, embed_dim]
        freeze_embed:  if True, pretrained embeddings are not updated during training
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        num_classes: int = 3,
        filter_sizes=(2, 3, 4),
        num_filters: int = 128,
        dropout: float = 0.5,
        embed_weights=None,
        freeze_embed: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embed_weights is not None:
            self.embedding.weight = nn.Parameter(embed_weights.clone().float())
            if freeze_embed:
                self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, input_ids):
        # input_ids: [batch, seq_len]
        x = self.embedding(input_ids)          # [batch, seq_len, embed_dim]
        x = x.permute(0, 2, 1)                 # [batch, embed_dim, seq_len]  (Conv1d expects this)

        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))                # [batch, num_filters, seq_len - fs + 1]
            c = F.max_pool1d(c, c.size(2))     # [batch, num_filters, 1]
            pooled.append(c.squeeze(2))        # [batch, num_filters]

        x = torch.cat(pooled, dim=1)           # [batch, num_filters * len(filter_sizes)]
        x = self.dropout(x)
        return self.classifier(x)              # [batch, num_classes]


# ── Split learning components ─────────────────────────────────────────────────
# Split point A: after the embedding layer
# Client sends [batch, seq_len, embed_dim] to the server.

class TextCNNClient(nn.Module):
    """
    Client-side TextCNN — embedding layer only.
    Outputs intermediate hidden states [batch, seq_len, embed_dim].
    """

    def __init__(self, vocab_size: int, embed_dim: int = 100, embed_weights=None, freeze_embed: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embed_weights is not None:
            self.embedding.weight = nn.Parameter(embed_weights.clone().float())
            if freeze_embed:
                self.embedding.weight.requires_grad = False

    def forward(self, input_ids):
        return self.embedding(input_ids)       # [batch, seq_len, embed_dim]


class TextCNNServer(nn.Module):
    """
    Server-side TextCNN — Conv filters + MaxPool + Classifier.
    Receives [batch, seq_len, embed_dim] from the client.
    """

    def __init__(
        self,
        embed_dim: int = 100,
        num_classes: int = 3,
        filter_sizes=(2, 3, 4),
        num_filters: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, hidden):
        # hidden: [batch, seq_len, embed_dim]
        x = hidden.permute(0, 2, 1)            # [batch, embed_dim, seq_len]

        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            c = F.max_pool1d(c, c.size(2))
            pooled.append(c.squeeze(2))

        x = torch.cat(pooled, dim=1)
        x = self.dropout(x)
        return self.classifier(x)             # [batch, num_classes]


# ── Factory helpers ───────────────────────────────────────────────────────────

def make_textcnn(vocab_size, embed_dim=100, num_classes=3, embed_weights=None):
    return TextCNN(vocab_size, embed_dim=embed_dim, num_classes=num_classes,
                   embed_weights=embed_weights)


def make_textcnn_split(vocab_size, embed_dim=100, num_classes=3, embed_weights=None):
    """Return (client, server) split pair initialised from the same embeddings."""
    client = TextCNNClient(vocab_size, embed_dim=embed_dim, embed_weights=embed_weights)
    server = TextCNNServer(embed_dim=embed_dim, num_classes=num_classes)
    return client, server
