"""
shared/vocab_dataset.py — Word-index based Dataset and DataLoader for classical models.

Unlike the BERT-tokenizer version in dataset.py, this converts texts to sequences of
integer word indices using a pre-built vocabulary (data/vocab.json).  Sequences are
padded or truncated to a fixed length.

Also provides:
  load_vocab()          — load vocab.json from the standard location
  load_glove()          — load GloVe txt embeddings into a weight matrix
  build_embedding_matrix() — map vocab indices to pretrained embedding vectors
"""

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).resolve().parent   # src/shared/
_ROOT        = _HERE.parent.parent               # project root
VOCAB_PATH   = _ROOT / "data" / "vocab.json"
GLOVE_PATH   = _ROOT / "data" / "glove.6B.100d.txt"   # user must download if using GloVe


# ── Tokeniser (must match build_vocab.py) ─────────────────────────────────────

def simple_tokenise(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    return text.split()


# ── Vocabulary helpers ─────────────────────────────────────────────────────────

def load_vocab(path: Optional[Path] = None) -> dict:
    """Load vocab.json. Raises FileNotFoundError with a helpful message."""
    p = path or VOCAB_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Vocabulary not found at {p}.\n"
            "Run first:  python src/shared/build_vocab.py"
        )
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_glove(glove_path: Optional[Path] = None) -> dict:
    """
    Load GloVe vectors from a .txt file into a {word: np.array} dict.

    Download (if not present):
      wget https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
      unzip glove.6B.zip -d data/
    """
    p = glove_path or GLOVE_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"GloVe file not found at {p}.\n"
            "Download: https://nlp.stanford.edu/data/glove.6B.zip\n"
            "Extract glove.6B.100d.txt into data/"
        )
    vectors = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word  = parts[0]
            vec   = np.array(parts[1:], dtype=np.float32)
            vectors[word] = vec
    print(f"Loaded {len(vectors):,} GloVe vectors from {p.name}")
    return vectors


def build_embedding_matrix(
    vocab: dict,
    embed_dim: int = 100,
    glove_path: Optional[Path] = None,
) -> torch.Tensor:
    """
    Build an embedding weight matrix of shape [vocab_size, embed_dim].

    If GloVe vectors are available, they are used; unknown words get random
    vectors (uniform [-0.1, 0.1]).  <PAD> (index 0) is always all-zeros.

    Args:
        vocab:      word→index dict from load_vocab()
        embed_dim:  embedding dimension (must match GloVe file if using GloVe)
        glove_path: path to GloVe .txt file; if None, uses default GLOVE_PATH

    Returns:
        FloatTensor of shape [vocab_size, embed_dim]
    """
    glove = {}
    glove_file = glove_path or GLOVE_PATH
    if glove_file.exists():
        glove = load_glove(glove_file)
    else:
        print("GloVe not found — using random embeddings. "
              "For better performance download glove.6B.zip into data/.")

    vocab_size = len(vocab)
    matrix = np.random.uniform(-0.1, 0.1, (vocab_size, embed_dim)).astype(np.float32)
    matrix[0] = np.zeros(embed_dim)   # <PAD> = zero vector

    hits = 0
    for word, idx in vocab.items():
        if word in glove:
            matrix[idx] = glove[word]
            hits += 1

    coverage = hits / max(vocab_size - 2, 1) * 100   # exclude special tokens
    print(f"Embedding matrix: {vocab_size:,} × {embed_dim}  |  "
          f"GloVe coverage: {hits:,}/{vocab_size-2:,} ({coverage:.1f}%)")
    return torch.tensor(matrix)


# ── Dataset ───────────────────────────────────────────────────────────────────

class VocabSMSDataset(Dataset):
    """
    Converts SMS texts to fixed-length integer index sequences.

    PAD_IDX = 0 (pad short sequences)
    UNK_IDX = 1 (out-of-vocabulary tokens)
    """

    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, texts, labels, vocab: dict, max_len: int = 128):
        self.max_len = max_len
        self.vocab   = vocab
        self.seqs    = [self._encode(t) for t in texts]
        self.labels  = torch.tensor(labels, dtype=torch.long)

    def _encode(self, text: str) -> torch.Tensor:
        tokens = simple_tokenise(str(text))[:self.max_len]
        ids    = [self.vocab.get(t, self.UNK_IDX) for t in tokens]
        # Pad to max_len
        pad_len = self.max_len - len(ids)
        ids     = ids + [self.PAD_IDX] * pad_len
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"input_ids": self.seqs[idx], "labels": self.labels[idx]}


def make_vocab_loader(
    texts,
    labels,
    vocab: dict,
    max_len: int = 128,
    batch_size: int = 16,
    shuffle: bool = True,
) -> DataLoader:
    """Return a DataLoader over a VocabSMSDataset."""
    ds = VocabSMSDataset(texts, labels, vocab, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
