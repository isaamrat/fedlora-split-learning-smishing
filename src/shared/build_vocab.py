"""
shared/build_vocab.py — One-off vocabulary builder for classical deep learning models.

Reads the training split, tokenises by whitespace (with basic cleaning), and builds a
word-to-index vocabulary.  The result is saved to data/vocab.json so all classical model
training scripts can load it without rebuilding.

Usage (run once from the project root):
    python src/shared/build_vocab.py
    python src/shared/build_vocab.py --max_vocab 30000 --min_freq 2

Output: data/vocab.json  — {"<PAD>": 0, "<UNK>": 1, "word": 2, ...}
"""

import re
import json
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd

# ── Project root resolution (works regardless of cwd) ─────────────────────────
_HERE = Path(__file__).resolve().parent          # src/shared/
_SRC  = _HERE.parent                             # src/
_ROOT = _SRC.parent                              # project root

import sys
sys.path.insert(0, str(_SRC))
from utils import DATA_SPLITS, PROJECT_ROOT


# ── Tokeniser ─────────────────────────────────────────────────────────────────

def simple_tokenise(text: str):
    """Lowercase + strip non-alphanumeric (keep apostrophes for contractions)."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    return text.split()


# ── Main ──────────────────────────────────────────────────────────────────────

def build_vocab(max_vocab: int = 30_000, min_freq: int = 1) -> dict:
    """
    Read train.csv, count token frequencies, return word→index dict.
    Special tokens:
      0 = <PAD>  (padding)
      1 = <UNK>  (out-of-vocabulary)
    """
    train_path = DATA_SPLITS / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Training split not found: {train_path}")

    df = pd.read_csv(train_path, encoding="utf-8", low_memory=False)
    texts = df["cleaned_text"].fillna("").tolist()

    counter: Counter = Counter()
    for text in texts:
        counter.update(simple_tokenise(text))

    # Filter by min frequency, sort by frequency descending
    vocab_words = [w for w, c in counter.most_common() if c >= min_freq]
    vocab_words = vocab_words[:max_vocab - 2]   # reserve 2 slots for special tokens

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for idx, word in enumerate(vocab_words, start=2):
        vocab[word] = idx

    print(f"Vocabulary size: {len(vocab):,}  (unique tokens in train: {len(counter):,})")
    return vocab


def main():
    parser = argparse.ArgumentParser(description="Build vocabulary from training data.")
    parser.add_argument("--max_vocab", type=int, default=30_000,
                        help="Maximum number of vocabulary entries (default: 30000)")
    parser.add_argument("--min_freq",  type=int, default=1,
                        help="Minimum token frequency to include (default: 1)")
    parser.add_argument("--output",    type=str, default=None,
                        help="Output path (default: data/vocab.json)")
    args = parser.parse_args()

    vocab = build_vocab(max_vocab=args.max_vocab, min_freq=args.min_freq)

    output_path = Path(args.output) if args.output else PROJECT_ROOT / "data" / "vocab.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    print(f"Vocabulary saved -> {output_path}")


if __name__ == "__main__":
    main()
