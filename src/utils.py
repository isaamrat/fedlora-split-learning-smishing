"""
utils.py — shared helpers used across all training scripts.

Sob scripts e ei file import kora hoy — path, label mapping, device selection, data loading.
PROJECT_ROOT use kore relative path determine kora hoy — hardcoded path nai, portable.
"""

import re
import json
import random
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_SPLITS  = PROJECT_ROOT / "data" / "splits"
DATA_CLIENTS = PROJECT_ROOT / "data" / "clients"
REPORTS_DIR  = PROJECT_ROOT / "reports"
MODELS_DIR   = PROJECT_ROOT / "models"

# Label encoding: ham=0, spam=1, smishing=2
LABEL2ID = {"ham": 0, "spam": 1, "smishing": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = 3


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_split(split: str = "train") -> pd.DataFrame:
    """Load a data split (train / val / test)."""
    path = DATA_SPLITS / f"{split}.csv"
    return pd.read_csv(path, encoding="utf-8", low_memory=False)


def load_client(client_id: str) -> pd.DataFrame:
    path = DATA_CLIENTS / f"{client_id}.csv"
    return pd.read_csv(path, encoding="utf-8", low_memory=False)


def encode_labels(series: pd.Series) -> np.ndarray:
    return series.map(LABEL2ID).values


def get_class_weights(y: np.ndarray) -> np.ndarray:
    """Compute balanced class weights for sklearn or manual use."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.array(sorted(set(y)))
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return weights


def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    except ImportError:
        return None


def append_result(results_path: Path, row: dict):
    """Append one result row to a CSV, creating it if needed."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if results_path.exists():
        df_old = pd.read_csv(results_path)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(results_path, index=False)


def save_json(obj: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
