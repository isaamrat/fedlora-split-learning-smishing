"""
evaluate_on_clean_test.py
Re-evaluates saved models on the clean test set (near-dup leakage removed).

Compares each model's dirty-test vs clean-test metrics so we can quantify
how much near-duplicate inflation affected the original reported numbers.

Usage:
  python src/evaluate_on_clean_test.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_split, encode_labels,
    LABEL2ID, ID2LABEL, NUM_LABELS,
    MODELS_DIR, REPORTS_DIR, set_seed, get_device, PROJECT_ROOT,
)
from evaluate import compute_metrics
import pandas as pd

set_seed(42)
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN    = 128
BATCH      = 32


class SMSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(list(texts), truncation=True, padding="max_length",
                             max_length=MAX_LEN, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def make_loader(texts, labels, tok):
    return DataLoader(SMSDataset(texts, labels, tok), batch_size=BATCH, shuffle=False)


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits
        preds.extend(logits.argmax(-1).cpu().numpy())
        trues.extend(batch["labels"].numpy())
    return np.array(trues), np.array(preds)


def load_peft(adapter_path, device):
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS,
        id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
    return model.to(device)


def fmt(m):
    return (f"F1={m['macro_f1']:.4f}  "
            f"SmishF1={m['per_class'].get('smishing',{}).get('f1',0):.3f}  "
            f"FNR={m['smishing_fnr']:.4f}  "
            f"FPR={m['smishing_fpr']:.4f}")


def main():
    device = get_device()
    if device is None or str(device) == "cpu":
        raise RuntimeError("GPU required.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # test_clean.csv included in data/splits/ — ekhane evaluate kora hobe
    test_clean_path = PROJECT_ROOT / "data" / "splits" / "test_clean.csv"
    if not test_clean_path.exists():
        print("test_clean.csv not found. Check data/splits/ — it should be included.")
        sys.exit(1)
    test_clean = pd.read_csv(test_clean_path)

    # Optional: also load dirty test if present (for inflation comparison)
    test_dirty_path = PROJECT_ROOT / "data" / "splits" / "test.csv"
    test_dirty = pd.read_csv(test_dirty_path) if test_dirty_path.exists() else None

    def make_loaders(df):
        X = df["cleaned_text"].fillna("").values
        y = encode_labels(df["label"])
        return make_loader(X, y, tokenizer)

    clean_loader = make_loaders(test_clean)
    dirty_loader = make_loaders(test_dirty) if test_dirty is not None else None

    print(f"Clean test: {len(test_clean):,} rows")
    if test_dirty is not None:
        print(f"Dirty test: {len(test_dirty):,} rows (for comparison only)")

    # Evaluate all adapters found in models/fedlora/ and models/centralized/
    fedlora_dir  = MODELS_DIR / "fedlora"
    central_dir  = MODELS_DIR / "centralized"
    models_to_eval = []
    for d in sorted(fedlora_dir.glob("*")) if fedlora_dir.exists() else []:
        if (d / "adapter_config.json").exists():
            models_to_eval.append((d.name, d))
    if central_dir.exists() and (central_dir / "adapter_config.json").exists():
        models_to_eval.insert(0, ("centralized_lora", central_dir))

    if not models_to_eval:
        print("No adapter folders found in models/. Train a model first.")
        sys.exit(1)

    rows = []
    for name, path in models_to_eval:
        print(f"\n{'='*55}\n  {name}")
        model = load_peft(path, device)

        yt_c, yp_c = run_eval(model, clean_loader, device)
        mc = compute_metrics(yt_c, yp_c)
        print(f"  Clean: {fmt(mc)}")

        row = {
            "model":           name,
            "clean_macro_f1":  round(mc["macro_f1"], 4),
            "clean_smishing_f1": round(mc["per_class"].get("smishing", {}).get("f1", 0), 3),
            "clean_fnr":       round(mc["smishing_fnr"], 4),
        }

        if dirty_loader is not None:
            yt_d, yp_d = run_eval(model, dirty_loader, device)
            md = compute_metrics(yt_d, yp_d)
            print(f"  Dirty: {fmt(md)}")
            delta = mc["smishing_fnr"] - md["smishing_fnr"]
            print(f"  Inflation: FNR {delta:+.4f}")
            row.update({
                "dirty_macro_f1":    round(md["macro_f1"], 4),
                "dirty_smishing_f1": round(md["per_class"].get("smishing", {}).get("f1", 0), 3),
                "dirty_fnr":         round(md["smishing_fnr"], 4),
                "fnr_delta":         round(delta, 4),
            })

        del model
        torch.cuda.empty_cache()
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = REPORTS_DIR / "results_clean_vs_dirty_test.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nComparison saved -> {out_csv}")
    print("\nSummary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
