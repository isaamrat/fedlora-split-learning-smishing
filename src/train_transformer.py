"""
train_transformer.py
E1: Centralized DistilBERT fine-tuning (full model)
E2: DistilBERT + LoRA adapter (PEFT) — set --lora flag

Usage:
  python src/train_transformer.py              # full fine-tune (E1)
  python src/train_transformer.py --lora       # LoRA adapters (E2)

Reads:  data/splits/train.csv, val.csv, test_clean.csv
Saves:  models/distilbert_central/ or models/distilbert_lora_central/
        reports/results_transformer_*.csv
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.utils.class_weight import compute_class_weight

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("WARNING: peft not installed. LoRA mode unavailable.")

sys.path.insert(0, str(Path(__file__).parent))
from utils import (load_split, encode_labels, LABEL2ID, ID2LABEL,
                   NUM_LABELS, MODELS_DIR, REPORTS_DIR, set_seed, get_device)
from evaluate import compute_metrics, report_metrics

MODELS_DIR.mkdir(parents=True, exist_ok=True)
set_seed(42)

MODEL_NAME  = "distilbert-base-uncased"
MAX_LEN     = 128
BATCH_SIZE  = 32
EPOCHS      = 4
LR          = 2e-5
WARMUP_FRAC = 0.1

LORA_CONFIG = dict(
    r              = 8,
    lora_alpha     = 16,
    lora_dropout   = 0.1,
    bias           = "none",
    task_type      = TaskType.SEQ_CLS,
    target_modules = ["q_lin", "v_lin"],
)


class SMSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def train_one_epoch(model, loader, optimizer, scheduler, device, class_weights_tensor):
    model.train()
    total_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))

    for batch in loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits
        loss    = loss_fn(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", action="store_true", help="Use LoRA adapters (E2)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args()

    experiment = "DistilBERT + LoRA (E2)" if args.lora else "DistilBERT Centralized (E1)"
    tag = "distilbert_lora_central" if args.lora else "distilbert_central"
    out_dir = MODELS_DIR / tag

    if args.lora and not PEFT_AVAILABLE:
        print("ERROR: peft not installed. Run: pip install peft")
        sys.exit(1)

    device = get_device()
    if device is None or str(device) == "cpu":
        raise RuntimeError(
            "GPU (CUDA) not available or not working. "
            "Neural network training requires the RTX 5060 Ti GPU. "
            "Fix PyTorch CUDA installation before running this script."
        )
    print(f"Device: {device}  ({torch.cuda.get_device_name(0)})")
    print(f"Experiment: {experiment}")

    print("\nLoading splits...")
    train_df = load_split("train")
    val_df   = load_split("val")
    test_df  = load_split("test_clean")  # near-duplicate leakage removed

    X_train = train_df["cleaned_text"].fillna("").values
    y_train = encode_labels(train_df["label"])
    X_val   = val_df["cleaned_text"].fillna("").values
    y_val   = encode_labels(val_df["label"])
    X_test  = test_df["cleaned_text"].fillna("").values
    y_test  = encode_labels(test_df["label"])

    # Class weights
    classes = np.array(sorted(set(y_train)))
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)
    print(f"Class weights: {dict(zip(classes.tolist(), weights.round(3).tolist()))}")

    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels   = NUM_LABELS,
        id2label     = ID2LABEL,
        label2id     = LABEL2ID,
        ignore_mismatched_sizes = True,
    )

    if args.lora:
        lora_cfg = LoraConfig(**LORA_CONFIG)
        model = get_peft_model(model, lora_cfg)
        trainable, total = model.get_nb_trainable_parameters()
        print(f"LoRA trainable params: {trainable:,} / {total:,} "
              f"({100 * trainable / total:.2f}%)")
    else:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable:,} / {total:,}")

    model = model.to(device)

    print("Tokenizing...")
    train_ds = SMSDataset(X_train, y_train, tokenizer)
    val_ds   = SMSDataset(X_val,   y_val,   tokenizer)
    test_ds  = SMSDataset(X_test,  y_test,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    total_steps  = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * WARMUP_FRAC)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_f1 = 0.0
    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                               device, class_weights_tensor)
        y_true_val, y_pred_val = evaluate_model(model, val_loader, device)
        val_metrics = compute_metrics(y_true_val, y_pred_val)
        print(f"  Epoch {epoch}/{args.epochs}  loss={loss:.4f}  "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}  "
              f"val_smishing_fnr={val_metrics['smishing_fnr']}")

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            out_dir.mkdir(parents=True, exist_ok=True)
            if args.lora:
                model.save_pretrained(str(out_dir))
            else:
                model.save_pretrained(str(out_dir))
            tokenizer.save_pretrained(str(out_dir))
            print(f"    Checkpoint saved (val macro_f1={best_val_f1:.4f})")

    print("\nEvaluating on test set (best checkpoint)...")
    y_true_test, y_pred_test = evaluate_model(model, test_loader, device)
    test_metrics = compute_metrics(y_true_test, y_pred_test)
    report_metrics(
        test_metrics, experiment, tag,
        y_true_test, y_pred_test,
        extra={
            "model":      MODEL_NAME,
            "epochs":     args.epochs,
            "batch_size": args.batch_size,
            "lr":         args.lr,
            "lora":       args.lora,
            "trainable_params": trainable,
        }
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
