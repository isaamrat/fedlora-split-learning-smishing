"""
train_fedlora.py — Federated LoRA Training (FedSmishGuard main experiment)

E3: Local-only training   — each client trains independently, no server (--local flag)
E4: FedLoRA               — clients train locally, server aggregates LoRA adapters

Ekhane real federated network simulate kora hocche — 5 ta client ek machine e train korbe.
Kono raw data share hoy na — shudhu LoRA adapter weights (2.96MB) exchange hoy.

Usage:
  python src/train_fedlora.py                                      # FedLoRA, default settings
  python src/train_fedlora.py --local                              # local-only (E3)
  python src/train_fedlora.py --rounds 20 --agg_weight smishing    # 20-round smishing-weighted

Key arguments:
  --rounds        Number of communication rounds (default: 10)
  --local_epochs  Local training epochs per client per round (default: 2)
  --clients_dir   Client CSV folder (default: data/clients)
  --agg_weight    Aggregation: smishing (best) | total | sqrt | balanced | uniform
  --resume        Resume from saved checkpoint
"""

import sys
import copy
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

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
    from peft import (
        get_peft_model, LoraConfig, TaskType,
        PeftModel, get_peft_model_state_dict, set_peft_model_state_dict,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_split, load_client, encode_labels,
    LABEL2ID, ID2LABEL, NUM_LABELS,
    MODELS_DIR, REPORTS_DIR, DATA_CLIENTS,
    set_seed, get_device, append_result,
)
from evaluate import compute_metrics, report_metrics

set_seed(42)

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME    = "distilbert-base-uncased"
MAX_LEN       = 128
LOCAL_BATCH   = 16
LOCAL_EPOCHS  = 1
COMM_ROUNDS   = 5
LR            = 2e-4

CLIENT_IDS = ["client_1", "client_2", "client_3", "client_4", "client_5"]

LORA_CONFIG = dict(
    r              = 8,
    lora_alpha     = 16,
    lora_dropout   = 0.1,
    bias           = "none",
    task_type      = TaskType.SEQ_CLS,
    target_modules = ["q_lin", "v_lin"],
)


# ── Dataset ────────────────────────────────────────────────────────────────────

class SMSDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding="max_length",
            max_length=max_len, return_tensors="pt"
        )
        self.encodings["attention_mask"] = self.encodings["attention_mask"].bool()
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def make_loader(texts, labels, tokenizer, shuffle=True):
    ds = SMSDataset(texts, labels, tokenizer)
    return DataLoader(ds, batch_size=LOCAL_BATCH, shuffle=shuffle, num_workers=0)


# ── Local training ─────────────────────────────────────────────────────────────

def local_train(model, loader, device, class_weights, n_epochs, lr):
    """Train model for n_epochs on loader. Returns updated model."""
    loss_fn   = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    total_steps  = len(loader) * n_epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.train()
    for _ in range(n_epochs):
        for batch in loader:
            optimizer.zero_grad()
            logits = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device).bool(),
            ).logits
            loss = loss_fn(logits, batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        logits = model(
            input_ids      = batch["input_ids"].to(device),
            attention_mask = batch["attention_mask"].to(device).bool(),
        ).logits
        preds.extend(logits.argmax(dim=-1).cpu().numpy())
        trues.extend(batch["labels"].numpy())
    return np.array(trues), np.array(preds)


# ── FedAvg over LoRA weights ───────────────────────────────────────────────────

def fedavg_lora(global_model, client_state_dicts: List[dict], client_sizes: List[int],
                smishing_sizes: Optional[List[int]] = None,
                agg_weight: str = "smishing"):
    """
    Weighted FedAvg over LoRA adapter state dicts.

    agg_weight:
      'smishing' — weight proportional to smishing sample count (default)
      'sqrt'     — weight proportional to sqrt(smishing count); less dominated by data-rich clients
      'uniform'  — equal weight per client
      'total'    — weight proportional to total sample count (original naive FedAvg)
    """
    import math
    n = len(client_state_dicts)

    if agg_weight == "uniform":
        weights = [1.0 / n] * n
    elif agg_weight == "total":
        total = sum(client_sizes)
        weights = [s / max(total, 1) for s in client_sizes]
    elif agg_weight == "sqrt" and smishing_sizes:
        roots = [math.sqrt(max(s, 1)) for s in smishing_sizes]
        total = sum(roots)
        weights = [r / total for r in roots]
    elif agg_weight == "balanced" and smishing_sizes:
        # weight_i = 0.5 * (total_i / sum_total) + 0.5 * (smishing_i / sum_smishing)
        # Balances total data contribution with smishing signal contribution.
        sum_total    = max(sum(client_sizes), 1)
        sum_smishing = max(sum(smishing_sizes), 1)
        w_total   = [s / sum_total    for s in client_sizes]
        w_smishing = [max(s, 1) / sum_smishing for s in smishing_sizes]
        weights = [0.5 * wt + 0.5 * ws for wt, ws in zip(w_total, w_smishing)]
    else:  # default: smishing-count linear (best option)
        # Smishing count diye weight kora — je client er beshi smishing data ache, tar weight beshi
        counts = smishing_sizes if smishing_sizes else client_sizes
        total = sum(max(c, 1) for c in counts)
        weights = [max(c, 1) / total for c in counts]

    # Weighted average of all client adapter states
    avg_state = {}
    for key in client_state_dicts[0]:
        tensors = [sd[key].float() for sd in client_state_dicts]
        avg_state[key] = sum(w * t for w, t in zip(weights, tensors))

    set_peft_model_state_dict(global_model, avg_state)
    return global_model


def fedavg_state_dict(reference_model, client_state_dicts: List[dict], client_sizes: List[int],
                      smishing_sizes: Optional[List[int]] = None,
                      agg_weight: str = "smishing"):
    """Weighted FedAvg over generic model state dicts."""
    import math
    n = len(client_state_dicts)

    if agg_weight == "uniform":
        weights = [1.0 / n] * n
    elif agg_weight == "total":
        total = sum(client_sizes)
        weights = [s / max(total, 1) for s in client_sizes]
    elif agg_weight == "sqrt" and smishing_sizes:
        roots = [math.sqrt(max(s, 1)) for s in smishing_sizes]
        total = sum(roots)
        weights = [r / total for r in roots]
    elif agg_weight == "balanced" and smishing_sizes:
        sum_total = max(sum(client_sizes), 1)
        sum_smishing = max(sum(smishing_sizes), 1)
        w_total = [s / sum_total for s in client_sizes]
        w_smishing = [max(s, 1) / sum_smishing for s in smishing_sizes]
        weights = [0.5 * wt + 0.5 * ws for wt, ws in zip(w_total, w_smishing)]
    else:
        counts = smishing_sizes if smishing_sizes else client_sizes
        total = sum(max(c, 1) for c in counts)
        weights = [max(c, 1) / total for c in counts]

    avg_state = {}
    for key in client_state_dicts[0]:
        tensors = [sd[key].float() for sd in client_state_dicts]
        avg_state[key] = sum(w * t for w, t in zip(weights, tensors))

    reference_model.load_state_dict(avg_state)
    return reference_model


class SplitDistilBertClient(torch.nn.Module):
    def __init__(self, base_model, split_layer: int):
        super().__init__()
        self.embeddings = copy.deepcopy(base_model.distilbert.embeddings)
        self.transformer = torch.nn.ModuleList(
            [copy.deepcopy(layer) for layer in base_model.distilbert.transformer.layer[:split_layer]]
        )

    def forward(self, input_ids, attention_mask):
        hidden = self.embeddings(input_ids)
        attention_mask = expand_split_attention_mask(attention_mask, hidden)
        for layer in self.transformer:
            hidden = first_hidden_state(layer(hidden, attention_mask=attention_mask))
        return hidden


class SplitDistilBertServer(torch.nn.Module):
    def __init__(self, base_model, split_layer: int):
        super().__init__()
        self.transformer = torch.nn.ModuleList(
            [copy.deepcopy(layer) for layer in base_model.distilbert.transformer.layer[split_layer:]]
        )
        self.pre_classifier = copy.deepcopy(base_model.pre_classifier)
        self.dropout = copy.deepcopy(base_model.dropout)
        self.classifier = copy.deepcopy(base_model.classifier)

    def forward(self, hidden, attention_mask):
        attention_mask = expand_split_attention_mask(attention_mask, hidden)
        for layer in self.transformer:
            hidden = first_hidden_state(layer(hidden, attention_mask=attention_mask))
        hidden = self.pre_classifier(hidden[:, 0])
        hidden = self.dropout(hidden)
        return self.classifier(hidden)


def expand_split_attention_mask(attention_mask, hidden_states):
    """Make a tokenizer mask broadcastable for directly-called DistilBERT layers.

    Full DistilBERT expands the 2D tokenizer mask internally. In split learning we
    bypass the parent model and call transformer layers ourselves, so newer
    Transformers SDPA attention needs the mask in [batch, 1, seq, seq] form.
    """
    if attention_mask is None or attention_mask.dim() != 2:
        return attention_mask
    batch_size, seq_len = hidden_states.shape[:2]
    key_len = attention_mask.shape[-1]
    return (
        attention_mask.to(device=hidden_states.device, dtype=torch.bool)
        .view(batch_size, 1, 1, key_len)
        .expand(batch_size, 1, seq_len, key_len)
        .contiguous()
    )


def first_hidden_state(layer_output):
    return layer_output[0] if isinstance(layer_output, (tuple, list)) else layer_output


def make_split_models(split_layer: int):
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels = NUM_LABELS,
        id2label   = ID2LABEL,
        label2id   = LABEL2ID,
        ignore_mismatched_sizes = True,
    )
    client = SplitDistilBertClient(base, split_layer)
    server = SplitDistilBertServer(base, split_layer)
    return client, server


def save_split_checkpoint(directory: Path, client_model, server_model, tokenizer):
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(client_model.state_dict(), directory / "client_state.pt")
    torch.save(server_model.state_dict(), directory / "server_state.pt")
    tokenizer.save_pretrained(str(directory))


def load_split_checkpoint(directory: Path, split_layer: int):
    client_model, server_model = make_split_models(split_layer)
    client_path = directory / "client_state.pt"
    server_path = directory / "server_state.pt"
    if client_path.exists() and server_path.exists():
        client_model.load_state_dict(torch.load(client_path, map_location="cpu"))
        server_model.load_state_dict(torch.load(server_path, map_location="cpu"))
    return client_model, server_model


def local_split_train(client_model, server_model, loader, device, class_weights, n_epochs, lr):
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(
        list(client_model.parameters()) + list(server_model.parameters()),
        lr=lr
    )
    total_steps = len(loader) * n_epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    client_model.train()
    server_model.train()

    for _ in range(n_epochs):
        for batch in loader:
            optimizer.zero_grad()
            hidden = client_model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device).bool(),
            )
            logits = server_model(hidden, attention_mask=batch["attention_mask"].to(device).bool())
            loss = loss_fn(logits, batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(client_model.parameters()) + list(server_model.parameters()), 1.0
            )
            optimizer.step()
            scheduler.step()

    return client_model, server_model


@torch.no_grad()
def evaluate_split(client_model, server_model, loader, device):
    client_model.eval()
    server_model.eval()
    preds, trues = [], []

    for batch in loader:
        hidden = client_model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device).bool(),
        )
        logits = server_model(hidden, attention_mask=batch["attention_mask"].to(device).bool())
        preds.extend(logits.argmax(dim=-1).cpu().numpy())
        trues.extend(batch["labels"].numpy())

    return np.array(trues), np.array(preds)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local",        action="store_true",
                        help="Local-only mode (E3): train each client independently, no aggregation")
    parser.add_argument("--split",        action="store_true",
                        help="Enable split learning with a client/server model partition")
    parser.add_argument("--split_layer",  type=int, default=3,
                        help="Number of DistilBERT layers to keep on the client side in split learning")
    parser.add_argument("--rounds",       type=int,   default=COMM_ROUNDS)
    parser.add_argument("--local_epochs", type=int,   default=LOCAL_EPOCHS)
    parser.add_argument("--lr",           type=float, default=LR)
    parser.add_argument("--resume",       action="store_true",
                        help="Resume from saved global adapter checkpoint")
    parser.add_argument("--clients_dir",  type=str,   default=None,
                        help="Path to client CSVs (default: data/clients/). Relative to project root.")
    parser.add_argument("--agg_weight",   type=str,   default="smishing",
                        choices=["smishing", "sqrt", "uniform", "total", "balanced"],
                        help="FedAvg weighting: smishing|sqrt|uniform|total|balanced")
    args = parser.parse_args()

    if not PEFT_AVAILABLE:
        print("ERROR: peft not installed. Run: pip install peft")
        sys.exit(1)

    from utils import PROJECT_ROOT
    clients_dir = (PROJECT_ROOT / args.clients_dir) if args.clients_dir else DATA_CLIENTS
    setting_name = clients_dir.name if args.clients_dir else "default"
    agg_tag = args.agg_weight if not args.local else "none"
    if not args.local and agg_tag != "smishing":
        setting_name = f"{setting_name}_{agg_tag}"

    if args.split:
        mode = "split_local" if args.local else "split_fed"
        experiment = ("Split Learning Local (E3)" if args.local
                      else f"Split Learning FedAvg (E4) [{setting_name}]")
    else:
        mode = "local_only" if args.local else "fedlora"
        experiment = ("Local-Only Client Training (E3)" if args.local
                      else f"FedLoRA FedAvg (E4) [{setting_name}]")
    print(
        f"Mode: {mode} | Rounds: {args.rounds} | Local epochs: {args.local_epochs} | "
        f"Split: {args.split} | Split layer: {args.split_layer if args.split else 'NA'} | "
        f"Setting: {setting_name} | Agg: {agg_tag}"
    )

    device = get_device()
    if device is None or str(device) == "cpu":
        raise RuntimeError(
            "GPU (CUDA) not available. FedLoRA requires RTX 5060 Ti. "
            "Fix PyTorch CUDA installation first."
        )
    import torch as _torch
    print(f"Device: {device}  ({_torch.cuda.get_device_name(0)})")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Load validation and test sets ─────────────────────────────────────────
    val_df  = load_split("val")
    test_df = load_split("test_clean")  # near-duplicate leakage removed

    X_val  = val_df["cleaned_text"].fillna("").values
    y_val  = encode_labels(val_df["label"])
    X_test = test_df["cleaned_text"].fillna("").values
    y_test = encode_labels(test_df["label"])

    val_loader  = make_loader(X_val,  y_val,  tokenizer, shuffle=False)
    test_loader = make_loader(X_test, y_test, tokenizer, shuffle=False)

    # ── Load client data ──────────────────────────────────────────────────────
    clients = {}
    for cid in CLIENT_IDS:
        try:
            csv_path = clients_dir / f"{cid}.csv"
            if not csv_path.exists():
                print(f"  WARNING: {cid}.csv not found in {clients_dir} — skipping")
                continue
            df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
            if len(df) < 10:
                print(f"  WARNING: {cid} has only {len(df)} rows — skipping")
                continue
            X = df["cleaned_text"].fillna("").values
            y = encode_labels(df["label"])
            # Per-client class weights
            classes = np.array(sorted(set(y)))
            cw = compute_class_weight("balanced", classes=classes, y=y)
            cw_dict = dict(zip(classes.tolist(), cw.tolist()))
            # Pad to NUM_LABELS if client is missing a class
            cw_tensor = torch.tensor(
                [cw_dict.get(i, 1.0) for i in range(NUM_LABELS)], dtype=torch.float32
            )
            clients[cid] = {"X": X, "y": y, "cw": cw_tensor, "n": len(y)}
            label_dist = dict(pd.Series(df["label"]).value_counts())
            print(f"  {cid}: {len(y)} rows | {label_dist}")
        except FileNotFoundError:
            print(f"  WARNING: {cid}.csv not found — skipping")

    if not clients:
        print("ERROR: No client data found. Check --clients_dir points to a folder with client_*.csv files (e.g. data/clients/setting_D_300).")
        sys.exit(1)

    active_clients = list(clients.keys())
    print(f"\nActive clients: {active_clients}")

    # ── Build global model or split learning models ───────────────────────────
    def make_base_model():
        base = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels = NUM_LABELS,
            id2label   = ID2LABEL,
            label2id   = LABEL2ID,
            ignore_mismatched_sizes = True,
        )
        lora_cfg = LoraConfig(**LORA_CONFIG)
        return get_peft_model(base, lora_cfg)

    global_adapter_dir = MODELS_DIR / "fedlora" / f"global_adapter_{setting_name}"
    global_adapter_dir.mkdir(parents=True, exist_ok=True)

    split_model_dir = MODELS_DIR / "split" / f"{setting_name}"
    split_model_dir.mkdir(parents=True, exist_ok=True)

    if args.split:
        print("\nInitialising split learning models...")
        if args.resume and (split_model_dir / "client_state.pt").exists() and (split_model_dir / "server_state.pt").exists():
            print(f"  Resuming split models from {split_model_dir}")
            global_client_model, global_server_model = load_split_checkpoint(split_model_dir, args.split_layer)
            global_client_model, global_server_model = global_client_model.to(device), global_server_model.to(device)
        else:
            global_client_model, global_server_model = make_split_models(args.split_layer)
            global_client_model, global_server_model = global_client_model.to(device), global_server_model.to(device)

        total = sum(p.numel() for p in global_client_model.parameters()) + sum(p.numel() for p in global_server_model.parameters())
        trainable = sum(p.numel() for p in global_client_model.parameters() if p.requires_grad) + sum(p.numel() for p in global_server_model.parameters() if p.requires_grad)
        print(f"Split model params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    else:
        print("\nInitialising global model...")
        if args.resume and (global_adapter_dir / "adapter_config.json").exists():
            print(f"  Resuming from {global_adapter_dir}")
            base = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
                ignore_mismatched_sizes=True,
            )
            global_model = PeftModel.from_pretrained(base, str(global_adapter_dir),
                                                     is_trainable=True).to(device)
        else:
            global_model = make_base_model().to(device)
        trainable, total = global_model.get_nb_trainable_parameters()
        print(f"LoRA trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    round_results = []

    if args.local:
        if args.split:
            # ── E3: Split learning local-only training ──────────────────────────
            print("\n[E3] Split learning local-only training (no aggregation)...")
            client_metrics = {}
            for cid in active_clients:
                print(f"\n  Training {cid}...")
                client_model, server_model = make_split_models(args.split_layer)
                client_model.load_state_dict(global_client_model.state_dict())
                server_model.load_state_dict(global_server_model.state_dict())
                client_model, server_model = client_model.to(device), server_model.to(device)

                loader = make_loader(clients[cid]["X"], clients[cid]["y"], tokenizer)
                client_model, server_model = local_split_train(
                    client_model, server_model, loader, device,
                    clients[cid]["cw"], args.local_epochs * args.rounds, args.lr
                )

                y_t, y_p = evaluate_split(client_model, server_model, test_loader, device)
                m = compute_metrics(y_t, y_p)
                client_metrics[cid] = m
                print(f"  {cid} test macro_f1={m['macro_f1']:.4f} smishing_fnr={m['smishing_fnr']}")

                save_dir = MODELS_DIR / "split" / f"{cid}_local_split"
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(client_model.state_dict(), save_dir / "client_state.pt")
                torch.save(server_model.state_dict(), save_dir / "server_state.pt")
                tokenizer.save_pretrained(str(save_dir))

            all_macro_f1 = np.mean([m["macro_f1"] for m in client_metrics.values()])
            print(f"\nLocal-only split avg macro_f1 across clients: {all_macro_f1:.4f}")

            append_result(
                REPORTS_DIR / "results_local_only_clients.csv",
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "experiment": "Split Learning Local (E3)",
                    "avg_macro_f1": round(all_macro_f1, 4),
                    **{f"{cid}_macro_f1": round(m["macro_f1"], 4)
                       for cid, m in client_metrics.items()},
                    "trainable_params": trainable,
                }
            )
        else:
            # ── E3: Local-only training ────────────────────────────────────────────
            print("\n[E3] Training each client independently (no aggregation)...")
            client_metrics = {}
            for cid in active_clients:
                print(f"\n  Training {cid}...")
                local_model = make_base_model().to(device)
                loader = make_loader(clients[cid]["X"], clients[cid]["y"], tokenizer)
                local_model = local_train(
                    local_model, loader, device,
                    clients[cid]["cw"], args.local_epochs * args.rounds, args.lr
                )
                y_t, y_p = evaluate(local_model, test_loader, device)
                m = compute_metrics(y_t, y_p)
                client_metrics[cid] = m
                print(f"  {cid} test macro_f1={m['macro_f1']:.4f} smishing_fnr={m['smishing_fnr']}")

                save_dir = MODELS_DIR / "fedlora" / f"{cid}_local_adapter"
                save_dir.mkdir(parents=True, exist_ok=True)
                local_model.save_pretrained(str(save_dir))

            all_macro_f1 = np.mean([m["macro_f1"] for m in client_metrics.values()])
            print(f"\nLocal-only avg macro_f1 across clients: {all_macro_f1:.4f}")

            append_result(
                REPORTS_DIR / "results_local_only_clients.csv",
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "experiment": "Local-Only (E3)",
                    "avg_macro_f1": round(all_macro_f1, 4),
                    **{f"{cid}_macro_f1": round(m["macro_f1"], 4)
                       for cid, m in client_metrics.items()},
                    "trainable_params": trainable,
                }
            )
    else:
        if args.split:
            # ── E4: Split Learning FedAvg ───────────────────────────────────────
            print(f"\n[E4] Split Learning FedAvg: {args.rounds} rounds, {args.local_epochs} local epoch(s) each, split_layer={args.split_layer}")
            best_val_f1 = 0.0
            best_round = 0
            best_split_dir = MODELS_DIR / "split" / f"{setting_name}_best"
            best_split_dir.mkdir(parents=True, exist_ok=True)

            for rnd in range(1, args.rounds + 1):
                print(f"\n--- Round {rnd}/{args.rounds} ---")
                client_states = []
                server_states = []
                client_sizes = []

                for cid in active_clients:
                    client_model, server_model = make_split_models(args.split_layer)
                    client_model.load_state_dict(global_client_model.state_dict())
                    server_model.load_state_dict(global_server_model.state_dict())
                    client_model, server_model = client_model.to(device), server_model.to(device)

                    loader = make_loader(clients[cid]["X"], clients[cid]["y"], tokenizer)
                    client_model, server_model = local_split_train(
                        client_model, server_model, loader, device,
                        clients[cid]["cw"], args.local_epochs, args.lr
                    )

                    client_states.append({k: v.cpu() for k, v in client_model.state_dict().items()})
                    server_states.append({k: v.cpu() for k, v in server_model.state_dict().items()})
                    client_sizes.append(clients[cid]["n"])

                    y_v, y_p = evaluate_split(client_model, server_model, val_loader, device)
                    m = compute_metrics(y_v, y_p)
                    print(f"  {cid}: val macro_f1={m['macro_f1']:.4f}")

                    del client_model
                    del server_model

                smishing_sizes = [
                    int((clients[cid]["y"] == LABEL2ID["smishing"]).sum())
                    for cid in active_clients
                ]
                global_client_model = fedavg_state_dict(
                    global_client_model, client_states, client_sizes,
                    smishing_sizes=smishing_sizes,
                    agg_weight=args.agg_weight
                )
                global_server_model = fedavg_state_dict(
                    global_server_model, server_states, client_sizes,
                    smishing_sizes=smishing_sizes,
                    agg_weight=args.agg_weight
                )

                y_v, y_p = evaluate_split(global_client_model, global_server_model, val_loader, device)
                global_metrics = compute_metrics(y_v, y_p)
                val_f1 = global_metrics["macro_f1"]
                print(f"  GLOBAL val macro_f1={val_f1:.4f}  "
                      f"smishing_fnr={global_metrics['smishing_fnr']}")

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_round = rnd
                    save_split_checkpoint(best_split_dir, global_client_model, global_server_model, tokenizer)
                    print(f"  [best checkpoint saved — R{rnd} val F1={val_f1:.4f}]")

                row = {
                    "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "round":          rnd,
                    "global_macro_f1": global_metrics["macro_f1"],
                    "global_weighted_f1": global_metrics["weighted_f1"],
                    "smishing_fnr":   global_metrics["smishing_fnr"],
                    "smishing_fpr":   global_metrics["smishing_fpr"],
                }
                round_results.append(row)
                append_result(REPORTS_DIR / "results_fedlora_rounds.csv", row)

            print("\n[E4] Final test evaluation (global split model)...")
            y_t, y_p = evaluate_split(global_client_model, global_server_model, test_loader, device)
            test_metrics = compute_metrics(y_t, y_p)
            report_metrics(
                test_metrics, experiment, "split_final",
                y_t, y_p,
                extra={
                    "rounds":          args.rounds,
                    "local_epochs":    args.local_epochs,
                    "lr":              args.lr,
                    "n_clients":       len(active_clients),
                    "trainable_params": trainable,
                    "comm_bytes_est":  trainable * 4 * len(active_clients) * args.rounds,
                }
            )

            save_split_checkpoint(split_model_dir, global_client_model, global_server_model, tokenizer)
            print(f"Global split model saved -> {split_model_dir}")
            print(f"Best split model (R{best_round}, val F1={best_val_f1:.4f}) saved -> {best_split_dir}")

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                rr = pd.DataFrame(round_results)
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(rr["round"], rr["global_macro_f1"], marker="o", label="Global Macro F1")
                ax.set_xlabel("Communication Round")
                ax.set_ylabel("Macro F1")
                ax.set_title("Split Learning — Global Macro F1 per Round")
                ax.legend()
                plt.tight_layout()
                fig_path = REPORTS_DIR / "figures" / "fedlora_round_f1.png"
                fig.savefig(fig_path, dpi=150)
                plt.close(fig)
                print(f"Round F1 plot saved -> {fig_path.name}")
            except Exception as e:
                print(f"  Plot skipped: {e}")
        else:
            # ── E4: FedLoRA ────────────────────────────────────────────────────────
            print(f"\n[E4] FedLoRA: {args.rounds} rounds, {args.local_epochs} local epoch(s) each")

            best_val_f1 = 0.0
            best_round  = 0
            best_adapter_dir = MODELS_DIR / "fedlora" / f"global_adapter_{setting_name}_best"
            best_adapter_dir.mkdir(parents=True, exist_ok=True)

            for rnd in range(1, args.rounds + 1):
                print(f"\n--- Round {rnd}/{args.rounds} ---")

                global_state = get_peft_model_state_dict(global_model)
                client_states = []
                client_sizes  = []

                for cid in active_clients:
                    client_model = make_base_model().to(device)
                    set_peft_model_state_dict(client_model, copy.deepcopy(global_state))

                    loader = make_loader(clients[cid]["X"], clients[cid]["y"], tokenizer)
                    client_model = local_train(
                        client_model, loader, device,
                        clients[cid]["cw"], args.local_epochs, args.lr
                    )

                    client_state = get_peft_model_state_dict(client_model)
                    client_states.append(client_state)
                    client_sizes.append(clients[cid]["n"])

                    y_v, y_p = evaluate(client_model, val_loader, device)
                    m = compute_metrics(y_v, y_p)
                    print(f"  {cid}: val macro_f1={m['macro_f1']:.4f}")

                    del client_model

                smishing_sizes = [
                    int((clients[cid]["y"] == LABEL2ID["smishing"]).sum())
                    for cid in active_clients
                ]
                global_model = fedavg_lora(global_model, client_states, client_sizes,
                                           smishing_sizes=smishing_sizes,
                                           agg_weight=args.agg_weight)

                y_v, y_p = evaluate(global_model, val_loader, device)
                global_metrics = compute_metrics(y_v, y_p)
                val_f1 = global_metrics["macro_f1"]
                print(f"  GLOBAL val macro_f1={val_f1:.4f}  "
                      f"smishing_fnr={global_metrics['smishing_fnr']}")

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_round  = rnd
                    global_model.save_pretrained(str(best_adapter_dir))
                    tokenizer.save_pretrained(str(best_adapter_dir))
                    print(f"  [best checkpoint saved — R{rnd} val F1={val_f1:.4f}]")

                row = {
                    "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "round":          rnd,
                    "global_macro_f1": global_metrics["macro_f1"],
                    "global_weighted_f1": global_metrics["weighted_f1"],
                    "smishing_fnr":   global_metrics["smishing_fnr"],
                    "smishing_fpr":   global_metrics["smishing_fpr"],
                }
                round_results.append(row)
                append_result(REPORTS_DIR / "results_fedlora_rounds.csv", row)

            print("\n[E4] Final test evaluation (global model)...")
            y_t, y_p = evaluate(global_model, test_loader, device)
            test_metrics = compute_metrics(y_t, y_p)
            report_metrics(
                test_metrics, experiment, "fedlora_final",
                y_t, y_p,
                extra={
                    "rounds":          args.rounds,
                    "local_epochs":    args.local_epochs,
                    "lr":              args.lr,
                    "n_clients":       len(active_clients),
                    "trainable_params": trainable,
                    "comm_bytes_est":  trainable * 4 * len(active_clients) * args.rounds,
                }
            )

            global_model.save_pretrained(str(global_adapter_dir))
            tokenizer.save_pretrained(str(global_adapter_dir))
            print(f"Global adapter saved -> {global_adapter_dir}")
            print(f"Best adapter (R{best_round}, val F1={best_val_f1:.4f}) saved -> {best_adapter_dir}")

            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                rr = pd.DataFrame(round_results)
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(rr["round"], rr["global_macro_f1"], marker="o", label="Global Macro F1")
                ax.set_xlabel("Communication Round")
                ax.set_ylabel("Macro F1")
                ax.set_title("FedLoRA — Global Macro F1 per Round")
                ax.legend()
                plt.tight_layout()
                fig_path = REPORTS_DIR / "figures" / "fedlora_round_f1.png"
                fig.savefig(fig_path, dpi=150)
                plt.close(fig)
                print(f"Round F1 plot saved -> {fig_path.name}")
            except Exception as e:
                print(f"  Plot skipped: {e}")

    print("\nFederated training complete.")


if __name__ == "__main__":
    main()
