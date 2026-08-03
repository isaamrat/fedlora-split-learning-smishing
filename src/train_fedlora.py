"""
train_fedlora.py — Federated LoRA Training (FedSmishGuard)

E3: Local-only training  — each client trains independently, no server  (--local flag)
E4: FedLoRA              — clients train locally, server aggregates LoRA adapters

Raw data is never shared. Only LoRA adapter weights (~2.96 MB per client) are exchanged.

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

For split learning, use train_split.py instead.
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
from shared.dataset import make_loader
from shared.fedavg import fedavg_lora

set_seed(42)

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"
MAX_LEN      = 128
LOCAL_BATCH  = 16
LOCAL_EPOCHS = 1
COMM_ROUNDS  = 5
LR           = 2e-4

CLIENT_IDS = ["client_1", "client_2", "client_3", "client_4", "client_5"]

LORA_CONFIG = dict(
    r              = 8,
    lora_alpha     = 16,
    lora_dropout   = 0.1,
    bias           = "none",
    task_type      = TaskType.SEQ_CLS,
    target_modules = ["q_lin", "v_lin"],
)


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


# ── Model factory ──────────────────────────────────────────────────────────────

def make_base_model():
    """Return a fresh DistilBERT wrapped with LoRA adapters."""
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels = NUM_LABELS,
        id2label   = ID2LABEL,
        label2id   = LABEL2ID,
        ignore_mismatched_sizes = True,
    )
    lora_cfg = LoraConfig(**LORA_CONFIG)
    return get_peft_model(base, lora_cfg)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Federated LoRA training for smishing detection. "
                    "For split learning use train_split.py."
    )
    parser.add_argument("--local",        action="store_true",
                        help="Local-only mode (E3): train each client independently, no aggregation")
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

    mode       = "local_only" if args.local else "fedlora"
    experiment = ("Local-Only Client Training (E3)" if args.local
                  else f"FedLoRA FedAvg (E4) [{setting_name}]")
    print(
        f"Mode: {mode} | Rounds: {args.rounds} | Local epochs: {args.local_epochs} | "
        f"Setting: {setting_name} | Agg: {agg_tag}"
    )

    device = get_device()
    if device is None or str(device) == "cpu":
        raise RuntimeError(
            "GPU (CUDA) not available. FedLoRA requires a CUDA-capable GPU. "
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
            classes = np.array(sorted(set(y)))
            cw = compute_class_weight("balanced", classes=classes, y=y)
            cw_dict = dict(zip(classes.tolist(), cw.tolist()))
            cw_tensor = torch.tensor(
                [cw_dict.get(i, 1.0) for i in range(NUM_LABELS)], dtype=torch.float32
            )
            clients[cid] = {"X": X, "y": y, "cw": cw_tensor, "n": len(y)}
            label_dist = dict(pd.Series(df["label"]).value_counts())
            print(f"  {cid}: {len(y)} rows | {label_dist}")
        except FileNotFoundError:
            print(f"  WARNING: {cid}.csv not found — skipping")

    if not clients:
        print("ERROR: No client data found. Check --clients_dir points to a folder with client_*.csv files.")
        sys.exit(1)

    active_clients = list(clients.keys())
    print(f"\nActive clients: {active_clients}")

    # ── Build global model ────────────────────────────────────────────────────
    print("\nInitialising global LoRA model...")
    global_adapter_dir = MODELS_DIR / "fedlora" / f"global_adapter_{setting_name}"
    global_adapter_dir.mkdir(parents=True, exist_ok=True)

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
                "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M"),
                "experiment":    "Local-Only (E3)",
                "avg_macro_f1":  round(all_macro_f1, 4),
                **{f"{cid}_macro_f1": round(m["macro_f1"], 4)
                   for cid, m in client_metrics.items()},
                "trainable_params": trainable,
            }
        )

    else:
        # ── E4: FedLoRA ────────────────────────────────────────────────────────
        print(f"\n[E4] FedLoRA: {args.rounds} rounds, {args.local_epochs} local epoch(s) each")

        best_val_f1  = 0.0
        best_round   = 0
        best_adapter_dir = MODELS_DIR / "fedlora" / f"global_adapter_{setting_name}_best"
        best_adapter_dir.mkdir(parents=True, exist_ok=True)

        for rnd in range(1, args.rounds + 1):
            print(f"\n--- Round {rnd}/{args.rounds} ---")

            global_state  = get_peft_model_state_dict(global_model)
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
                "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M"),
                "round":              rnd,
                "global_macro_f1":    global_metrics["macro_f1"],
                "global_weighted_f1": global_metrics["weighted_f1"],
                "smishing_fnr":       global_metrics["smishing_fnr"],
                "smishing_fpr":       global_metrics["smishing_fpr"],
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
                "rounds":           args.rounds,
                "local_epochs":     args.local_epochs,
                "lr":               args.lr,
                "n_clients":        len(active_clients),
                "trainable_params": trainable,
                "comm_bytes_est":   trainable * 4 * len(active_clients) * args.rounds,
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

    print("\nFederated LoRA training complete.")


if __name__ == "__main__":
    main()
