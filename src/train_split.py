"""
train_split.py — Split Learning Training (FedSmishGuard)

E3: Split local-only  — each client trains its own client+server halves independently  (--local flag)
E4: Split FedAvg      — clients train locally, both halves are FedAvg-aggregated each round

Model partition (default: split_layer=3 out of 6 DistilBERT layers):
  Client side  → embeddings + transformer layers [0, split_layer)
  Server side  → transformer layers [split_layer, 6) + pre_classifier + classifier

During forward pass the client computes intermediate hidden states and sends them to the
server, which completes the forward pass and computes the loss. Gradients flow back
through the server to the client (simulated on one machine).

Usage:
  python src/train_split.py                                          # Split FedAvg, defaults
  python src/train_split.py --local                                  # local-only (E3)
  python src/train_split.py --split_layer 2 --rounds 10 --lr 2e-4   # custom layer split

Key arguments:
  --split_layer   Number of DistilBERT layers kept on the client (default: 3)
  --rounds        Number of communication rounds (default: 10)
  --local_epochs  Local training epochs per client per round (default: 2)
  --clients_dir   Path to client CSV folder (default: data/clients)
  --agg_weight    Aggregation: smishing (best) | total | sqrt | balanced | uniform
  --resume        Resume from saved split checkpoint

For LoRA-based federated learning, use train_fedlora.py instead.
"""

import sys
import copy
import argparse
from pathlib import Path
from datetime import datetime

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

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_split, encode_labels,
    LABEL2ID, ID2LABEL, NUM_LABELS,
    MODELS_DIR, REPORTS_DIR, DATA_CLIENTS,
    set_seed, get_device, append_result,
)
from evaluate import compute_metrics, report_metrics
from shared.dataset import make_loader
from shared.fedavg import fedavg_state_dict

set_seed(42)

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"
LOCAL_BATCH  = 16
LOCAL_EPOCHS = 1
COMM_ROUNDS  = 5
LR           = 2e-4

CLIENT_IDS = ["client_1", "client_2", "client_3", "client_4", "client_5"]


# ── Split model components ─────────────────────────────────────────────────────

def _expand_attention_mask(attention_mask, hidden_states):
    """
    Expand a 2-D tokenizer mask to [batch, 1, seq, seq] for direct DistilBERT layer calls.

    The full DistilBERT model expands the mask internally. In split learning we bypass
    the parent model and call transformer layers directly, so newer Transformers SDPA
    attention requires the mask in 4-D form.
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


def _first_hidden_state(layer_output):
    """DistilBERT layers return either a tensor or a tuple — unwrap either."""
    return layer_output[0] if isinstance(layer_output, (tuple, list)) else layer_output


class SplitDistilBertClient(torch.nn.Module):
    """
    Client-side slice of DistilBERT.
    Holds the embedding layer + the first `split_layer` transformer blocks.
    Outputs intermediate hidden states that are sent to the server.
    """

    def __init__(self, base_model, split_layer: int):
        super().__init__()
        self.embeddings = copy.deepcopy(base_model.distilbert.embeddings)
        self.transformer = torch.nn.ModuleList(
            [copy.deepcopy(layer)
             for layer in base_model.distilbert.transformer.layer[:split_layer]]
        )

    def forward(self, input_ids, attention_mask):
        hidden = self.embeddings(input_ids)
        attention_mask = _expand_attention_mask(attention_mask, hidden)
        for layer in self.transformer:
            hidden = _first_hidden_state(layer(hidden, attention_mask=attention_mask))
        return hidden


class SplitDistilBertServer(torch.nn.Module):
    """
    Server-side slice of DistilBERT.
    Holds transformer blocks from `split_layer` onwards + the classification head.
    Receives intermediate hidden states from the client and produces logits.
    """

    def __init__(self, base_model, split_layer: int):
        super().__init__()
        self.transformer = torch.nn.ModuleList(
            [copy.deepcopy(layer)
             for layer in base_model.distilbert.transformer.layer[split_layer:]]
        )
        self.pre_classifier = copy.deepcopy(base_model.pre_classifier)
        self.dropout        = copy.deepcopy(base_model.dropout)
        self.classifier     = copy.deepcopy(base_model.classifier)

    def forward(self, hidden, attention_mask):
        attention_mask = _expand_attention_mask(attention_mask, hidden)
        for layer in self.transformer:
            hidden = _first_hidden_state(layer(hidden, attention_mask=attention_mask))
        hidden = self.pre_classifier(hidden[:, 0])
        hidden = self.dropout(hidden)
        return self.classifier(hidden)


# ── Model factory & checkpoint helpers ────────────────────────────────────────

def make_split_models(split_layer: int):
    """Return a freshly initialised (client, server) pair split at `split_layer`."""
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels = NUM_LABELS,
        id2label   = ID2LABEL,
        label2id   = LABEL2ID,
        ignore_mismatched_sizes = True,
    )
    return SplitDistilBertClient(base, split_layer), SplitDistilBertServer(base, split_layer)


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


# ── Local training & evaluation ────────────────────────────────────────────────

def local_split_train(client_model, server_model, loader, device, class_weights, n_epochs, lr):
    """Train the client+server pair end-to-end for n_epochs."""
    loss_fn   = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(
        list(client_model.parameters()) + list(server_model.parameters()), lr=lr
    )
    total_steps  = len(loader) * n_epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    client_model.train()
    server_model.train()

    for _ in range(n_epochs):
        for batch in loader:
            optimizer.zero_grad()
            hidden = client_model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device).bool(),
            )
            logits = server_model(hidden, attention_mask=batch["attention_mask"].to(device).bool())
            loss   = loss_fn(logits, batch["labels"].to(device))
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
            input_ids      = batch["input_ids"].to(device),
            attention_mask = batch["attention_mask"].to(device).bool(),
        )
        logits = server_model(hidden, attention_mask=batch["attention_mask"].to(device).bool())
        preds.extend(logits.argmax(dim=-1).cpu().numpy())
        trues.extend(batch["labels"].numpy())

    return np.array(trues), np.array(preds)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Split learning training for smishing detection. "
                    "For LoRA-based federated learning use train_fedlora.py."
    )
    parser.add_argument("--local",        action="store_true",
                        help="Local-only mode (E3): each client trains independently, no aggregation")
    parser.add_argument("--split_layer",  type=int, default=3,
                        help="Number of DistilBERT layers to keep on the client side (default: 3)")
    parser.add_argument("--rounds",       type=int,   default=COMM_ROUNDS)
    parser.add_argument("--local_epochs", type=int,   default=LOCAL_EPOCHS)
    parser.add_argument("--lr",           type=float, default=LR)
    parser.add_argument("--resume",       action="store_true",
                        help="Resume from saved split checkpoint")
    parser.add_argument("--clients_dir",  type=str,   default=None,
                        help="Path to client CSVs (default: data/clients/). Relative to project root.")
    parser.add_argument("--agg_weight",   type=str,   default="smishing",
                        choices=["smishing", "sqrt", "uniform", "total", "balanced"],
                        help="FedAvg weighting: smishing|sqrt|uniform|total|balanced")
    args = parser.parse_args()

    from utils import PROJECT_ROOT
    clients_dir  = (PROJECT_ROOT / args.clients_dir) if args.clients_dir else DATA_CLIENTS
    setting_name = clients_dir.name if args.clients_dir else "default"
    agg_tag      = args.agg_weight if not args.local else "none"

    mode       = "split_local" if args.local else "split_fed"
    experiment = ("Split Learning Local (E3)" if args.local
                  else f"Split Learning FedAvg (E4) [{setting_name}]")
    print(
        f"Mode: {mode} | Rounds: {args.rounds} | Local epochs: {args.local_epochs} | "
        f"Split layer: {args.split_layer} | Setting: {setting_name} | Agg: {agg_tag}"
    )

    device = get_device()
    if device is None or str(device) == "cpu":
        raise RuntimeError(
            "GPU (CUDA) not available. Split learning requires a CUDA-capable GPU. "
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

    # ── Build split models ────────────────────────────────────────────────────
    split_model_dir = MODELS_DIR / "split" / setting_name
    split_model_dir.mkdir(parents=True, exist_ok=True)

    print("\nInitialising split learning models...")
    if args.resume and (split_model_dir / "client_state.pt").exists():
        print(f"  Resuming from {split_model_dir}")
        global_client_model, global_server_model = load_split_checkpoint(
            split_model_dir, args.split_layer
        )
    else:
        global_client_model, global_server_model = make_split_models(args.split_layer)

    global_client_model = global_client_model.to(device)
    global_server_model = global_server_model.to(device)

    total     = (sum(p.numel() for p in global_client_model.parameters())
                 + sum(p.numel() for p in global_server_model.parameters()))
    trainable = (sum(p.numel() for p in global_client_model.parameters() if p.requires_grad)
                 + sum(p.numel() for p in global_server_model.parameters() if p.requires_grad))
    print(f"Split model params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    round_results = []

    if args.local:
        # ── E3: Split local-only ───────────────────────────────────────────────
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
                "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M"),
                "experiment":    "Split Learning Local (E3)",
                "avg_macro_f1":  round(all_macro_f1, 4),
                **{f"{cid}_macro_f1": round(m["macro_f1"], 4)
                   for cid, m in client_metrics.items()},
                "trainable_params": trainable,
            }
        )

    else:
        # ── E4: Split FedAvg ───────────────────────────────────────────────────
        print(f"\n[E4] Split Learning FedAvg: {args.rounds} rounds, "
              f"{args.local_epochs} local epoch(s) each, split_layer={args.split_layer}")

        best_val_f1  = 0.0
        best_round   = 0
        best_split_dir = MODELS_DIR / "split" / f"{setting_name}_best"
        best_split_dir.mkdir(parents=True, exist_ok=True)

        for rnd in range(1, args.rounds + 1):
            print(f"\n--- Round {rnd}/{args.rounds} ---")
            client_states = []
            server_states = []
            client_sizes  = []

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

                del client_model, server_model

            smishing_sizes = [
                int((clients[cid]["y"] == LABEL2ID["smishing"]).sum())
                for cid in active_clients
            ]
            global_client_model = fedavg_state_dict(
                global_client_model, client_states, client_sizes,
                smishing_sizes=smishing_sizes, agg_weight=args.agg_weight,
            )
            global_server_model = fedavg_state_dict(
                global_server_model, server_states, client_sizes,
                smishing_sizes=smishing_sizes, agg_weight=args.agg_weight,
            )

            y_v, y_p = evaluate_split(global_client_model, global_server_model, val_loader, device)
            global_metrics = compute_metrics(y_v, y_p)
            val_f1 = global_metrics["macro_f1"]
            print(f"  GLOBAL val macro_f1={val_f1:.4f}  "
                  f"smishing_fnr={global_metrics['smishing_fnr']}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_round  = rnd
                save_split_checkpoint(best_split_dir, global_client_model, global_server_model, tokenizer)
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

        print("\n[E4] Final test evaluation (global split model)...")
        y_t, y_p = evaluate_split(global_client_model, global_server_model, test_loader, device)
        test_metrics = compute_metrics(y_t, y_p)
        report_metrics(
            test_metrics, experiment, "split_final",
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
            fig_path = REPORTS_DIR / "figures" / "split_round_f1.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            print(f"Round F1 plot saved -> {fig_path.name}")
        except Exception as e:
            print(f"  Plot skipped: {e}")

    print("\nSplit learning training complete.")


if __name__ == "__main__":
    main()
