"""
train_classical_split.py — Split learning for classical deep learning models.

Supports five model architectures via --model, each with defined split points:

  cnn       Split A: after embedding (only split point)
  lstm      Split A: after embedding
  bilstm    Split A: after embedding
  gru       Split A: after embedding
  bigru     Split A: after embedding
  cnn_lstm  Split A: after embedding  |  Split B: after CNN (recommended ⭐)

In split learning:
  - Client holds the first portion of the model and sends intermediate hidden states
  - Server holds the remaining layers and returns logits
  - Gradients flow end-to-end (simulated on one machine)
  - Both halves are FedAvg-aggregated each round using fedavg_state_dict

Requires:
  data/vocab.json      — build with: python src/shared/build_vocab.py
  data/glove.6B.100d.txt (optional)

Usage:
  python src/train_classical_split.py --model gru
  python src/train_classical_split.py --model cnn_lstm --split_point B
  python src/train_classical_split.py --model bilstm --rounds 10 --local_epochs 2
  python src/train_classical_split.py --model lstm --local   # local-only, no federation
"""

import sys
import copy
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_split, encode_labels,
    LABEL2ID, ID2LABEL, NUM_LABELS,
    MODELS_DIR, REPORTS_DIR, DATA_CLIENTS,
    set_seed, get_device, append_result,
)
from evaluate import compute_metrics, report_metrics
from shared.vocab_dataset import load_vocab, build_embedding_matrix, make_vocab_loader
from shared.fedavg import fedavg_state_dict

# Model split factories
from models.textcnn  import make_textcnn_split
from models.lstm     import make_lstm_split
from models.gru      import make_gru_split
from models.cnn_lstm import make_cnn_lstm_split

set_seed(42)

# ── Config ─────────────────────────────────────────────────────────────────────
MAX_LEN      = 128
LOCAL_BATCH  = 16
LOCAL_EPOCHS = 1
COMM_ROUNDS  = 10
LR           = 1e-3

CLIENT_IDS = ["client_1", "client_2", "client_3", "client_4", "client_5"]

SUPPORTED_MODELS = ["cnn", "lstm", "bilstm", "gru", "bigru", "cnn_lstm"]

# Split point A is valid for all models; B is only for cnn_lstm
VALID_SPLIT_POINTS = {
    "cnn":      ["A"],
    "lstm":     ["A"],
    "bilstm":   ["A"],
    "gru":      ["A"],
    "bigru":    ["A"],
    "cnn_lstm": ["A", "B"],
}


# ── Split model factory ────────────────────────────────────────────────────────

def build_split_models(args, vocab_size: int, embed_weights=None):
    """Return (client, server) pair for the selected architecture and split point."""
    m  = args.model.lower()
    sp = args.split_point.upper()

    valid = VALID_SPLIT_POINTS.get(m, ["A"])
    if sp not in valid:
        print(f"WARNING: split_point '{sp}' not valid for {m}. Valid options: {valid}. Defaulting to A.")
        sp = "A"

    kw = dict(vocab_size=vocab_size, embed_dim=args.embed_dim, embed_weights=embed_weights)

    if m == "cnn":
        return make_textcnn_split(**kw, num_classes=NUM_LABELS)
    if m == "lstm":
        return make_lstm_split(**kw, hidden_dim=args.hidden_dim, num_classes=NUM_LABELS, bidirectional=False)
    if m == "bilstm":
        return make_lstm_split(**kw, hidden_dim=args.hidden_dim, num_classes=NUM_LABELS, bidirectional=True)
    if m == "gru":
        return make_gru_split(**kw, hidden_dim=args.hidden_dim, num_classes=NUM_LABELS, bidirectional=False)
    if m == "bigru":
        return make_gru_split(**kw, hidden_dim=args.hidden_dim, num_classes=NUM_LABELS, bidirectional=True)
    if m == "cnn_lstm":
        return make_cnn_lstm_split(**kw, split_point=sp,
                                   num_filters=args.num_filters,
                                   hidden_dim=args.hidden_dim,
                                   num_classes=NUM_LABELS)
    raise ValueError(f"Unknown model: {args.model}")


# ── Local split training ───────────────────────────────────────────────────────

def local_split_train(client_model, server_model, loader, device, class_weights, n_epochs, lr):
    """Train client+server end-to-end. Gradients flow from server back through client."""
    loss_fn   = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = Adam(
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
            hidden = client_model(batch["input_ids"].to(device))
            logits = server_model(hidden)
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
        hidden = client_model(batch["input_ids"].to(device))
        logits = server_model(hidden)
        preds.extend(logits.argmax(dim=-1).cpu().numpy())
        trues.extend(batch["labels"].numpy())
    return np.array(trues), np.array(preds)


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_split_checkpoint(directory: Path, client_model, server_model):
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(client_model.state_dict(), directory / "client_state.pt")
    torch.save(server_model.state_dict(), directory / "server_state.pt")


def load_split_checkpoint(directory: Path, args, vocab_size, embed_weights):
    client_model, server_model = build_split_models(args, vocab_size, embed_weights)
    cp = directory / "client_state.pt"
    sp = directory / "server_state.pt"
    if cp.exists() and sp.exists():
        client_model.load_state_dict(torch.load(cp, map_location="cpu"))
        server_model.load_state_dict(torch.load(sp, map_location="cpu"))
    return client_model, server_model


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Split learning for classical deep learning models. "
                    "For FedAvg use train_classical_fed.py. "
                    "For DistilBERT split use train_split.py."
    )
    parser.add_argument("--model",        type=str, required=True, choices=SUPPORTED_MODELS,
                        help="Model architecture: cnn | lstm | bilstm | gru | bigru | cnn_lstm")
    parser.add_argument("--split_point",  type=str, default="A", choices=["A", "B"],
                        help="Split point: A=after embedding, B=after CNN (cnn_lstm only, default: A)")
    parser.add_argument("--local",        action="store_true",
                        help="Local-only mode: each client trains independently, no aggregation")
    parser.add_argument("--rounds",       type=int,   default=COMM_ROUNDS)
    parser.add_argument("--local_epochs", type=int,   default=LOCAL_EPOCHS)
    parser.add_argument("--lr",           type=float, default=LR)
    parser.add_argument("--embed_dim",    type=int,   default=100)
    parser.add_argument("--hidden_dim",   type=int,   default=256,
                        help="Hidden state dimension for LSTM/GRU (default: 256)")
    parser.add_argument("--num_filters",  type=int,   default=128,
                        help="CNN filters for cnn_lstm (default: 128)")
    parser.add_argument("--no_glove",     action="store_true",
                        help="Use random embeddings instead of GloVe")
    parser.add_argument("--clients_dir",  type=str,   default=None)
    parser.add_argument("--agg_weight",   type=str,   default="smishing",
                        choices=["smishing", "sqrt", "uniform", "total", "balanced"])
    parser.add_argument("--resume",       action="store_true")
    args = parser.parse_args()

    from utils import PROJECT_ROOT
    clients_dir  = (PROJECT_ROOT / args.clients_dir) if args.clients_dir else DATA_CLIENTS
    setting_name = clients_dir.name if args.clients_dir else "default"
    agg_tag      = args.agg_weight if not args.local else "none"
    sp_label     = args.split_point.upper() if args.model == "cnn_lstm" else "A"
    model_tag    = f"{args.model}_split{sp_label}_{setting_name}"

    mode       = "split_local" if args.local else "split_fed"
    experiment = (f"{args.model.upper()} Split-{sp_label} Local (E3)"
                  if args.local
                  else f"{args.model.upper()} Split-{sp_label} FedAvg (E4) [{setting_name}]")
    print(
        f"Model: {args.model} | Split: {sp_label} | Mode: {mode} | "
        f"Rounds: {args.rounds} | Local epochs: {args.local_epochs} | Agg: {agg_tag}"
    )

    device = get_device()
    print(f"Device: {device}")

    # ── Vocabulary & embeddings ────────────────────────────────────────────────
    print("\nLoading vocabulary...")
    vocab = load_vocab()
    vocab_size = len(vocab)
    print(f"Vocab size: {vocab_size:,}")

    embed_weights = None
    if not args.no_glove:
        try:
            embed_weights = build_embedding_matrix(vocab, embed_dim=args.embed_dim)
        except FileNotFoundError as e:
            print(f"  GloVe not found — using random embeddings. ({e})")

    # ── Data loaders ──────────────────────────────────────────────────────────
    val_df  = load_split("val")
    test_df = load_split("test_clean")
    val_loader  = make_vocab_loader(val_df["cleaned_text"].fillna("").values,
                                    encode_labels(val_df["label"]), vocab,
                                    max_len=MAX_LEN, shuffle=False)
    test_loader = make_vocab_loader(test_df["cleaned_text"].fillna("").values,
                                    encode_labels(test_df["label"]), vocab,
                                    max_len=MAX_LEN, shuffle=False)

    # ── Client data ────────────────────────────────────────────────────────────
    clients = {}
    for cid in CLIENT_IDS:
        csv_path = clients_dir / f"{cid}.csv"
        if not csv_path.exists():
            print(f"  WARNING: {cid}.csv not found — skipping")
            continue
        df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
        if len(df) < 10:
            continue
        X = df["cleaned_text"].fillna("").values
        y = encode_labels(df["label"])
        classes   = np.array(sorted(set(y)))
        cw        = compute_class_weight("balanced", classes=classes, y=y)
        cw_dict   = dict(zip(classes.tolist(), cw.tolist()))
        cw_tensor = torch.tensor([cw_dict.get(i, 1.0) for i in range(NUM_LABELS)], dtype=torch.float32)
        clients[cid] = {"X": X, "y": y, "cw": cw_tensor, "n": len(y)}
        print(f"  {cid}: {len(y)} rows | {dict(pd.Series(df['label']).value_counts())}")

    if not clients:
        print("ERROR: No client data found.")
        sys.exit(1)

    active_clients = list(clients.keys())
    print(f"\nActive clients: {active_clients}")

    # ── Build global split models ──────────────────────────────────────────────
    print(f"\nInitialising {args.model.upper()} split models (split point {sp_label})...")
    split_dir = MODELS_DIR / "classical_split" / model_tag
    split_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and (split_dir / "client_state.pt").exists():
        print(f"  Resuming from {split_dir}")
        global_client, global_server = load_split_checkpoint(split_dir, args, vocab_size, embed_weights)
    else:
        global_client, global_server = build_split_models(args, vocab_size, embed_weights)

    global_client = global_client.to(device)
    global_server = global_server.to(device)

    total     = (sum(p.numel() for p in global_client.parameters())
                 + sum(p.numel() for p in global_server.parameters()))
    trainable = (sum(p.numel() for p in global_client.parameters() if p.requires_grad)
                 + sum(p.numel() for p in global_server.parameters() if p.requires_grad))
    print(f"Split model params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    best_dir   = MODELS_DIR / "classical_split" / f"{model_tag}_best"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = 0.0
    best_round  = 0
    round_results = []

    if args.local:
        # ── E3: Split local-only ───────────────────────────────────────────────
        print(f"\n[E3] {args.model.upper()} split local-only (no aggregation)...")
        client_metrics = {}
        for cid in active_clients:
            print(f"\n  Training {cid}...")
            c_model, s_model = build_split_models(args, vocab_size, embed_weights)
            c_model.load_state_dict(global_client.state_dict())
            s_model.load_state_dict(global_server.state_dict())
            c_model, s_model = c_model.to(device), s_model.to(device)

            loader = make_vocab_loader(clients[cid]["X"], clients[cid]["y"], vocab,
                                       max_len=MAX_LEN, batch_size=LOCAL_BATCH)
            c_model, s_model = local_split_train(c_model, s_model, loader, device,
                                                  clients[cid]["cw"],
                                                  args.local_epochs * args.rounds, args.lr)
            y_t, y_p = evaluate_split(c_model, s_model, test_loader, device)
            m = compute_metrics(y_t, y_p)
            client_metrics[cid] = m
            print(f"  {cid} test macro_f1={m['macro_f1']:.4f} smishing_fnr={m['smishing_fnr']}")
            save_split_checkpoint(MODELS_DIR / "classical_split" / f"{cid}_{args.model}_split{sp_label}_local",
                                   c_model, s_model)

        all_f1 = np.mean([m["macro_f1"] for m in client_metrics.values()])
        print(f"\nLocal-only split avg macro_f1: {all_f1:.4f}")
        append_result(
            REPORTS_DIR / "results_classical_split_local.csv",
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "model": args.model, "split_point": sp_label, "experiment": experiment,
                "avg_macro_f1": round(all_f1, 4),
                **{f"{cid}_macro_f1": round(m["macro_f1"], 4) for cid, m in client_metrics.items()},
                "trainable_params": trainable,
            }
        )

    else:
        # ── E4: Split FedAvg ───────────────────────────────────────────────────
        print(f"\n[E4] {args.model.upper()} Split-{sp_label} FedAvg: "
              f"{args.rounds} rounds, {args.local_epochs} epoch(s) each")

        for rnd in range(1, args.rounds + 1):
            print(f"\n--- Round {rnd}/{args.rounds} ---")
            client_states = []
            server_states = []
            client_sizes  = []

            for cid in active_clients:
                c_model, s_model = build_split_models(args, vocab_size, embed_weights)
                c_model.load_state_dict(copy.deepcopy(global_client.state_dict()))
                s_model.load_state_dict(copy.deepcopy(global_server.state_dict()))
                c_model, s_model = c_model.to(device), s_model.to(device)

                loader = make_vocab_loader(clients[cid]["X"], clients[cid]["y"], vocab,
                                           max_len=MAX_LEN, batch_size=LOCAL_BATCH)
                c_model, s_model = local_split_train(c_model, s_model, loader, device,
                                                      clients[cid]["cw"], args.local_epochs, args.lr)

                client_states.append({k: v.cpu() for k, v in c_model.state_dict().items()})
                server_states.append({k: v.cpu() for k, v in s_model.state_dict().items()})
                client_sizes.append(clients[cid]["n"])

                y_v, y_p = evaluate_split(c_model, s_model, val_loader, device)
                m = compute_metrics(y_v, y_p)
                print(f"  {cid}: val macro_f1={m['macro_f1']:.4f}")
                del c_model, s_model

            smishing_sizes = [
                int((clients[cid]["y"] == LABEL2ID["smishing"]).sum())
                for cid in active_clients
            ]
            global_client = fedavg_state_dict(global_client, client_states, client_sizes,
                                               smishing_sizes=smishing_sizes, agg_weight=args.agg_weight)
            global_server = fedavg_state_dict(global_server, server_states, client_sizes,
                                               smishing_sizes=smishing_sizes, agg_weight=args.agg_weight)

            y_v, y_p = evaluate_split(global_client, global_server, val_loader, device)
            gm = compute_metrics(y_v, y_p)
            val_f1 = gm["macro_f1"]
            print(f"  GLOBAL val macro_f1={val_f1:.4f}  smishing_fnr={gm['smishing_fnr']}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_round  = rnd
                save_split_checkpoint(best_dir, global_client, global_server)
                print(f"  [best checkpoint saved — R{rnd} val F1={val_f1:.4f}]")

            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "model": args.model, "split_point": sp_label, "round": rnd,
                "global_macro_f1": gm["macro_f1"], "global_weighted_f1": gm["weighted_f1"],
                "smishing_fnr": gm["smishing_fnr"], "smishing_fpr": gm["smishing_fpr"],
            }
            round_results.append(row)
            append_result(REPORTS_DIR / f"results_{args.model}_split{sp_label}_rounds.csv", row)

        print(f"\n[E4] Final test evaluation ({args.model.upper()} split-{sp_label})...")
        y_t, y_p = evaluate_split(global_client, global_server, test_loader, device)
        test_metrics = compute_metrics(y_t, y_p)
        report_metrics(
            test_metrics, experiment, f"{args.model}_split{sp_label}_final",
            y_t, y_p,
            extra={
                "model": args.model, "split_point": sp_label,
                "rounds": args.rounds, "local_epochs": args.local_epochs,
                "n_clients": len(active_clients), "trainable_params": trainable,
                "comm_bytes_est": trainable * 4 * len(active_clients) * args.rounds,
            }
        )

        save_split_checkpoint(split_dir, global_client, global_server)
        print(f"Final split model saved -> {split_dir}")
        print(f"Best split model (R{best_round}, val F1={best_val_f1:.4f}) saved -> {best_dir}")

        # Round F1 plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            rr = pd.DataFrame(round_results)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(rr["round"], rr["global_macro_f1"], marker="o", label="Global Macro F1")
            ax.set_xlabel("Communication Round")
            ax.set_ylabel("Macro F1")
            ax.set_title(f"{args.model.upper()} Split-{sp_label} FedAvg — Macro F1 per Round")
            ax.legend()
            plt.tight_layout()
            fig_path = REPORTS_DIR / "figures" / f"{args.model}_split{sp_label}_round_f1.png"
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            print(f"Round F1 plot saved -> {fig_path.name}")
        except Exception as e:
            print(f"  Plot skipped: {e}")

    print(f"\n{args.model.upper()} split learning complete.")


if __name__ == "__main__":
    main()
