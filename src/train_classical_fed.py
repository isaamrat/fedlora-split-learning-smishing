"""
train_classical_fed.py — Federated learning (FedAvg) for classical deep learning models.

Supports five model architectures via --model:
  cnn       TextCNN (parallel Conv1d filters)
  lstm      Unidirectional LSTM
  bilstm    Bidirectional LSTM
  gru       Unidirectional GRU
  bigru     Bidirectional GRU
  cnn_lstm  Hybrid CNN-LSTM

Federation uses standard FedAvg over full model weights (fedavg_state_dict from shared/fedavg.py).
LoRA is not applicable here — these models have no transformer attention layers.

Requires:
  data/vocab.json      — build with: python src/shared/build_vocab.py
  data/glove.6B.100d.txt (optional) — for pretrained embeddings

Usage:
  python src/train_classical_fed.py --model gru
  python src/train_classical_fed.py --model bilstm --rounds 10 --local_epochs 2
  python src/train_classical_fed.py --model cnn_lstm --embed_dim 100 --hidden_dim 128
  python src/train_classical_fed.py --model lstm --no_glove   # learned embeddings only
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

# Model factories
from models.textcnn  import make_textcnn
from models.lstm     import make_lstm
from models.gru      import make_gru
from models.cnn_lstm import make_cnn_lstm

set_seed(42)

# ── Config ─────────────────────────────────────────────────────────────────────
MAX_LEN      = 128
LOCAL_BATCH  = 16
LOCAL_EPOCHS = 1
COMM_ROUNDS  = 10
LR           = 1e-3

CLIENT_IDS = ["client_1", "client_2", "client_3", "client_4", "client_5"]

SUPPORTED_MODELS = ["cnn", "lstm", "bilstm", "gru", "bigru", "cnn_lstm"]


# ── Model factory ──────────────────────────────────────────────────────────────

def build_model(args, vocab_size: int, embed_weights=None):
    """Instantiate the selected model architecture."""
    m = args.model.lower()
    kw = dict(
        vocab_size    = vocab_size,
        embed_dim     = args.embed_dim,
        num_classes   = NUM_LABELS,
        embed_weights = embed_weights,
    )
    if m == "cnn":
        return make_textcnn(**kw)
    if m == "lstm":
        return make_lstm(**kw, hidden_dim=args.hidden_dim, bidirectional=False)
    if m == "bilstm":
        return make_lstm(**kw, hidden_dim=args.hidden_dim, bidirectional=True)
    if m == "gru":
        return make_gru(**kw, hidden_dim=args.hidden_dim, bidirectional=False)
    if m == "bigru":
        return make_gru(**kw, hidden_dim=args.hidden_dim, bidirectional=True)
    if m == "cnn_lstm":
        return make_cnn_lstm(**kw, num_filters=args.num_filters, hidden_dim=args.hidden_dim)
    raise ValueError(f"Unknown model: {args.model}. Choose from: {SUPPORTED_MODELS}")


# ── Local training ─────────────────────────────────────────────────────────────

def local_train(model, loader, device, class_weights, n_epochs, lr):
    """Train model for n_epochs. Returns updated model."""
    loss_fn   = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    total_steps  = len(loader) * n_epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.train()
    for _ in range(n_epochs):
        for batch in loader:
            optimizer.zero_grad()
            logits = model(batch["input_ids"].to(device))
            loss   = loss_fn(logits, batch["labels"].to(device))
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
        logits = model(batch["input_ids"].to(device))
        preds.extend(logits.argmax(dim=-1).cpu().numpy())
        trues.extend(batch["labels"].numpy())
    return np.array(trues), np.array(preds)


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_model(model, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / "model_state.pt")


def load_model_weights(model, directory: Path):
    state_path = directory / "model_state.pt"
    if state_path.exists():
        model.load_state_dict(torch.load(state_path, map_location="cpu"))
    return model


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Federated learning (FedAvg) for classical deep learning models. "
                    "For split learning use train_classical_split.py. "
                    "For DistilBERT+LoRA use train_fedlora.py."
    )
    parser.add_argument("--model",        type=str, required=True, choices=SUPPORTED_MODELS,
                        help="Model architecture: cnn | lstm | bilstm | gru | bigru | cnn_lstm")
    parser.add_argument("--local",        action="store_true",
                        help="Local-only mode: each client trains independently, no aggregation")
    parser.add_argument("--rounds",       type=int,   default=COMM_ROUNDS)
    parser.add_argument("--local_epochs", type=int,   default=LOCAL_EPOCHS)
    parser.add_argument("--lr",           type=float, default=LR)
    parser.add_argument("--embed_dim",    type=int,   default=100,
                        help="Embedding dimension (must match GloVe file if using GloVe, default: 100)")
    parser.add_argument("--hidden_dim",   type=int,   default=256,
                        help="Hidden state dimension for LSTM/GRU (default: 256)")
    parser.add_argument("--num_filters",  type=int,   default=128,
                        help="Number of CNN filters for cnn/cnn_lstm (default: 128)")
    parser.add_argument("--no_glove",     action="store_true",
                        help="Skip GloVe loading and use random embeddings")
    parser.add_argument("--clients_dir",  type=str,   default=None,
                        help="Path to client CSVs relative to project root (default: data/clients)")
    parser.add_argument("--agg_weight",   type=str,   default="smishing",
                        choices=["smishing", "sqrt", "uniform", "total", "balanced"])
    parser.add_argument("--resume",       action="store_true",
                        help="Resume from saved checkpoint")
    args = parser.parse_args()

    from utils import PROJECT_ROOT
    clients_dir  = (PROJECT_ROOT / args.clients_dir) if args.clients_dir else DATA_CLIENTS
    setting_name = clients_dir.name if args.clients_dir else "default"
    agg_tag      = args.agg_weight if not args.local else "none"
    model_tag    = f"{args.model}_{setting_name}"

    mode       = "local" if args.local else "fed"
    experiment = (f"{args.model.upper()} Local-Only (E3)"
                  if args.local
                  else f"{args.model.upper()} FedAvg (E4) [{setting_name}]")
    print(
        f"Model: {args.model} | Mode: {mode} | Rounds: {args.rounds} | "
        f"Local epochs: {args.local_epochs} | Setting: {setting_name} | Agg: {agg_tag}"
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

    # ── Validation and test loaders ────────────────────────────────────────────
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
        classes  = np.array(sorted(set(y)))
        cw       = compute_class_weight("balanced", classes=classes, y=y)
        cw_dict  = dict(zip(classes.tolist(), cw.tolist()))
        cw_tensor = torch.tensor([cw_dict.get(i, 1.0) for i in range(NUM_LABELS)], dtype=torch.float32)
        clients[cid] = {"X": X, "y": y, "cw": cw_tensor, "n": len(y)}
        print(f"  {cid}: {len(y)} rows | {dict(pd.Series(df['label']).value_counts())}")

    if not clients:
        print("ERROR: No client data found.")
        sys.exit(1)

    active_clients = list(clients.keys())
    print(f"\nActive clients: {active_clients}")

    # ── Global model ───────────────────────────────────────────────────────────
    print(f"\nInitialising {args.model.upper()} model...")
    global_model = build_model(args, vocab_size, embed_weights).to(device)

    save_dir = MODELS_DIR / "classical" / model_tag
    save_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and (save_dir / "model_state.pt").exists():
        global_model = load_model_weights(global_model, save_dir)
        print(f"  Resumed from {save_dir}")

    total     = sum(p.numel() for p in global_model.parameters())
    trainable = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    print(f"Params: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

    best_dir   = MODELS_DIR / "classical" / f"{model_tag}_best"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = 0.0
    best_round  = 0
    round_results = []

    if args.local:
        # ── E3: Local-only ────────────────────────────────────────────────────
        print(f"\n[E3] {args.model.upper()} local-only training (no aggregation)...")
        client_metrics = {}
        for cid in active_clients:
            print(f"\n  Training {cid}...")
            local_model = build_model(args, vocab_size, embed_weights).to(device)
            loader = make_vocab_loader(clients[cid]["X"], clients[cid]["y"], vocab,
                                       max_len=MAX_LEN, batch_size=LOCAL_BATCH)
            local_model = local_train(local_model, loader, device,
                                      clients[cid]["cw"], args.local_epochs * args.rounds, args.lr)
            y_t, y_p = evaluate(local_model, test_loader, device)
            m = compute_metrics(y_t, y_p)
            client_metrics[cid] = m
            print(f"  {cid} test macro_f1={m['macro_f1']:.4f} smishing_fnr={m['smishing_fnr']}")
            save_model(local_model, MODELS_DIR / "classical" / f"{cid}_{args.model}_local")

        all_f1 = np.mean([m["macro_f1"] for m in client_metrics.values()])
        print(f"\nLocal-only avg macro_f1: {all_f1:.4f}")
        append_result(
            REPORTS_DIR / "results_classical_local.csv",
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "model": args.model, "experiment": experiment,
                "avg_macro_f1": round(all_f1, 4),
                **{f"{cid}_macro_f1": round(m["macro_f1"], 4) for cid, m in client_metrics.items()},
                "trainable_params": trainable,
            }
        )

    else:
        # ── E4: FedAvg ────────────────────────────────────────────────────────
        print(f"\n[E4] {args.model.upper()} FedAvg: {args.rounds} rounds, {args.local_epochs} epoch(s) each")

        for rnd in range(1, args.rounds + 1):
            print(f"\n--- Round {rnd}/{args.rounds} ---")
            global_state  = copy.deepcopy(global_model.state_dict())
            client_states = []
            client_sizes  = []

            for cid in active_clients:
                client_model = build_model(args, vocab_size, embed_weights).to(device)
                client_model.load_state_dict(copy.deepcopy(global_state))
                loader = make_vocab_loader(clients[cid]["X"], clients[cid]["y"], vocab,
                                           max_len=MAX_LEN, batch_size=LOCAL_BATCH)
                client_model = local_train(client_model, loader, device,
                                           clients[cid]["cw"], args.local_epochs, args.lr)

                client_states.append({k: v.cpu() for k, v in client_model.state_dict().items()})
                client_sizes.append(clients[cid]["n"])

                y_v, y_p = evaluate(client_model, val_loader, device)
                m = compute_metrics(y_v, y_p)
                print(f"  {cid}: val macro_f1={m['macro_f1']:.4f}")
                del client_model

            smishing_sizes = [
                int((clients[cid]["y"] == LABEL2ID["smishing"]).sum())
                for cid in active_clients
            ]
            global_model = fedavg_state_dict(
                global_model, client_states, client_sizes,
                smishing_sizes=smishing_sizes, agg_weight=args.agg_weight,
            )

            y_v, y_p = evaluate(global_model, val_loader, device)
            gm = compute_metrics(y_v, y_p)
            val_f1 = gm["macro_f1"]
            print(f"  GLOBAL val macro_f1={val_f1:.4f}  smishing_fnr={gm['smishing_fnr']}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_round  = rnd
                save_model(global_model, best_dir)
                print(f"  [best checkpoint saved — R{rnd} val F1={val_f1:.4f}]")

            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "model": args.model, "round": rnd,
                "global_macro_f1": gm["macro_f1"], "global_weighted_f1": gm["weighted_f1"],
                "smishing_fnr": gm["smishing_fnr"], "smishing_fpr": gm["smishing_fpr"],
            }
            round_results.append(row)
            append_result(REPORTS_DIR / f"results_{args.model}_fed_rounds.csv", row)

        print(f"\n[E4] Final test evaluation ({args.model.upper()})...")
        y_t, y_p = evaluate(global_model, test_loader, device)
        test_metrics = compute_metrics(y_t, y_p)
        report_metrics(
            test_metrics, experiment, f"{args.model}_fed_final",
            y_t, y_p,
            extra={
                "model": args.model, "rounds": args.rounds,
                "local_epochs": args.local_epochs, "lr": args.lr,
                "n_clients": len(active_clients), "trainable_params": trainable,
                "comm_bytes_est": trainable * 4 * len(active_clients) * args.rounds,
            }
        )

        save_model(global_model, save_dir)
        print(f"Final model saved -> {save_dir}")
        print(f"Best model (R{best_round}, val F1={best_val_f1:.4f}) saved -> {best_dir}")

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
            ax.set_title(f"{args.model.upper()} FedAvg — Macro F1 per Round")
            ax.legend()
            plt.tight_layout()
            fig_path = REPORTS_DIR / "figures" / f"{args.model}_fed_round_f1.png"
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            print(f"Round F1 plot saved -> {fig_path.name}")
        except Exception as e:
            print(f"  Plot skipped: {e}")

    print(f"\n{args.model.upper()} federated training complete.")


if __name__ == "__main__":
    main()
