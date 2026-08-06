"""
train_split.py — Split Learning Training (FedSmishGuard)

Improvements over baseline:
  ★ Priority 1 — Split-point sweep: --split_layer 2/3/4 (already flag-driven)
  ★ Priority 2 — Warm-start from E1/E2 centralized checkpoint (--warmstart_path)
  ★ Priority 3 — FedProx proximal term (--mu), smishing-F1 checkpoint criterion

E3: Split local-only  — each client trains its own client+server halves independently (--local)
E4: Split FedAvg      — clients train locally, both halves are FedAvg-aggregated each round

Model partition (default: split_layer=3 out of 6 DistilBERT layers):
  Client side  → embeddings + transformer layers [0, split_layer)
  Server side  → transformer layers [split_layer, 6) + pre_classifier + classifier

U-Shaped split (--u_split):
  Client side  → embeddings + layers [0, split_layer) + layers [6-tail_layers, 6) + head
  Server side  → transformer layers [split_layer, 6-tail_layers)
  Privacy benefit: classifier never leaves the client.

Usage:
  # Standard split FedAvg (10 rounds, layer=3)
  python src/train_split.py --rounds 10 --local_epochs 2 --clients_dir data/clients/setting_D_300

  # Priority 1: Split-point sweep
  python src/train_split.py --split_layer 2 --rounds 10 --clients_dir data/clients/setting_D_300
  python src/train_split.py --split_layer 4 --rounds 10 --clients_dir data/clients/setting_D_300

  # Priority 2: Warm-start from centralized E1 full fine-tune checkpoint
  python src/train_split.py --warmstart_path models/e1_full_finetune --rounds 10

  # Priority 2: Warm-start from LoRA adapter checkpoint
  python src/train_split.py --warmstart_path models/lora_adapter --warmstart_lora --rounds 10

  # Priority 3: FedProx (mu=0.01 recommended starting point)
  python src/train_split.py --mu 0.01 --rounds 10

  # Priority 4b: U-shaped split (classifier stays on client)
  python src/train_split.py --u_split --split_layer 3 --tail_layers 1 --rounds 10

  # Combined (recommended):
  python src/train_split.py --split_layer 2 --warmstart_path models/e1_full_finetune \\
    --mu 0.01 --rounds 10 --local_epochs 2 --clients_dir data/clients/setting_D_300

Key arguments:
  --split_layer     Number of DistilBERT layers kept on the client side (default: 3)
  --rounds          Number of communication rounds (default: 10)
  --local_epochs    Local training epochs per client per round (default: 2)
  --warmstart_path  Path to a full fine-tune or LoRA adapter for warm-starting
  --warmstart_lora  If set, treat warmstart_path as a PEFT/LoRA adapter directory
  --mu              FedProx proximal coefficient (0 = standard FedAvg, default: 0.0)
  --u_split         Enable U-shaped split (client holds first + last layers + classifier)
  --tail_layers     Number of final transformer layers to keep on client in U-split (default: 1)
  --agg_weight      FedAvg weighting: smishing|sqrt|uniform|total|balanced
  --resume          Resume from saved split checkpoint

For LoRA+Split hybrid use train_split_lora.py.
For LoRA-only federated learning use train_fedlora.py.
"""

import sys
import copy
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

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
LOCAL_EPOCHS = 2
COMM_ROUNDS  = 10
LR           = 2e-4

CLIENT_IDS = ["client_1", "client_2", "client_3", "client_4", "client_5"]


# ── Attention mask helper ──────────────────────────────────────────────────────

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


# ── Standard Split model components ───────────────────────────────────────────

class SplitDistilBertClient(torch.nn.Module):
    """
    Client-side slice of DistilBERT.
    Holds the embedding layer + the first `split_layer` transformer blocks.
    Outputs intermediate hidden states [batch, seq_len, 768] sent to the server.
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


# ── U-Shaped Split model components (Priority 4b) ─────────────────────────────

class SplitDistilBertClientU(torch.nn.Module):
    """
    Client-side of U-shaped split learning.

    Holds:
      - Embeddings + layers [0, split_layer)          (early feature extraction)
      - Layers [6-tail_layers, 6) + head              (final classification — never leaves client)

    This means the final decision layers remain on-device for maximum privacy.
    The server only processes middle layers and never sees the classification head.

    Args:
        base_model:   full AutoModelForSequenceClassification
        split_layer:  first N layers kept on client (sent forward to server)
        tail_layers:  last M layers kept on client (receives from server)
    """

    def __init__(self, base_model, split_layer: int, tail_layers: int = 1):
        super().__init__()
        n_layers = len(base_model.distilbert.transformer.layer)   # 6 for DistilBERT
        tail_start = n_layers - tail_layers

        if split_layer >= tail_start:
            raise ValueError(
                f"split_layer={split_layer} must be < tail_start={tail_start}. "
                f"Reduce split_layer or tail_layers."
            )

        self.embeddings   = copy.deepcopy(base_model.distilbert.embeddings)
        self.head_layers  = torch.nn.ModuleList(
            [copy.deepcopy(layer)
             for layer in base_model.distilbert.transformer.layer[:split_layer]]
        )
        self.tail_layers  = torch.nn.ModuleList(
            [copy.deepcopy(layer)
             for layer in base_model.distilbert.transformer.layer[tail_start:]]
        )
        self.pre_classifier = copy.deepcopy(base_model.pre_classifier)
        self.dropout        = copy.deepcopy(base_model.dropout)
        self.classifier     = copy.deepcopy(base_model.classifier)

    def forward_head(self, input_ids, attention_mask):
        """Client forward: embeddings + head layers → hidden for server."""
        hidden = self.embeddings(input_ids)
        attention_mask = _expand_attention_mask(attention_mask, hidden)
        for layer in self.head_layers:
            hidden = _first_hidden_state(layer(hidden, attention_mask=attention_mask))
        return hidden

    def forward_tail(self, hidden, attention_mask):
        """Client backward: receives server output → final classification."""
        attention_mask = _expand_attention_mask(attention_mask, hidden)
        for layer in self.tail_layers:
            hidden = _first_hidden_state(layer(hidden, attention_mask=attention_mask))
        hidden = self.pre_classifier(hidden[:, 0])
        hidden = self.dropout(hidden)
        return self.classifier(hidden)

    def forward(self, input_ids, attention_mask, server_model):
        """Full U-shaped forward pass (convenience method for evaluation)."""
        hidden = self.forward_head(input_ids, attention_mask)
        hidden = server_model(hidden, attention_mask)
        return self.forward_tail(hidden, attention_mask)


class SplitDistilBertServerU(torch.nn.Module):
    """
    Server-side of U-shaped split learning.

    Holds only the middle transformer layers [split_layer, 6-tail_layers).
    Receives hidden states from the client head, processes them,
    and returns to the client tail.
    """

    def __init__(self, base_model, split_layer: int, tail_layers: int = 1):
        super().__init__()
        n_layers   = len(base_model.distilbert.transformer.layer)
        tail_start = n_layers - tail_layers
        self.transformer = torch.nn.ModuleList(
            [copy.deepcopy(layer)
             for layer in base_model.distilbert.transformer.layer[split_layer:tail_start]]
        )

    def forward(self, hidden, attention_mask):
        attention_mask = _expand_attention_mask(attention_mask, hidden)
        for layer in self.transformer:
            hidden = _first_hidden_state(layer(hidden, attention_mask=attention_mask))
        return hidden


# ── Warm-start helpers (Priority 2) ───────────────────────────────────────────

def load_warmstart_weights(
    warmstart_path: str,
    split_layer: int,
    is_lora: bool = False,
    u_split: bool = False,
    tail_layers: int = 1,
):
    """
    Load a warm-start checkpoint and return an initialised (client, server) pair.

    Supports two checkpoint types:
      1. Full fine-tune: AutoModelForSequenceClassification saved directory
      2. LoRA adapter: PEFT adapter saved with model.save_pretrained()

    Args:
        warmstart_path: path to saved model directory
        split_layer:    where to split the model
        is_lora:        True if the checkpoint is a PEFT/LoRA adapter
        u_split:        True if building U-shaped split models
        tail_layers:    U-split tail size (only used if u_split=True)

    Returns:
        (client_model, server_model) initialised from the warmstart weights
    """
    wpath = Path(warmstart_path)
    if not wpath.exists():
        raise FileNotFoundError(
            f"Warm-start path not found: {wpath}\n"
            "Set --warmstart_path to the directory saved by train_fedlora.py or "
            "a full fine-tune checkpoint."
        )

    print(f"  Loading warm-start from: {wpath} (lora={is_lora})")

    if is_lora:
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError("peft not installed. Run: pip install peft")
        base = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        merged_model = PeftModel.from_pretrained(base, str(wpath))
        # Merge LoRA weights into base for clean extraction
        merged_model = merged_model.merge_and_unload()
        base_for_split = merged_model
    else:
        base_for_split = AutoModelForSequenceClassification.from_pretrained(
            str(wpath),
            num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )

    if u_split:
        client = SplitDistilBertClientU(base_for_split, split_layer, tail_layers)
        server = SplitDistilBertServerU(base_for_split, split_layer, tail_layers)
    else:
        client = SplitDistilBertClient(base_for_split, split_layer)
        server = SplitDistilBertServer(base_for_split, split_layer)

    print(f"  Warm-start complete. Split at layer {split_layer}.")
    return client, server


# ── Model factory & checkpoint helpers ────────────────────────────────────────

def make_split_models(split_layer: int, u_split: bool = False, tail_layers: int = 1):
    """Return a freshly initialised (client, server) pair split at `split_layer`."""
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels = NUM_LABELS,
        id2label   = ID2LABEL,
        label2id   = LABEL2ID,
        ignore_mismatched_sizes = True,
    )
    if u_split:
        return SplitDistilBertClientU(base, split_layer, tail_layers), \
               SplitDistilBertServerU(base, split_layer, tail_layers)
    return SplitDistilBertClient(base, split_layer), SplitDistilBertServer(base, split_layer)


def save_split_checkpoint(directory: Path, client_model, server_model, tokenizer):
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(client_model.state_dict(), directory / "client_state.pt")
    torch.save(server_model.state_dict(), directory / "server_state.pt")
    tokenizer.save_pretrained(str(directory))


def load_split_checkpoint(directory: Path, split_layer: int,
                           u_split: bool = False, tail_layers: int = 1):
    client_model, server_model = make_split_models(split_layer, u_split, tail_layers)
    client_path = directory / "client_state.pt"
    server_path = directory / "server_state.pt"
    if client_path.exists() and server_path.exists():
        client_model.load_state_dict(torch.load(client_path, map_location="cpu"))
        server_model.load_state_dict(torch.load(server_path, map_location="cpu"))
    return client_model, server_model


# ── Local training (Priority 3: FedProx proximal term) ────────────────────────

def local_split_train(
    client_model, server_model, loader, device, class_weights, n_epochs, lr,
    mu: float = 0.0,
    global_client_state: Optional[dict] = None,
    global_server_state: Optional[dict] = None,
    u_split: bool = False,
):
    """
    Train the client+server pair end-to-end for n_epochs.

    Args:
        mu:                  FedProx proximal coefficient (0 = standard FedAvg)
        global_client_state: global client weights for FedProx (required if mu > 0)
        global_server_state: global server weights for FedProx (required if mu > 0)
        u_split:             True for U-shaped split forward pass
    """
    loss_fn   = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = AdamW(
        list(client_model.parameters()) + list(server_model.parameters()), lr=lr
    )
    total_steps  = len(loader) * n_epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Pre-load global parameters for FedProx (on device for fast comparison)
    global_client_params = None
    global_server_params = None
    if mu > 0.0 and global_client_state and global_server_state:
        global_client_params = [v.to(device).detach() for v in global_client_state.values()]
        global_server_params = [v.to(device).detach() for v in global_server_state.values()]

    client_model.train()
    server_model.train()

    for _ in range(n_epochs):
        for batch in loader:
            optimizer.zero_grad()
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device).bool()
            labels         = batch["labels"].to(device)

            if u_split:
                # U-shaped: client_head → server → client_tail
                hidden = client_model.forward_head(input_ids, attention_mask)
                hidden = server_model(hidden, attention_mask)
                logits = client_model.forward_tail(hidden, attention_mask)
            else:
                hidden = client_model(input_ids, attention_mask)
                logits = server_model(hidden, attention_mask)

            loss = loss_fn(logits, labels)

            # ── FedProx proximal term ──────────────────────────────────────────
            if mu > 0.0 and global_client_params and global_server_params:
                prox = torch.tensor(0.0, device=device)
                for p_local, p_global in zip(client_model.parameters(), global_client_params):
                    if p_local.requires_grad:
                        prox = prox + ((p_local - p_global) ** 2).sum()
                for p_local, p_global in zip(server_model.parameters(), global_server_params):
                    if p_local.requires_grad:
                        prox = prox + ((p_local - p_global) ** 2).sum()
                loss = loss + (mu / 2.0) * prox

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(client_model.parameters()) + list(server_model.parameters()), 1.0
            )
            optimizer.step()
            scheduler.step()

    return client_model, server_model


@torch.no_grad()
def evaluate_split(client_model, server_model, loader, device, u_split: bool = False):
    client_model.eval()
    server_model.eval()
    preds, trues = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device).bool()

        if u_split:
            hidden = client_model.forward_head(input_ids, attention_mask)
            hidden = server_model(hidden, attention_mask)
            logits = client_model.forward_tail(hidden, attention_mask)
        else:
            hidden = client_model(input_ids, attention_mask)
            logits = server_model(hidden, attention_mask)

        preds.extend(logits.argmax(dim=-1).cpu().numpy())
        trues.extend(batch["labels"].numpy())

    return np.array(trues), np.array(preds)


def _checkpoint_score(metrics: dict) -> float:
    """
    Composite checkpoint criterion: prioritises smishing F1 over macro F1.
    smishing_f1 is the primary metric; macro_f1 prevents class collapse.
    """
    smishing_f1 = metrics.get("per_class", {}).get("smishing", {}).get("f1", 0.0)
    macro_f1    = metrics.get("macro_f1", 0.0)
    return 0.3 * macro_f1 + 0.7 * smishing_f1


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Split learning training for smishing detection. "
                    "Supports warm-start, FedProx, and U-shaped split. "
                    "For LoRA+Split hybrid use train_split_lora.py. "
                    "For LoRA-only federated learning use train_fedlora.py."
    )
    # Core
    parser.add_argument("--local",        action="store_true",
                        help="Local-only mode (E3): each client trains independently, no aggregation")
    parser.add_argument("--split_layer",  type=int, default=3,
                        help="Number of DistilBERT layers to keep on the client side (1–5, default: 3)")
    parser.add_argument("--rounds",       type=int,   default=COMM_ROUNDS,
                        help=f"Communication rounds (default: {COMM_ROUNDS})")
    parser.add_argument("--local_epochs", type=int,   default=LOCAL_EPOCHS,
                        help=f"Local training epochs per round (default: {LOCAL_EPOCHS})")
    parser.add_argument("--lr",           type=float, default=LR)

    # Priority 2: Warm-start
    parser.add_argument("--warmstart_path", type=str, default=None,
                        help="Path to E1/E2 checkpoint to warm-start from "
                             "(AutoModelForSequenceClassification or PEFT adapter directory)")
    parser.add_argument("--warmstart_lora", action="store_true",
                        help="Treat --warmstart_path as a PEFT/LoRA adapter (merges before splitting)")

    # Priority 3: FedProx
    parser.add_argument("--mu",           type=float, default=0.0,
                        help="FedProx proximal coefficient (0=standard FedAvg, e.g. 0.01, default: 0)")

    # Priority 4b: U-shaped split
    parser.add_argument("--u_split",      action="store_true",
                        help="Enable U-shaped split: client holds first+last layers+classifier, "
                             "server holds only middle layers")
    parser.add_argument("--tail_layers",  type=int, default=1,
                        help="Number of final transformer layers kept on client in U-split (default: 1)")

    # Federation
    parser.add_argument("--resume",       action="store_true",
                        help="Resume from saved split checkpoint")
    parser.add_argument("--clients_dir",  type=str, default=None,
                        help="Path to client CSVs relative to project root (default: data/clients/)")
    parser.add_argument("--agg_weight",   type=str, default="smishing",
                        choices=["smishing", "sqrt", "uniform", "total", "balanced"])
    args = parser.parse_args()

    from utils import PROJECT_ROOT
    clients_dir  = (PROJECT_ROOT / args.clients_dir) if args.clients_dir else DATA_CLIENTS
    setting_name = clients_dir.name if args.clients_dir else "default"
    agg_tag      = args.agg_weight if not args.local else "none"
    split_tag    = f"L{args.split_layer}" + ("U" if args.u_split else "")
    mu_tag       = f"_mu{args.mu}" if args.mu > 0 else ""
    ws_tag       = "_ws" if args.warmstart_path else ""

    mode       = "split_local" if args.local else "split_fed"
    u_label    = " U-shaped" if args.u_split else ""
    experiment = (
        f"Split{u_label} Local (E3) [{split_tag}]"
        if args.local
        else f"Split{u_label} FedAvg (E4) [{split_tag}{mu_tag}{ws_tag} {setting_name}]"
    )

    print(
        f"Mode: {mode} | Rounds: {args.rounds} | Local epochs: {args.local_epochs} | "
        f"Split layer: {args.split_layer} | U-split: {args.u_split} | "
        f"FedProx mu: {args.mu} | Warm-start: {bool(args.warmstart_path)} | "
        f"Setting: {setting_name} | Agg: {agg_tag}"
    )

    device = get_device()
    if device is None or str(device) == "cpu":
        raise RuntimeError(
            "GPU (CUDA) not available. Split learning requires a CUDA-capable GPU."
        )
    import torch as _torch
    print(f"Device: {device}  ({_torch.cuda.get_device_name(0)})")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"\nLoading tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Validation and test sets ───────────────────────────────────────────────
    val_df  = load_split("val")
    test_df = load_split("test_clean")

    val_loader  = make_loader(val_df["cleaned_text"].fillna("").values,
                              encode_labels(val_df["label"]), tokenizer, shuffle=False)
    test_loader = make_loader(test_df["cleaned_text"].fillna("").values,
                              encode_labels(test_df["label"]), tokenizer, shuffle=False)

    # ── Client data ────────────────────────────────────────────────────────────
    clients = {}
    for cid in CLIENT_IDS:
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
        classes   = np.array(sorted(set(y)))
        cw        = compute_class_weight("balanced", classes=classes, y=y)
        cw_dict   = dict(zip(classes.tolist(), cw.tolist()))
        cw_tensor = torch.tensor([cw_dict.get(i, 1.0) for i in range(NUM_LABELS)], dtype=torch.float32)
        clients[cid] = {"X": X, "y": y, "cw": cw_tensor, "n": len(y)}
        label_dist = dict(pd.Series(df["label"]).value_counts())
        print(f"  {cid}: {len(y)} rows | {label_dist}")

    if not clients:
        print("ERROR: No client data found. Check --clients_dir.")
        sys.exit(1)

    active_clients = list(clients.keys())
    print(f"\nActive clients: {active_clients}")

    # ── Build split models ─────────────────────────────────────────────────────
    split_model_dir = MODELS_DIR / "split" / f"{split_tag}_{setting_name}{mu_tag}{ws_tag}"
    split_model_dir.mkdir(parents=True, exist_ok=True)

    print("\nInitialising split learning models...")
    if args.resume and (split_model_dir / "client_state.pt").exists():
        print(f"  Resuming from {split_model_dir}")
        global_client_model, global_server_model = load_split_checkpoint(
            split_model_dir, args.split_layer, args.u_split, args.tail_layers
        )
    elif args.warmstart_path:
        print("  [Priority 2] Warm-starting from checkpoint...")
        global_client_model, global_server_model = load_warmstart_weights(
            args.warmstart_path, args.split_layer,
            is_lora=args.warmstart_lora,
            u_split=args.u_split,
            tail_layers=args.tail_layers,
        )
    else:
        global_client_model, global_server_model = make_split_models(
            args.split_layer, args.u_split, args.tail_layers
        )

    global_client_model = global_client_model.to(device)
    global_server_model = global_server_model.to(device)

    total     = (sum(p.numel() for p in global_client_model.parameters())
                 + sum(p.numel() for p in global_server_model.parameters()))
    trainable = (sum(p.numel() for p in global_client_model.parameters() if p.requires_grad)
                 + sum(p.numel() for p in global_server_model.parameters() if p.requires_grad))
    print(f"Split model params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    if args.mu > 0:
        print(f"[Priority 3] FedProx enabled: mu={args.mu}")
    if args.warmstart_path:
        print("[Priority 2] Warm-start active.")
    if args.u_split:
        print(f"[Priority 4b] U-shaped split: head={args.split_layer} layers, "
              f"tail={args.tail_layers} layer(s)")

    round_results = []

    if args.local:
        # ── E3: Split local-only ───────────────────────────────────────────────
        print(f"\n[E3] Split local-only training (no aggregation)...")
        client_metrics = {}
        for cid in active_clients:
            print(f"\n  Training {cid}...")
            client_model, server_model = make_split_models(
                args.split_layer, args.u_split, args.tail_layers
            )
            client_model.load_state_dict(global_client_model.state_dict())
            server_model.load_state_dict(global_server_model.state_dict())
            client_model, server_model = client_model.to(device), server_model.to(device)

            loader = make_loader(clients[cid]["X"], clients[cid]["y"], tokenizer)
            client_model, server_model = local_split_train(
                client_model, server_model, loader, device,
                clients[cid]["cw"], args.local_epochs * args.rounds, args.lr,
                mu=args.mu, u_split=args.u_split,
            )

            y_t, y_p = evaluate_split(client_model, server_model, test_loader, device, args.u_split)
            m = compute_metrics(y_t, y_p)
            client_metrics[cid] = m
            print(f"  {cid} test macro_f1={m['macro_f1']:.4f} smishing_fnr={m['smishing_fnr']}")

            save_dir = MODELS_DIR / "split" / f"{cid}_local_{split_tag}"
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(client_model.state_dict(), save_dir / "client_state.pt")
            torch.save(server_model.state_dict(), save_dir / "server_state.pt")
            tokenizer.save_pretrained(str(save_dir))

        all_macro_f1 = np.mean([m["macro_f1"] for m in client_metrics.values()])
        print(f"\nLocal-only split avg macro_f1: {all_macro_f1:.4f}")
        append_result(
            REPORTS_DIR / "results_local_only_clients.csv",
            {
                "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M"),
                "experiment":  experiment,
                "split_tag":   split_tag,
                "avg_macro_f1": round(all_macro_f1, 4),
                **{f"{cid}_macro_f1": round(m["macro_f1"], 4) for cid, m in client_metrics.items()},
                "trainable_params": trainable,
            }
        )

    else:
        # ── E4: Split FedAvg ───────────────────────────────────────────────────
        print(f"\n[E4] Split FedAvg: {args.rounds} rounds, {args.local_epochs} epoch(s), "
              f"split_layer={args.split_layer}, mu={args.mu}")

        best_score  = 0.0
        best_round  = 0
        best_split_dir = MODELS_DIR / "split" / f"{split_tag}_{setting_name}{mu_tag}{ws_tag}_best"
        best_split_dir.mkdir(parents=True, exist_ok=True)

        for rnd in range(1, args.rounds + 1):
            print(f"\n--- Round {rnd}/{args.rounds} ---")
            client_states = []
            server_states = []
            client_sizes  = []

            # Cache global state for FedProx (CPU tensors)
            global_c_state = {k: v.cpu().clone() for k, v in global_client_model.state_dict().items()} \
                if args.mu > 0 else None
            global_s_state = {k: v.cpu().clone() for k, v in global_server_model.state_dict().items()} \
                if args.mu > 0 else None

            for cid in active_clients:
                client_model, server_model = make_split_models(
                    args.split_layer, args.u_split, args.tail_layers
                )
                client_model.load_state_dict(copy.deepcopy(global_client_model.state_dict()))
                server_model.load_state_dict(copy.deepcopy(global_server_model.state_dict()))
                client_model, server_model = client_model.to(device), server_model.to(device)

                loader = make_loader(clients[cid]["X"], clients[cid]["y"], tokenizer)
                client_model, server_model = local_split_train(
                    client_model, server_model, loader, device,
                    clients[cid]["cw"], args.local_epochs, args.lr,
                    mu=args.mu,
                    global_client_state=global_c_state,
                    global_server_state=global_s_state,
                    u_split=args.u_split,
                )

                client_states.append({k: v.cpu() for k, v in client_model.state_dict().items()})
                server_states.append({k: v.cpu() for k, v in server_model.state_dict().items()})
                client_sizes.append(clients[cid]["n"])

                y_v, y_p = evaluate_split(client_model, server_model, val_loader, device, args.u_split)
                m = compute_metrics(y_v, y_p)
                sm_f1 = m.get("per_class", {}).get("smishing", {}).get("f1", 0.0)
                print(f"  {cid}: val macro_f1={m['macro_f1']:.4f}  smishing_f1={sm_f1:.4f}")
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

            y_v, y_p = evaluate_split(
                global_client_model, global_server_model, val_loader, device, args.u_split
            )
            gm = compute_metrics(y_v, y_p)
            g_sm_f1 = gm.get("per_class", {}).get("smishing", {}).get("f1", 0.0)
            score   = _checkpoint_score(gm)     # 0.3*macro + 0.7*smishing_f1
            print(f"  GLOBAL val macro_f1={gm['macro_f1']:.4f}  smishing_f1={g_sm_f1:.4f}  "
                  f"smishing_fnr={gm['smishing_fnr']}  score={score:.4f}")

            if score > best_score:
                best_score = score
                best_round = rnd
                save_split_checkpoint(best_split_dir, global_client_model, global_server_model, tokenizer)
                print(f"  [★ best checkpoint saved — R{rnd}  score={score:.4f}  "
                      f"smishing_f1={g_sm_f1:.4f}  fnr={gm['smishing_fnr']}]")

            row = {
                "timestamp":          datetime.now().strftime("%Y-%m-%d %H:%M"),
                "round":              rnd,
                "split_tag":          split_tag,
                "mu":                 args.mu,
                "warmstart":          bool(args.warmstart_path),
                "global_macro_f1":    gm["macro_f1"],
                "global_weighted_f1": gm["weighted_f1"],
                "smishing_fnr":       gm["smishing_fnr"],
                "smishing_fpr":       gm["smishing_fpr"],
                "smishing_f1":        g_sm_f1,
                "checkpoint_score":   score,
            }
            round_results.append(row)
            append_result(REPORTS_DIR / f"results_split_{split_tag}_rounds.csv", row)

        # ── Final test evaluation ──────────────────────────────────────────────
        print(f"\n[E4] Final test evaluation ({experiment})...")
        y_t, y_p = evaluate_split(
            global_client_model, global_server_model, test_loader, device, args.u_split
        )
        test_metrics = compute_metrics(y_t, y_p)
        report_metrics(
            test_metrics, experiment, f"split_{split_tag}_final",
            y_t, y_p,
            extra={
                "split_tag":        split_tag,
                "split_layer":      args.split_layer,
                "u_split":          args.u_split,
                "mu":               args.mu,
                "warmstart":        bool(args.warmstart_path),
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
        print(f"Best split model (R{best_round}, score={best_score:.4f}) saved -> {best_split_dir}")

        # Round metrics plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            rr  = pd.DataFrame(round_results)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].plot(rr["round"], rr["global_macro_f1"], marker="o", label="Macro F1")
            axes[0].plot(rr["round"], rr["smishing_f1"],     marker="s", label="Smishing F1")
            axes[0].set_xlabel("Round"); axes[0].set_ylabel("F1")
            axes[0].set_title(f"F1 per Round [{split_tag}]"); axes[0].legend()

            axes[1].plot(rr["round"], rr["smishing_fnr"], marker="^", color="red", label="Smishing FNR")
            axes[1].set_xlabel("Round"); axes[1].set_ylabel("FNR")
            axes[1].set_title(f"Smishing FNR per Round [{split_tag}]"); axes[1].legend()

            plt.tight_layout()
            fig_path = REPORTS_DIR / "figures" / f"split_{split_tag}_rounds.png"
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            print(f"Round plot saved -> {fig_path.name}")
        except Exception as e:
            print(f"  Plot skipped: {e}")

    print("\nSplit learning training complete.")


if __name__ == "__main__":
    main()
