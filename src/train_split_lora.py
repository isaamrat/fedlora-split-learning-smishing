"""
train_split_lora.py — Hybrid Split + LoRA Federated Learning (Priority 4a)

Novel contribution: combines the privacy benefit of split learning with the
communication efficiency of LoRA. Instead of aggregating full model weights
(66M params), only the LoRA adapter weights (~295K) are aggregated per round.

Architecture:
  Client side → embeddings + layers [0, split_layer) with LoRA adapters
  Server side → layers [split_layer, 6) + classifier with LoRA adapters

  Forward pass: client ──hidden states──► server
  FedAvg:       only LoRA adapter weights aggregated (not base model weights)

Key advantages over standard split learning:
  - Communication: ~295K vs ~66M params aggregated per round (~220× reduction)
  - Stability: fewer parameters means less Non-IID aggregation drift
  - Privacy: split structure preserved — classifier on server, base layers on client

Comparison with train_split.py:
  | Metric               | Standard split | Hybrid Split+LoRA |
  |----------------------|----------------|-------------------|
  | Trainable params     | ~66M           | ~295K             |
  | Comm per round       | ~1.3GB         | ~5.9MB            |
  | Non-IID sensitivity  | High           | Low               |
  | Split privacy        | ✅             | ✅                |

Usage:
  # Default (split_layer=3, rank=8)
  python src/train_split_lora.py --rounds 10 --clients_dir data/clients/setting_D_300

  # Warm-start from E1/E2 checkpoint
  python src/train_split_lora.py --warmstart_path models/e1_full_finetune --rounds 10

  # Custom split + LoRA rank + FedProx
  python src/train_split_lora.py --split_layer 2 --lora_r 16 --mu 0.01 --rounds 10

  # Different target modules (full attention block)
  python src/train_split_lora.py --lora_targets q_lin k_lin v_lin out_lin --rounds 10

Key arguments:
  --split_layer    DistilBERT layers on client side (default: 3)
  --lora_r         LoRA rank (default: 8)
  --lora_alpha     LoRA alpha, controls scaling (default: 16)
  --lora_dropout   LoRA dropout (default: 0.1)
  --lora_targets   Modules to apply LoRA to (default: q_lin v_lin)
  --warmstart_path Path to full fine-tune or LoRA checkpoint
  --warmstart_lora Treat warmstart_path as a PEFT adapter directory
  --mu             FedProx proximal coefficient (default: 0.0)
  --rounds         Communication rounds (default: 10)
  --local_epochs   Local training epochs per round (default: 2)

For standard split learning (all params) use train_split.py.
For LoRA-only federated learning use train_fedlora.py.
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
        get_peft_model_state_dict, set_peft_model_state_dict,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

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
LORA_R       = 8
LORA_ALPHA   = 16
LORA_DROPOUT = 0.1
LORA_TARGETS = ["q_lin", "v_lin"]

CLIENT_IDS = ["client_1", "client_2", "client_3", "client_4", "client_5"]


# ── Attention mask helper ──────────────────────────────────────────────────────

def _expand_attention_mask(attention_mask, hidden_states):
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
    return layer_output[0] if isinstance(layer_output, (tuple, list)) else layer_output


# ── LoRA-augmented split model components ─────────────────────────────────────

class SplitLoRAClient(torch.nn.Module):
    """
    Client-side LoRA split model.

    Base weights (embeddings + first `split_layer` transformer blocks) are frozen.
    LoRA adapters are added to specified attention modules within those layers.
    Only the LoRA adapter weights are trained and aggregated via FedAvg.

    Args:
        base_model:   full AutoModelForSequenceClassification
        split_layer:  number of transformer layers to keep on client
        lora_config:  PEFT LoraConfig instance
    """

    def __init__(self, base_model, split_layer: int, lora_config):
        super().__init__()
        # Extract client-side sub-model
        import copy as _copy

        self.embeddings  = _copy.deepcopy(base_model.distilbert.embeddings)
        # Build a mini transformer wrapping only the client-side layers
        # We apply LoRA by wrapping each layer with PEFT individually
        raw_layers = [_copy.deepcopy(layer)
                      for layer in base_model.distilbert.transformer.layer[:split_layer]]
        self.transformer = torch.nn.ModuleList(raw_layers)

        # Freeze all base weights
        for param in self.embeddings.parameters():
            param.requires_grad = False
        for layer in self.transformer:
            for param in layer.parameters():
                param.requires_grad = False

        # Add LoRA adapters to target modules within each transformer layer
        self._add_lora_adapters(lora_config)

    def _add_lora_adapters(self, lora_config):
        """Attach LoRA linear layers to target modules in each transformer block."""
        from peft.tuners.lora import Linear as LoRALinear
        import re

        self._lora_layers = torch.nn.ModuleDict()
        for layer_idx, layer in enumerate(self.transformer):
            for module_name in lora_config.target_modules:
                # DistilBERT attention uses: attention.q_lin, attention.k_lin, etc.
                for subpath, module in layer.named_modules():
                    if subpath.endswith(module_name) and isinstance(module, torch.nn.Linear):
                        lora_linear = LoRALinear(
                            base_layer  = module,
                            adapter_name = "default",
                            r            = lora_config.r,
                            lora_alpha   = lora_config.lora_alpha,
                            lora_dropout = lora_config.lora_dropout,
                            fan_in_fan_out = False,
                            init_lora_weights = True,
                        )
                        # Replace the original module with the LoRA-wrapped one
                        parent_name = ".".join(subpath.split(".")[:-1])
                        attr_name   = subpath.split(".")[-1]
                        parent = layer
                        if parent_name:
                            for part in parent_name.split("."):
                                parent = getattr(parent, part)
                        setattr(parent, attr_name, lora_linear)
                        key = f"layer{layer_idx}_{subpath.replace('.', '_')}"
                        self._lora_layers[key] = lora_linear

    def forward(self, input_ids, attention_mask):
        hidden = self.embeddings(input_ids)
        attention_mask = _expand_attention_mask(attention_mask, hidden)
        for layer in self.transformer:
            hidden = _first_hidden_state(layer(hidden, attention_mask=attention_mask))
        return hidden

    def get_lora_state_dict(self) -> dict:
        """Return only the LoRA adapter parameters (A, B matrices)."""
        return {k: v for k, v in self.state_dict().items()
                if "lora_A" in k or "lora_B" in k}

    def set_lora_state_dict(self, lora_state: dict):
        """Load LoRA adapter parameters without touching base weights."""
        own_state = self.state_dict()
        for k, v in lora_state.items():
            if k in own_state:
                own_state[k].copy_(v)


class SplitLoRAServer(torch.nn.Module):
    """
    Server-side LoRA split model.

    Base weights (transformer layers [split_layer, 6) + classifier head) are frozen.
    LoRA adapters are added to specified attention modules within those layers.
    Only LoRA adapter weights are trained and aggregated via FedAvg.

    Args:
        base_model:   full AutoModelForSequenceClassification
        split_layer:  first transformer layer on the server
        lora_config:  PEFT LoraConfig instance
    """

    def __init__(self, base_model, split_layer: int, lora_config):
        super().__init__()
        import copy as _copy

        raw_layers = [_copy.deepcopy(layer)
                      for layer in base_model.distilbert.transformer.layer[split_layer:]]
        self.transformer    = torch.nn.ModuleList(raw_layers)
        self.pre_classifier = _copy.deepcopy(base_model.pre_classifier)
        self.dropout        = _copy.deepcopy(base_model.dropout)
        self.classifier     = _copy.deepcopy(base_model.classifier)

        # Freeze all base weights
        for layer in self.transformer:
            for param in layer.parameters():
                param.requires_grad = False
        for param in self.pre_classifier.parameters():
            param.requires_grad = False
        # Classifier head stays trainable (small, task-specific)
        # — leave classifier trainable to allow task adaptation

        self._add_lora_adapters(lora_config)

    def _add_lora_adapters(self, lora_config):
        from peft.tuners.lora import Linear as LoRALinear

        self._lora_layers = torch.nn.ModuleDict()
        for layer_idx, layer in enumerate(self.transformer):
            for module_name in lora_config.target_modules:
                for subpath, module in layer.named_modules():
                    if subpath.endswith(module_name) and isinstance(module, torch.nn.Linear):
                        lora_linear = LoRALinear(
                            base_layer   = module,
                            adapter_name = "default",
                            r            = lora_config.r,
                            lora_alpha   = lora_config.lora_alpha,
                            lora_dropout = lora_config.lora_dropout,
                            fan_in_fan_out = False,
                            init_lora_weights = True,
                        )
                        parent_name = ".".join(subpath.split(".")[:-1])
                        attr_name   = subpath.split(".")[-1]
                        parent = layer
                        if parent_name:
                            for part in parent_name.split("."):
                                parent = getattr(parent, part)
                        setattr(parent, attr_name, lora_linear)
                        key = f"layer{layer_idx}_{subpath.replace('.', '_')}"
                        self._lora_layers[key] = lora_linear

    def forward(self, hidden, attention_mask):
        attention_mask = _expand_attention_mask(attention_mask, hidden)
        for layer in self.transformer:
            hidden = _first_hidden_state(layer(hidden, attention_mask=attention_mask))
        hidden = self.pre_classifier(hidden[:, 0])
        hidden = self.dropout(hidden)
        return self.classifier(hidden)

    def get_lora_state_dict(self) -> dict:
        """Return LoRA adapter params + classifier head (both small and trainable)."""
        lora = {k: v for k, v in self.state_dict().items()
                if "lora_A" in k or "lora_B" in k}
        clf  = {f"classifier.{k}": v for k, v in self.classifier.state_dict().items()}
        return {**lora, **clf}

    def set_lora_state_dict(self, lora_state: dict):
        own_state = self.state_dict()
        for k, v in lora_state.items():
            if k in own_state:
                own_state[k].copy_(v)


# ── Model factory ──────────────────────────────────────────────────────────────

def make_lora_config(r, alpha, dropout, targets):
    return LoraConfig(
        r              = r,
        lora_alpha     = alpha,
        lora_dropout   = dropout,
        target_modules = targets,
        bias           = "none",
        task_type      = TaskType.SEQ_CLS,
    )


def make_split_lora_models(split_layer: int, lora_config,
                            warmstart_path: Optional[str] = None,
                            warmstart_lora: bool = False):
    """
    Build (client, server) split+LoRA pair.

    If warmstart_path is given, base weights are loaded from that checkpoint
    instead of the vanilla pretrained DistilBERT.
    """
    if warmstart_path:
        wpath = Path(warmstart_path)
        print(f"  Loading base from warm-start: {wpath}")
        if warmstart_lora:
            try:
                from peft import PeftModel
            except ImportError:
                raise ImportError("peft not installed")
            base_raw = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL,
                label2id=LABEL2ID, ignore_mismatched_sizes=True,
            )
            base_raw = PeftModel.from_pretrained(base_raw, str(wpath))
            base = base_raw.merge_and_unload()
        else:
            base = AutoModelForSequenceClassification.from_pretrained(
                str(wpath), num_labels=NUM_LABELS, id2label=ID2LABEL,
                label2id=LABEL2ID, ignore_mismatched_sizes=True,
            )
    else:
        base = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL,
            label2id=LABEL2ID, ignore_mismatched_sizes=True,
        )

    client = SplitLoRAClient(base, split_layer, lora_config)
    server = SplitLoRAServer(base, split_layer, lora_config)
    return client, server


def save_split_lora_checkpoint(directory: Path, client_model, server_model, tokenizer):
    """Save only LoRA adapter weights (not full model) + tokenizer."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(client_model.get_lora_state_dict(), directory / "client_lora.pt")
    torch.save(server_model.get_lora_state_dict(), directory / "server_lora.pt")
    tokenizer.save_pretrained(str(directory))


def load_split_lora_checkpoint(directory: Path, client_model, server_model):
    """Load LoRA adapter weights into existing models."""
    cp = directory / "client_lora.pt"
    sp = directory / "server_lora.pt"
    if cp.exists():
        client_model.set_lora_state_dict(torch.load(cp, map_location="cpu"))
    if sp.exists():
        server_model.set_lora_state_dict(torch.load(sp, map_location="cpu"))
    return client_model, server_model


# ── Training ───────────────────────────────────────────────────────────────────

def local_split_lora_train(
    client_model, server_model, loader, device, class_weights, n_epochs, lr,
    mu: float = 0.0,
    global_client_lora: Optional[dict] = None,
    global_server_lora: Optional[dict] = None,
):
    """
    Train only LoRA adapter weights end-to-end through the split boundary.

    FedProx is applied to the LoRA parameters specifically.
    """
    loss_fn   = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    trainable = (list(filter(lambda p: p.requires_grad, client_model.parameters())) +
                 list(filter(lambda p: p.requires_grad, server_model.parameters())))
    optimizer = AdamW(trainable, lr=lr)
    total_steps  = len(loader) * n_epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # FedProx global reference on device
    g_c_params = None
    g_s_params = None
    if mu > 0.0 and global_client_lora and global_server_lora:
        g_c_params = {k: v.to(device).detach() for k, v in global_client_lora.items()}
        g_s_params = {k: v.to(device).detach() for k, v in global_server_lora.items()}

    client_model.train()
    server_model.train()

    for _ in range(n_epochs):
        for batch in loader:
            optimizer.zero_grad()
            hidden = client_model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device).bool(),
            )
            logits = server_model(hidden, batch["attention_mask"].to(device).bool())
            loss   = loss_fn(logits, batch["labels"].to(device))

            # FedProx on LoRA parameters
            if mu > 0.0 and g_c_params and g_s_params:
                prox = torch.tensor(0.0, device=device)
                c_lora = client_model.get_lora_state_dict()
                s_lora = server_model.get_lora_state_dict()
                for k, p_local in c_lora.items():
                    if k in g_c_params and p_local.requires_grad:
                        prox = prox + ((p_local - g_c_params[k]) ** 2).sum()
                for k, p_local in s_lora.items():
                    if k in g_s_params and p_local.requires_grad:
                        prox = prox + ((p_local - g_s_params[k]) ** 2).sum()
                loss = loss + (mu / 2.0) * prox

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()

    return client_model, server_model


@torch.no_grad()
def evaluate_split_lora(client_model, server_model, loader, device):
    client_model.eval()
    server_model.eval()
    preds, trues = [], []
    for batch in loader:
        hidden = client_model(batch["input_ids"].to(device),
                              batch["attention_mask"].to(device).bool())
        logits = server_model(hidden, batch["attention_mask"].to(device).bool())
        preds.extend(logits.argmax(dim=-1).cpu().numpy())
        trues.extend(batch["labels"].numpy())
    return np.array(trues), np.array(preds)


def _checkpoint_score(metrics: dict) -> float:
    smishing_f1 = metrics.get("per_class", {}).get("smishing", {}).get("f1", 0.0)
    macro_f1    = metrics.get("macro_f1", 0.0)
    return 0.3 * macro_f1 + 0.7 * smishing_f1


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not PEFT_AVAILABLE:
        print("ERROR: peft not installed. Run: pip install peft>=0.10.0")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Hybrid Split+LoRA federated learning. Combines privacy of split learning "
                    "with communication efficiency of LoRA. Only LoRA adapter weights are "
                    "aggregated per FedAvg round (~295K vs ~66M for standard split)."
    )
    parser.add_argument("--split_layer",   type=int,   default=3)
    parser.add_argument("--lora_r",        type=int,   default=LORA_R)
    parser.add_argument("--lora_alpha",    type=int,   default=LORA_ALPHA)
    parser.add_argument("--lora_dropout",  type=float, default=LORA_DROPOUT)
    parser.add_argument("--lora_targets",  nargs="+",  default=LORA_TARGETS,
                        help="Target modules for LoRA (default: q_lin v_lin)")
    parser.add_argument("--rounds",        type=int,   default=COMM_ROUNDS)
    parser.add_argument("--local_epochs",  type=int,   default=LOCAL_EPOCHS)
    parser.add_argument("--lr",            type=float, default=LR)
    parser.add_argument("--mu",            type=float, default=0.0,
                        help="FedProx coefficient (0=standard FedAvg)")
    parser.add_argument("--warmstart_path", type=str, default=None)
    parser.add_argument("--warmstart_lora", action="store_true")
    parser.add_argument("--clients_dir",   type=str,   default=None)
    parser.add_argument("--agg_weight",    type=str,   default="smishing",
                        choices=["smishing", "sqrt", "uniform", "total", "balanced"])
    parser.add_argument("--resume",        action="store_true")
    args = parser.parse_args()

    from utils import PROJECT_ROOT
    clients_dir  = (PROJECT_ROOT / args.clients_dir) if args.clients_dir else DATA_CLIENTS
    setting_name = clients_dir.name if args.clients_dir else "default"
    mu_tag       = f"_mu{args.mu}" if args.mu > 0 else ""
    ws_tag       = "_ws" if args.warmstart_path else ""
    split_tag    = f"L{args.split_layer}"
    experiment   = (f"Split+LoRA FedAvg (P4a) "
                    f"[{split_tag} r={args.lora_r}{mu_tag}{ws_tag} {setting_name}]")

    print(
        f"Hybrid Split+LoRA | Split: {args.split_layer} | "
        f"LoRA r={args.lora_r} alpha={args.lora_alpha} targets={args.lora_targets} | "
        f"Rounds: {args.rounds} | mu={args.mu} | Warm-start: {bool(args.warmstart_path)} | "
        f"Setting: {setting_name}"
    )

    device = get_device()
    if device is None or str(device) == "cpu":
        raise RuntimeError("GPU required for split+LoRA training.")
    import torch as _torch
    print(f"Device: {device}  ({_torch.cuda.get_device_name(0)})")

    tokenizer   = AutoTokenizer.from_pretrained(MODEL_NAME)
    lora_config = make_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_targets)

    val_df  = load_split("val")
    test_df = load_split("test_clean")
    val_loader  = make_loader(val_df["cleaned_text"].fillna("").values,
                              encode_labels(val_df["label"]), tokenizer, shuffle=False)
    test_loader = make_loader(test_df["cleaned_text"].fillna("").values,
                              encode_labels(test_df["label"]), tokenizer, shuffle=False)

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
        label_dist = dict(pd.Series(df["label"]).value_counts())
        print(f"  {cid}: {len(y)} rows | {label_dist}")

    if not clients:
        print("ERROR: No client data found.")
        sys.exit(1)
    active_clients = list(clients.keys())

    # ── Build global split+LoRA models ─────────────────────────────────────────
    save_dir = MODELS_DIR / "split_lora" / f"{split_tag}_{setting_name}{mu_tag}{ws_tag}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBuilding Split+LoRA models (split_layer={args.split_layer}, r={args.lora_r})...")
    global_client, global_server = make_split_lora_models(
        args.split_layer, lora_config,
        warmstart_path=args.warmstart_path,
        warmstart_lora=args.warmstart_lora,
    )

    if args.resume:
        global_client, global_server = load_split_lora_checkpoint(save_dir, global_client, global_server)
        print(f"  Resumed LoRA weights from {save_dir}")

    global_client = global_client.to(device)
    global_server = global_server.to(device)

    total_c = sum(p.numel() for p in global_client.parameters())
    total_s = sum(p.numel() for p in global_server.parameters())
    train_c = sum(p.numel() for p in global_client.parameters() if p.requires_grad)
    train_s = sum(p.numel() for p in global_server.parameters() if p.requires_grad)
    lora_c  = sum(v.numel() for v in global_client.get_lora_state_dict().values())
    lora_s  = sum(v.numel() for v in global_server.get_lora_state_dict().values())

    print(f"Client params: {train_c:,} trainable / {total_c:,} total")
    print(f"Server params: {train_s:,} trainable / {total_s:,} total")
    print(f"LoRA aggregated per round: {lora_c + lora_s:,} params "
          f"({(lora_c + lora_s) * 4 / 1e6:.2f} MB per client)")
    print(f"Communication vs standard split: "
          f"{(lora_c + lora_s) / (total_c + total_s) * 100:.2f}% of full weights")

    best_score = 0.0
    best_round = 0
    best_dir   = MODELS_DIR / "split_lora" / f"{split_tag}_{setting_name}{mu_tag}{ws_tag}_best"
    best_dir.mkdir(parents=True, exist_ok=True)
    round_results = []

    print(f"\n[P4a] Split+LoRA FedAvg: {args.rounds} rounds, {args.local_epochs} epoch(s), "
          f"split={args.split_layer}, r={args.lora_r}, mu={args.mu}")

    for rnd in range(1, args.rounds + 1):
        print(f"\n--- Round {rnd}/{args.rounds} ---")
        client_lora_states = []
        server_lora_states = []
        client_sizes       = []

        # FedProx global LoRA reference
        g_c_lora = global_client.get_lora_state_dict() if args.mu > 0 else None
        g_s_lora = global_server.get_lora_state_dict() if args.mu > 0 else None

        for cid in active_clients:
            c_model, s_model = make_split_lora_models(args.split_layer, lora_config)
            # Copy global LoRA adapter weights into local models
            c_model.set_lora_state_dict(global_client.get_lora_state_dict())
            s_model.set_lora_state_dict(global_server.get_lora_state_dict())
            c_model, s_model = c_model.to(device), s_model.to(device)

            loader = make_loader(clients[cid]["X"], clients[cid]["y"], tokenizer)
            c_model, s_model = local_split_lora_train(
                c_model, s_model, loader, device,
                clients[cid]["cw"], args.local_epochs, args.lr,
                mu=args.mu,
                global_client_lora=g_c_lora,
                global_server_lora=g_s_lora,
            )

            client_lora_states.append({k: v.cpu() for k, v in c_model.get_lora_state_dict().items()})
            server_lora_states.append({k: v.cpu() for k, v in s_model.get_lora_state_dict().items()})
            client_sizes.append(clients[cid]["n"])

            y_v, y_p = evaluate_split_lora(c_model, s_model, val_loader, device)
            m = compute_metrics(y_v, y_p)
            sm_f1 = m.get("per_class", {}).get("smishing", {}).get("f1", 0.0)
            print(f"  {cid}: val macro_f1={m['macro_f1']:.4f}  smishing_f1={sm_f1:.4f}")
            del c_model, s_model

        smishing_sizes = [
            int((clients[cid]["y"] == LABEL2ID["smishing"]).sum())
            for cid in active_clients
        ]

        # ── Aggregate ONLY LoRA adapter weights ───────────────────────────────
        # Build temporary full state dicts for fedavg_state_dict compatibility
        def _lora_fedavg(global_model, lora_states, sizes, smishing_sizes):
            """Aggregate LoRA-only dicts via weighted average."""
            weights = np.array([s for s in smishing_sizes], dtype=np.float64)
            if weights.sum() == 0:
                weights = np.ones(len(sizes))
            weights = weights / weights.sum()

            merged = {}
            for key in lora_states[0].keys():
                stacked = torch.stack([st[key].float() for st in lora_states], dim=0)
                w_tensor = torch.tensor(weights, dtype=stacked.dtype).view(
                    -1, *([1] * (stacked.dim() - 1))
                )
                merged[key] = (stacked * w_tensor).sum(dim=0)

            global_model.set_lora_state_dict(merged)
            return global_model

        global_client = _lora_fedavg(global_client, client_lora_states, client_sizes, smishing_sizes)
        global_server = _lora_fedavg(global_server, server_lora_states, client_sizes, smishing_sizes)

        y_v, y_p = evaluate_split_lora(global_client, global_server, val_loader, device)
        gm = compute_metrics(y_v, y_p)
        g_sm_f1 = gm.get("per_class", {}).get("smishing", {}).get("f1", 0.0)
        score   = 0.3 * gm["macro_f1"] + 0.7 * g_sm_f1

        print(f"  GLOBAL val macro_f1={gm['macro_f1']:.4f}  smishing_f1={g_sm_f1:.4f}  "
              f"smishing_fnr={gm['smishing_fnr']}  score={score:.4f}")

        if score > best_score:
            best_score = score
            best_round = rnd
            save_split_lora_checkpoint(best_dir, global_client, global_server, tokenizer)
            print(f"  [★ best checkpoint saved — R{rnd}  score={score:.4f}  fnr={gm['smishing_fnr']}]")

        row = {
            "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M"),
            "round":          rnd,
            "split_layer":    args.split_layer,
            "lora_r":         args.lora_r,
            "mu":             args.mu,
            "warmstart":      bool(args.warmstart_path),
            "macro_f1":       gm["macro_f1"],
            "weighted_f1":    gm["weighted_f1"],
            "smishing_fnr":   gm["smishing_fnr"],
            "smishing_fpr":   gm["smishing_fpr"],
            "smishing_f1":    g_sm_f1,
            "score":          score,
            "lora_comm_mb":   (lora_c + lora_s) * 4 / 1e6,
        }
        round_results.append(row)
        append_result(REPORTS_DIR / f"results_split_lora_{split_tag}_rounds.csv", row)

    # ── Final test ─────────────────────────────────────────────────────────────
    print(f"\n[P4a] Final test evaluation...")
    y_t, y_p = evaluate_split_lora(global_client, global_server, test_loader, device)
    test_metrics = compute_metrics(y_t, y_p)
    report_metrics(
        test_metrics, experiment, f"split_lora_{split_tag}_final",
        y_t, y_p,
        extra={
            "split_layer":    args.split_layer,
            "lora_r":         args.lora_r,
            "lora_alpha":     args.lora_alpha,
            "mu":             args.mu,
            "warmstart":      bool(args.warmstart_path),
            "rounds":         args.rounds,
            "local_epochs":   args.local_epochs,
            "n_clients":      len(active_clients),
            "lora_params":    lora_c + lora_s,
            "total_params":   total_c + total_s,
            "comm_bytes_est": (lora_c + lora_s) * 4 * len(active_clients) * args.rounds,
        }
    )

    save_split_lora_checkpoint(save_dir, global_client, global_server, tokenizer)
    print(f"Final Split+LoRA model saved -> {save_dir}")
    print(f"Best model (R{best_round}, score={best_score:.4f}) saved -> {best_dir}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rr = pd.DataFrame(round_results)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(rr["round"], rr["macro_f1"],    marker="o", label="Macro F1")
        axes[0].plot(rr["round"], rr["smishing_f1"], marker="s", label="Smishing F1")
        axes[0].set_xlabel("Round"); axes[0].set_ylabel("F1")
        axes[0].set_title(f"Split+LoRA [{split_tag} r={args.lora_r}] — F1"); axes[0].legend()
        axes[1].plot(rr["round"], rr["smishing_fnr"], marker="^", color="red", label="FNR")
        axes[1].set_xlabel("Round"); axes[1].set_ylabel("FNR")
        axes[1].set_title(f"Split+LoRA — Smishing FNR"); axes[1].legend()
        plt.tight_layout()
        fig_path = REPORTS_DIR / "figures" / f"split_lora_{split_tag}_rounds.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"Plot saved -> {fig_path.name}")
    except Exception as e:
        print(f"  Plot skipped: {e}")

    print("\nSplit+LoRA training complete.")


if __name__ == "__main__":
    main()
