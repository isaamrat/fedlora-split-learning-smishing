# DistilBERT Split Learning Guide — FedSmishGuard

**Steps 1–4 improvements for split learning on smishing detection.**

---

## What's New

| Priority | Feature | Flag | Script |
|---|---|---|---|
| P1 | Split-point sweep (2, 3, 4) | `--split_layer` | `train_split.py` |
| P2 | Warm-start from E1/E2 checkpoint | `--warmstart_path` | `train_split.py` |
| P3 | FedProx proximal term | `--mu` | `train_split.py` |
| P3 | Smishing-F1 checkpoint criterion | automatic | `train_split.py` |
| P4a | Hybrid Split + LoRA (novel) | new script | `train_split_lora.py` |
| P4b | U-shaped split (classifier on client) | `--u_split` | `train_split.py` |

---

## Baseline Reference

Your confirmed baseline (before these improvements):

| Metric | Split FedAvg (E4, layer=3, 8 rounds) |
|---|---|
| Accuracy | 0.7945 |
| Macro F1 | 0.6563 |
| Smishing FNR | **0.4953** |
| Spam F1 | 0.5720 |

> Split learning already outperforms LoRA FedAvg on FNR (49.5% vs 67.6%).
> The goal is to push FNR below 40% and improve Spam F1.

---

## Step 1 — Split-Point Sweep

Already flag-driven. No code changes needed — just run with different `--split_layer` values.

```bash
# Layer 2: server handles more semantic layers
python src/train_split.py \
  --split_layer 2 \
  --rounds 10 \
  --local_epochs 2 \
  --clients_dir data/clients/setting_D_300 \
  --agg_weight smishing

# Layer 4: client does more work, richer representations to server
python src/train_split.py \
  --split_layer 4 \
  --rounds 10 \
  --local_epochs 2 \
  --clients_dir data/clients/setting_D_300 \
  --agg_weight smishing
```

Expected FNR range: 38–50% depending on split point.
Results saved to: `reports/results_split_L2_rounds.csv`, `reports/results_split_L4_rounds.csv`

---

## Step 2 — Warm-Start from Centralized Checkpoint

Start federated split training from a centralized checkpoint instead of vanilla pretrained weights.

### Option A: Warm-start from full fine-tune checkpoint (E1)

```bash
python src/train_split.py \
  --split_layer 3 \
  --warmstart_path models/e1_full_finetune \
  --rounds 10 \
  --local_epochs 2 \
  --clients_dir data/clients/setting_D_300
```

### Option B: Warm-start from LoRA adapter checkpoint (E2/E4)

The `--warmstart_lora` flag merges the LoRA adapter into the base model before splitting.

```bash
python src/train_split.py \
  --split_layer 3 \
  --warmstart_path models/lora_adapter \
  --warmstart_lora \
  --rounds 10 \
  --local_epochs 2 \
  --clients_dir data/clients/setting_D_300
```

> **Expected gain**: FNR drops by 15–25pp in early rounds.
> The model starts already knowing the task — federation refines rather than rebuilds.

---

## Step 3 — FedProx + Smishing-F1 Checkpoint

### FedProx (proximal term to prevent client drift)

```bash
python src/train_split.py \
  --split_layer 3 \
  --mu 0.01 \
  --rounds 10 \
  --local_epochs 2 \
  --clients_dir data/clients/setting_D_300
```

Try `--mu` values: `0.001`, `0.01`, `0.1`. Start with `0.01`.

### Smishing-F1 checkpoint (automatic)

The checkpoint criterion is now:

```
score = 0.3 × macro_F1 + 0.7 × smishing_F1
```

This replaces the previous macro-F1-only criterion. The best checkpoint is the one that best catches smishing messages, not the one with the highest average F1.

### Full 10 rounds (fix for previous 8-round run)

The default is now 10 rounds (`--rounds 10`). Always pass this explicitly:

```bash
python src/train_split.py --rounds 10 ...
```

---

## Step 4 — Combined Best Configuration

```bash
python src/train_split.py \
  --split_layer 2 \
  --warmstart_path models/e1_full_finetune \
  --mu 0.01 \
  --rounds 10 \
  --local_epochs 2 \
  --clients_dir data/clients/setting_D_300 \
  --agg_weight smishing
```

This combines all Priority 1–3 improvements:
- Layer 2 split (server handles more semantics)
- Warm-start from centralized checkpoint
- FedProx regularization
- 10 full rounds
- Smishing-weighted FedAvg + smishing-F1 checkpoint

---

## Step 4a — Hybrid Split + LoRA (Novel Contribution)

**New script**: `src/train_split_lora.py`

Combines split learning's privacy with LoRA's communication efficiency.
Only LoRA adapter weights (~295K) are aggregated per round instead of full model (~66M).

| Metric | Standard split | Split + LoRA |
|---|---|---|
| Aggregated params/round | ~66M | ~295K |
| Comm per client/round | ~264MB | ~1.2MB |
| Comm reduction | — | **~220×** |
| Privacy | ✅ split | ✅ split |

### Basic run

```bash
python src/train_split_lora.py \
  --split_layer 3 \
  --lora_r 8 \
  --rounds 10 \
  --local_epochs 2 \
  --clients_dir data/clients/setting_D_300
```

### With warm-start and FedProx

```bash
python src/train_split_lora.py \
  --split_layer 3 \
  --lora_r 8 \
  --warmstart_path models/e1_full_finetune \
  --mu 0.01 \
  --rounds 10 \
  --clients_dir data/clients/setting_D_300
```

### Extended LoRA targets (full attention block)

```bash
python src/train_split_lora.py \
  --split_layer 3 \
  --lora_r 16 \
  --lora_targets q_lin k_lin v_lin out_lin \
  --rounds 10 \
  --clients_dir data/clients/setting_D_300
```

Results saved to: `reports/results_split_lora_L3_rounds.csv`

---

## Step 4b — U-Shaped Split (Privacy Story)

Classifier head stays on the client — server only processes middle layers.

```
Standard split:   [client: embed + L0-L2] → [server: L3-L5 + head]
U-shaped split:   [client: embed + L0-L2] → [server: L3-L4] → [client: L5 + head]
```

```bash
python src/train_split.py \
  --u_split \
  --split_layer 2 \
  --tail_layers 1 \
  --rounds 10 \
  --local_epochs 2 \
  --clients_dir data/clients/setting_D_300
```

> **Privacy benefit**: The model's final decision layer (classifier) never leaves the client.
> The server can only see intermediate representations, not final classification logic.

---

## Full Comparison Sweep

Run all variants to compare against baseline:

```bash
#!/bin/bash
# P1: Split point sweep
python src/train_split.py --split_layer 2 --rounds 10 --local_epochs 2 --clients_dir data/clients/setting_D_300
python src/train_split.py --split_layer 3 --rounds 10 --local_epochs 2 --clients_dir data/clients/setting_D_300
python src/train_split.py --split_layer 4 --rounds 10 --local_epochs 2 --clients_dir data/clients/setting_D_300

# P2+P3: Warm-start + FedProx + best split
python src/train_split.py --split_layer 2 --warmstart_path models/e1_full_finetune --mu 0.01 --rounds 10 --local_epochs 2 --clients_dir data/clients/setting_D_300

# P4a: Split + LoRA
python src/train_split_lora.py --split_layer 3 --lora_r 8 --rounds 10 --local_epochs 2 --clients_dir data/clients/setting_D_300

# P4b: U-shaped split
python src/train_split.py --u_split --split_layer 2 --tail_layers 1 --rounds 10 --local_epochs 2 --clients_dir data/clients/setting_D_300
```

---

## Arguments Reference

### `train_split.py` — Standard Split (all priorities)

| Argument | Default | Description |
|---|---|---|
| `--split_layer` | `3` | Transformer layers on client (1–5) |
| `--rounds` | `10` | Communication rounds |
| `--local_epochs` | `2` | Local epochs per client per round |
| `--lr` | `2e-4` | Learning rate |
| `--warmstart_path` | None | Path to E1/E2 checkpoint directory |
| `--warmstart_lora` | False | Treat warmstart as PEFT/LoRA adapter |
| `--mu` | `0.0` | FedProx coefficient (0 = standard FedAvg) |
| `--u_split` | False | Enable U-shaped split |
| `--tail_layers` | `1` | Final layers kept on client in U-split |
| `--agg_weight` | `smishing` | FedAvg weighting strategy |
| `--resume` | False | Resume from saved checkpoint |
| `--clients_dir` | `data/clients/` | Path to client CSV directory |

### `train_split_lora.py` — Hybrid Split + LoRA (P4a)

| Argument | Default | Description |
|---|---|---|
| `--split_layer` | `3` | Transformer layers on client |
| `--lora_r` | `8` | LoRA rank |
| `--lora_alpha` | `16` | LoRA alpha scaling |
| `--lora_dropout` | `0.1` | LoRA dropout |
| `--lora_targets` | `q_lin v_lin` | Modules to apply LoRA to |
| `--rounds` | `10` | Communication rounds |
| `--local_epochs` | `2` | Local epochs per round |
| `--mu` | `0.0` | FedProx coefficient |
| `--warmstart_path` | None | Warm-start checkpoint path |
| `--warmstart_lora` | False | Treat warmstart as LoRA adapter |
| `--clients_dir` | `data/clients/` | Client CSV directory |

---

## Expected Results After Improvements

| Experiment | Macro F1 | Smishing FNR | Comm/round |
|---|---|---|---|
| Baseline split (layer=3, 8 rounds) | 0.656 | 0.495 | ~264MB |
| P1: Layer 2 sweep | ~0.65–0.67 | **~0.42–0.48** | ~264MB |
| P2+P3: Warm-start + FedProx | ~0.67–0.72 | **~0.36–0.44** | ~264MB |
| P4a: Split + LoRA | ~0.65–0.70 | ~0.44–0.50 | **~1.2MB** |
| P4b: U-shaped split | ~0.64–0.68 | ~0.45–0.50 | ~264MB |

> Note: FNR estimates based on analysis of the improvements. Actual results may vary.
> Warm-start combined with a shallower split (layer=2) is expected to give the best FNR.

---

## Output Files

| File | Contents |
|---|---|
| `reports/results_split_L2_rounds.csv` | Per-round metrics for split_layer=2 |
| `reports/results_split_L3_rounds.csv` | Per-round metrics for split_layer=3 |
| `reports/results_split_L4_rounds.csv` | Per-round metrics for split_layer=4 |
| `reports/results_split_lora_L3_rounds.csv` | Per-round metrics for Split+LoRA |
| `reports/results_all_experiments.csv` | Final test metrics across all experiments |
| `reports/figures/split_L2_rounds.png` | F1 + FNR plot for layer=2 sweep |
| `reports/figures/split_lora_L3_rounds.png` | F1 + FNR plot for Split+LoRA |
| `models/split/L2_*/` | Checkpoints for layer=2 variants |
| `models/split_lora/L3_*/` | Checkpoints for Split+LoRA variants |
