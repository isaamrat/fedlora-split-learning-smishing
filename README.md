# FedSmishGuard

**Smishing-Aware Federated LoRA for Non-IID SMS Phishing Detection**

A privacy-preserving SMS phishing (smishing) detection system using Federated Learning with LoRA adapters on DistilBERT. Five heterogeneous mobile clients train locally on their own SMS data — raw messages never leave the device. Only lightweight LoRA adapter weights (2.96MB per client) are exchanged.

---

## Key Results

| Method | Clean FNR ↓ | Clean Macro F1 ↑ |
|---|---|---|
| TF-IDF + SVM (baseline) | ~36% | ~0.76 |
| Centralized LoRA (E2) | ~36% (est.) | ~0.74 (est.) |
| Naive FedAvg-LoRA | 88.8% | 0.61 |
| **FedLoRA D_300 smishing** | **67.6%** | **0.6757** |
| Personalized (own category) | 5–25% | — |

LoRA achieves **98.3% of full fine-tune performance** at just **1.09% of parameters**.

---

## Folder Structure

```
FedSmishGuard_Portable/
├── src/
│   ├── shared/
│   │   ├── dataset.py            # SMSDataset + DataLoader factory
│   │   └── fedavg.py             # Weighted FedAvg helpers (LoRA + state-dict)
│   ├── utils.py                  # Shared helpers, paths, constants
│   ├── evaluate.py               # Metric computation (imported by training scripts)
│   ├── train_baseline_tfidf.py   # TF-IDF + LR/SVM baselines (CPU)
│   ├── train_transformer.py      # Centralized DistilBERT / LoRA
│   ├── train_fedlora.py          # Federated LoRA training (E3 local-only / E4 FedLoRA)
│   ├── train_split.py            # Split learning training (E3 local-only / E4 split FedAvg)
│   └── evaluate_on_clean_test.py # Evaluate adapters on clean test
│
├── data/
│   ├── splits/                   # train.csv, val.csv, test_clean.csv
│   ├── clients/setting_D_300/    # 5 client CSVs (best split, smishing floor 300)
│   ├── processed/                # Source distribution, leakage summary
│   ├── sample/                   # 30-row example dataset
│   └── README_DATA.md
│
├── models/
│   ├── fedlora/                  # Best federated LoRA adapter (D_300 smishing)
│   ├── split/                    # Best split learning checkpoint (D_300 smishing)
│   ├── centralized/              # Centralized LoRA adapter (E2)
│   └── README_MODELS.md
│
├── results/                      # Experiment result CSVs
├── reports/RESULTS_COMPLETE.md   # All results in one document
├── docs/WHAT_WAS_TRIED.md        # Method explanations (beginner-friendly)
│
├── README.md
├── PROJECT_SUMMARY.md
├── RUN_INSTRUCTIONS.md
├── requirements.txt
└── .gitignore
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate  |  Linux/Mac: source .venv/bin/activate

# Install PyTorch (pick one):
pip install torch                                                      # CPU
pip install torch --index-url https://download.pytorch.org/whl/cu124  # CUDA 12.4
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128  # RTX 50xx

pip install -r requirements.txt
```

### 2. Run TF-IDF baseline (no GPU needed)

```bash
python src/train_baseline_tfidf.py
```

### 3. Evaluate the included best federated adapter

```bash
python src/evaluate_on_clean_test.py
```

### 4. Train federated model (GPU required)

```bash
python src/train_fedlora.py \
  --rounds 10 --local_epochs 2 --lr 2e-4 \
  --clients_dir data/clients/setting_D_300 \
  --agg_weight smishing
```

### 4a. Train using split learning

```bash
python src/train_split.py \
  --split_layer 3 --rounds 10 --local_epochs 2 --lr 2e-4 \
  --clients_dir data/clients/setting_D_300 \
  --agg_weight smishing
```

See `RUN_INSTRUCTIONS.md` for full reproduction steps from raw data.

---

## Training Arguments

### `train_fedlora.py`

| Argument | Default | Description |
|---|---|---|
| `--rounds` | 10 | Number of federated communication rounds |
| `--local_epochs` | 2 | Local epochs per client per round |
| `--lr` | 2e-4 | Learning rate |
| `--clients_dir` | `data/clients` | Path to client CSV folder |
| `--agg_weight` | `smishing` | Aggregation: `smishing`, `total`, `sqrt`, `balanced`, `uniform` |
| `--local` | flag | Run local-only training (E3, no federation) |
| `--resume` | flag | Resume from saved global adapter checkpoint |

### `train_split.py`

| Argument | Default | Description |
|---|---|---|
| `--split_layer` | 3 | Number of DistilBERT layers kept on the client side |
| `--rounds` | 10 | Number of federated communication rounds |
| `--local_epochs` | 2 | Local epochs per client per round |
| `--lr` | 2e-4 | Learning rate |
| `--clients_dir` | `data/clients` | Path to client CSV folder |
| `--agg_weight` | `smishing` | Aggregation: `smishing`, `total`, `sqrt`, `balanced`, `uniform` |
| `--local` | flag | Run local-only training (E3, no federation) |
| `--resume` | flag | Resume from saved split checkpoint |

Best-checkpoint saving is automatic — the adapter with highest validation F1 is saved to `models/fedlora/global_adapter_{setting}_best/`.

---

## Aggregation Strategies

| Strategy | Formula | Result |
|---|---|---|
| `smishing` | weight ∝ smishing_count | **Best** — use this |
| `total` | weight ∝ total_samples | Fails — smishing signal erased |
| `sqrt` | weight ∝ sqrt(smishing) | Worse — noisy low-resource clients |
| `balanced` | 0.5×total + 0.5×smishing | Worse — dilutes smishing signal |
| `uniform` | equal weights | Not recommended |

---

## Client Split Settings

| Setting | Description |
|---|---|
| A (harsh) | Original — 100% ham/spam overlap (buggy, for reference only) |
| B (balanced) | Zero overlap — genuine Non-IID partitioning |
| C_α (Dirichlet) | Random heterogeneity via Dirichlet(α=0.1/0.3/0.5/1.0) |
| **D_300** | Setting B + smishing floor of 300 per client **(best)** |

---

## Limitations

- Federated model has ~32pp higher FNR than centralized (67.6% vs ~36%)
- Simulation only — not tested on a real federated network
- English SMS only
- Client_4 smishing top-up uses borrowed (non-government_tax) samples
