# RUN INSTRUCTIONS — FedSmishGuard

---

## Hardware

| Component | Recommended | Minimum |
| --- | --- | --- |
| GPU | NVIDIA RTX 4060+ (8GB VRAM) | None (CPU fallback for baselines) |
| RAM | 16GB | 8GB |
| Python | 3.10–3.12 | 3.9+ |

---

## Installation

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# Install PyTorch — pick one:
pip install torch                                                       # CPU only
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA 12.4
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128  # RTX 50xx

pip install -r requirements.txt
```

---

## What You Can Run (data is already included)

### 1. TF-IDF baseline — CPU, no GPU needed

```bash
python src/train_baseline_tfidf.py
```

Trains TF-IDF + Logistic Regression and SVM on `data/splits/train.csv`, evaluates on `test_clean.csv`.  
Results → `results/results_tfidf_*.csv`

---

### 2. Evaluate the included pre-trained adapters

```bash
python src/evaluate_on_clean_test.py
```

Evaluates all adapters found in `models/` on `data/splits/test_clean.csv`.  
Results → `results/results_clean_vs_dirty_test.csv`

---

### 3. Train centralized DistilBERT / LoRA — GPU required

```bash
# E1: Full fine-tune (upper bound) — saves ~256MB to models/distilbert_central/
python src/train_transformer.py

# E2: LoRA only (parameter-efficient) — saves ~3MB to models/distilbert_lora_central/
python src/train_transformer.py --lora
```

Uses `data/splits/train.csv`, `val.csv`, `test_clean.csv`. Best val checkpoint is saved automatically.

---

### 4. Train federated LoRA — GPU required

```bash
python src/train_fedlora.py \
  --rounds 10 \
  --local_epochs 2 \
  --lr 2e-4 \
  --clients_dir data/clients/setting_D_300 \
  --agg_weight smishing
```

### 4a. Train using split learning — GPU required

```bash
python src/train_fedlora.py \
  --split --split_layer 3 --rounds 10 \
  --local_epochs 2 --lr 2e-4 \
  --clients_dir data/clients/setting_D_300 \
  --agg_weight smishing
```

Uses `data/clients/setting_D_300/` (5 client CSVs included).  
Saves final adapter → `models/fedlora/global_adapter_setting_D_300/`  
Saves best val-F1 checkpoint → `models/fedlora/global_adapter_setting_D_300_best/`

**`--agg_weight` options:**

| Value | Formula | Result |
| --- | --- | --- |
| `smishing` | weight ∝ smishing_count | **Best — use this** |
| `total` | weight ∝ total_samples | Fails under Non-IID |
| `sqrt` | weight ∝ √smishing | Underperforms |
| `balanced` | 0.5×total + 0.5×smishing | Underperforms |

---

## Scripts in This Package

| Script | Purpose | Needs GPU |
| --- | --- | --- |
| `src/utils.py` | Shared paths, labels, helpers (imported by others) | No |
| `src/evaluate.py` | Metric computation (imported by others) | No |
| `src/train_baseline_tfidf.py` | TF-IDF + LR/SVM baselines | No |
| `src/train_transformer.py` | Centralized DistilBERT / LoRA | Yes |
| `src/train_fedlora.py` | Federated LoRA training | Yes |
| `src/evaluate_on_clean_test.py` | Evaluate any adapter on clean test | Yes |
