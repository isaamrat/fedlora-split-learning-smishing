# Classical Models Guide — FedSmishGuard

**CNN, LSTM, BiLSTM, GRU, BiGRU, CNN-LSTM with Federated Learning and Split Learning**

This guide covers the new classical deep learning models added alongside the existing DistilBERT+LoRA baseline.  
All models run under the same federated setup (5 clients, D_300 split, smishing-weighted FedAvg) for a fair comparison.

---

## Quick Overview

| Mode | Script | Description |
|---|---|---|
| **FedAvg** | `train_classical_fed.py` | Full model weights aggregated each round |
| **Split learning** | `train_classical_split.py` | Model split between client and server |
| **DistilBERT + LoRA** | `train_fedlora.py` | Original experiment (unchanged) |
| **DistilBERT split** | `train_split.py` | Original split experiment (unchanged) |

---

## Step 0 — Prerequisites

### Install dependencies (if not done yet)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA 12.4
pip install -r requirements.txt
```

---

## Step 1 — Build the Vocabulary (run once)

The classical models use a word-index vocabulary instead of the BERT tokenizer.  
Build it from the training data:

```bash
python src/shared/build_vocab.py
```

Output: `data/vocab.json`  (~30,000 entries, built in seconds)

**Options:**
```bash
python src/shared/build_vocab.py --max_vocab 20000   # smaller vocab
python src/shared/build_vocab.py --min_freq 2         # exclude rare words
```

---

## Step 2 — (Optional) Download GloVe Embeddings

GloVe pretrained embeddings significantly improve performance.  
Without them, random embeddings are used (models still train, but slower to converge).

```bash
# Download and extract into data/
cd data
wget https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
unzip glove.6B.zip glove.6B.100d.txt
cd ..
```

Expected file: `data/glove.6B.100d.txt` (~347 MB)

> If GloVe is not present, the scripts print a notice and continue with random embeddings automatically.

---

## Step 3 — Federated Learning (FedAvg)

### Run all models with default settings

```bash
# GRU (fastest — start here)
python src/train_classical_fed.py --model gru \
  --rounds 10 --local_epochs 2 --lr 1e-3 \
  --clients_dir data/clients/setting_D_300 \
  --agg_weight smishing

# BiGRU
python src/train_classical_fed.py --model bigru \
  --rounds 10 --local_epochs 2 \
  --clients_dir data/clients/setting_D_300

# LSTM
python src/train_classical_fed.py --model lstm \
  --rounds 10 --local_epochs 2 \
  --clients_dir data/clients/setting_D_300

# BiLSTM
python src/train_classical_fed.py --model bilstm \
  --rounds 10 --local_epochs 2 \
  --clients_dir data/clients/setting_D_300

# TextCNN
python src/train_classical_fed.py --model cnn \
  --rounds 10 --local_epochs 2 \
  --clients_dir data/clients/setting_D_300

# CNN-LSTM (hybrid)
python src/train_classical_fed.py --model cnn_lstm \
  --rounds 10 --local_epochs 2 \
  --clients_dir data/clients/setting_D_300
```

### Local-only mode (no federation — ablation)

Add `--local` to any command above:

```bash
python src/train_classical_fed.py --model gru --local \
  --rounds 10 --local_epochs 2 \
  --clients_dir data/clients/setting_D_300
```

---

## Step 4 — Split Learning

### Split point reference

| Model | Split A (all models) | Split B (cnn_lstm only) |
|---|---|---|
| `cnn` | After embedding → server does Conv + Classify | — |
| `lstm` | After embedding → server does LSTM + Classify | — |
| `bilstm` | After embedding → server does BiLSTM + Classify | — |
| `gru` | After embedding → server does GRU + Classify | — |
| `bigru` | After embedding → server does BiGRU + Classify | — |
| `cnn_lstm` | After embedding → server does CNN+LSTM+Classify | After CNN → server does LSTM+Classify (recommended) |

### Run split learning

```bash
# GRU split (split A — only option)
python src/train_classical_split.py --model gru \
  --rounds 10 --local_epochs 2 \
  --clients_dir data/clients/setting_D_300

# BiLSTM split
python src/train_classical_split.py --model bilstm \
  --rounds 10 --local_epochs 2 \
  --clients_dir data/clients/setting_D_300

# CNN-LSTM split B (recommended — most semantically meaningful)
python src/train_classical_split.py --model cnn_lstm --split_point B \
  --rounds 10 --local_epochs 2 \
  --clients_dir data/clients/setting_D_300

# CNN-LSTM split A (for comparison)
python src/train_classical_split.py --model cnn_lstm --split_point A \
  --rounds 10 --local_epochs 2 \
  --clients_dir data/clients/setting_D_300
```

### Local-only split mode

```bash
python src/train_classical_split.py --model lstm --local \
  --rounds 10 --local_epochs 2 \
  --clients_dir data/clients/setting_D_300
```

---

## All Arguments Reference

### `train_classical_fed.py`

| Argument | Default | Description |
|---|---|---|
| `--model` | *required* | `cnn` / `lstm` / `bilstm` / `gru` / `bigru` / `cnn_lstm` |
| `--local` | flag | Local-only mode — no federation |
| `--rounds` | 10 | Federated communication rounds |
| `--local_epochs` | 1 | Local training epochs per client per round |
| `--lr` | 1e-3 | Learning rate |
| `--embed_dim` | 100 | Embedding dimension (must match GloVe file) |
| `--hidden_dim` | 256 | LSTM/GRU hidden state dimension |
| `--num_filters` | 128 | CNN filters (for `cnn` and `cnn_lstm`) |
| `--no_glove` | flag | Use random embeddings (skip GloVe) |
| `--clients_dir` | `data/clients` | Path to client CSV folder |
| `--agg_weight` | `smishing` | `smishing` / `total` / `sqrt` / `balanced` / `uniform` |
| `--resume` | flag | Resume from saved checkpoint |

### `train_classical_split.py`

All of the above, plus:

| Argument | Default | Description |
|---|---|---|
| `--split_point` | `A` | `A` (after embedding) / `B` (after CNN — `cnn_lstm` only) |

---

## Outputs

| Path | Content |
|---|---|
| `models/classical/<model>_<setting>/model_state.pt` | Final trained model weights |
| `models/classical/<model>_<setting>_best/model_state.pt` | Best val-F1 checkpoint |
| `models/classical_split/<model>_split<A/B>_<setting>/` | Split model client+server states |
| `reports/results_<model>_fed_rounds.csv` | Per-round metrics (FedAvg) |
| `reports/results_<model>_split<A/B>_rounds.csv` | Per-round metrics (split) |
| `reports/results_<tag>.csv` | Final test metrics |
| `reports/RESULTS.md` | Appended results summary |
| `reports/figures/<model>_fed_round_f1.png` | Macro F1 vs round plot |

---

## Full Comparison Run (all experiments)

Run this sequence to populate the complete comparison table:

```bash
# 1. Build vocabulary (once)
python src/shared/build_vocab.py

# 2. DistilBERT baselines (already trained — evaluate if adapters exist)
python src/evaluate_on_clean_test.py

# 3. Classical FedAvg
for model in gru bigru lstm bilstm cnn cnn_lstm; do
  python src/train_classical_fed.py --model $model \
    --rounds 10 --local_epochs 2 \
    --clients_dir data/clients/setting_D_300 \
    --agg_weight smishing
done

# 4. Classical split learning
for model in gru bigru lstm bilstm cnn; do
  python src/train_classical_split.py --model $model \
    --rounds 10 --local_epochs 2 \
    --clients_dir data/clients/setting_D_300
done

# CNN-LSTM split B (recommended split point)
python src/train_classical_split.py --model cnn_lstm --split_point B \
  --rounds 10 --local_epochs 2 \
  --clients_dir data/clients/setting_D_300
```

---

## Expected Results

> Approximate ranges based on similar SMS classification literature.
> Actual results depend on data distribution and hyperparameter tuning.

| Model | Mode | Expected Macro F1 | Expected FNR |
|---|---|---|---|
| TF-IDF + SVM | baseline | ~0.76 | ~36% |
| DistilBERT + LoRA | FedAvg | 0.6757 | 67.6% |
| TextCNN | FedAvg | 0.68–0.74 | 45–55% |
| LSTM | FedAvg | 0.70–0.75 | 40–50% |
| BiLSTM | FedAvg | 0.71–0.76 | 38–48% |
| GRU | FedAvg | 0.70–0.75 | 40–50% |
| BiGRU | FedAvg | 0.71–0.76 | 38–48% |
| CNN-LSTM | FedAvg | 0.70–0.75 | 40–50% |
| CNN-LSTM | Split-B | 0.69–0.74 | 42–52% |

> **Key research question**: Can any classical model match or beat DistilBERT+LoRA FedAvg
> (F1=0.6757, FNR=67.6%) under identical Non-IID federated conditions?

---

## File Structure Reference

```
src/
├── models/
│   ├── __init__.py
│   ├── textcnn.py           # TextCNN full + split (Client/Server)
│   ├── lstm.py              # LSTM + BiLSTM full + split
│   ├── gru.py               # GRU + BiGRU full + split
│   └── cnn_lstm.py          # CNN-LSTM full + split (points A and B)
│
├── shared/
│   ├── build_vocab.py       # One-off vocab builder -> data/vocab.json
│   ├── vocab_dataset.py     # Word-index Dataset, GloVe loader, make_vocab_loader
│   ├── dataset.py           # BERT tokenizer dataset (DistilBERT scripts)
│   └── fedavg.py            # Weighted FedAvg helpers (shared by all scripts)
│
├── train_classical_fed.py   # FedAvg for CNN/LSTM/GRU/BiLSTM/BiGRU/CNN-LSTM
├── train_classical_split.py # Split learning for the same 6 models
├── train_fedlora.py         # DistilBERT + LoRA FedAvg (unchanged)
└── train_split.py           # DistilBERT split learning (unchanged)
```
