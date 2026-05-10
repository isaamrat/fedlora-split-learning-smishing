# Models Directory — FedSmishGuard

## What Is Included

| Path | Size | Description |
|---|---|---|
| `fedlora/` | ~3.6MB | Best federated LoRA adapter (D_300 smishing, 20-round best checkpoint) |
| `centralized/` | ~3.6MB | Centralized LoRA adapter (E2 — DistilBERT + LoRA, full training set) |

Both adapters are **PEFT LoRA adapters** for `distilbert-base-uncased`. They do not include the base model weights (267MB) — those are downloaded automatically from HuggingFace on first use.

## Adapter Configuration

```json
{
  "base_model_name_or_path": "distilbert-base-uncased",
  "r": 8,
  "lora_alpha": 16,
  "target_modules": ["q_lin", "v_lin"],
  "lora_dropout": 0.1,
  "task_type": "SEQ_CLS",
  "num_labels": 3
}
```

- **Trainable params:** 740,355 / 67,696,134 (1.09%)
- **Labels:** 0=ham, 1=spam, 2=smishing

## Loading the Adapter

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = PROJECT_ROOT / "models" / "fedlora"

base_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3,
    id2label={0: "ham", 1: "spam", 2: "smishing"},
    label2id={"ham": 0, "spam": 1, "smishing": 2},
    ignore_mismatched_sizes=True,
)
model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH), is_trainable=False)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Predict
inputs = tokenizer(["FREE prize! Click here now"], return_tensors="pt", truncation=True, max_length=128)
logits = model(**inputs).logits
label = ["ham", "spam", "smishing"][logits.argmax(-1).item()]
```

## What Is NOT Included

| Model | Size | Reason | How to Reproduce |
|---|---|---|---|
| `distilbert_central/` | 256MB | Full DistilBERT fine-tune weights | Run `train_transformer.py --mode full` |
| `fedlora/global_adapter_setting_B/` | 3.5MB | Setting B global adapter | Run `train_fedlora.py --clients_dir data/clients/setting_B` |
| `fedlora/client_*_personalized_adapter/` | 2.8MB × 5 | Personalized specialist adapters | Run `personalize_fedlora.py` |
| `fedlora/global_adapter_setting_D_300/` | 3.5MB | 20-round final (round-20 checkpoint) | Run 20-round training |

## Reproducing the Best Federated Adapter

```bash
# 20-round D_300 smishing run (best-checkpoint auto-saved)
python src/train_fedlora.py \
  --rounds 20 \
  --local_epochs 2 \
  --lr 2e-4 \
  --clients_dir data/clients/setting_D_300 \
  --agg_weight smishing

# Best val-F1 checkpoint: models/fedlora/global_adapter_setting_D_300_best/
# Final round checkpoint: models/fedlora/global_adapter_setting_D_300/
```

## Expected Results (Clean Test)

| Adapter | Clean Macro F1 | Clean FNR | Clean Smishing F1 |
|---|---|---|---|
| `centralized/` (E2) | ~0.74 (est.) | ~36% (est.) | ~0.64 (est.) |
| `fedlora/` (D_300 best) | 0.6757 | 67.6% | 0.411 |

> Note: Centralized clean-test numbers are estimates — E2 was not formally evaluated on test_clean.csv. Run `evaluate_on_clean_test.py` to compute exact numbers after adding the centralized adapter to the models list.
