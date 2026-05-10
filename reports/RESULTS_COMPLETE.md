# FedSmishGuard — Complete Results

**Last updated:** 2026-05-08 | **Protocol:** DistilBERT + LoRA (r=8, α=16) | **10 rounds × 2 local epochs × LR=2e-4**

> Always evaluate on `data/splits/test_clean.csv` — the regular test.csv contains near-duplicates that inflate FNR by 8–10pp.

---

## Quick Reference — Best Results (Clean Test)

| Method | Clean Macro F1 | Clean FNR ↓ | Clean Smishing F1 |
|---|---|---|---|
| TF-IDF + SVM (CPU baseline) | ~0.76 (dirty) | ~36% (dirty) | 0.658 |
| Centralized DistilBERT (E1) | 0.781 (dirty) | 30.2% (dirty) | 0.689 |
| Centralized LoRA (E2) | 0.768 (dirty) | 30.4% (dirty) | 0.678 |
| Local-only average (E3) | ~0.621 (dirty) | ~80.5% (dirty) | — |
| Naive FedAvg (E4a) | ~0.612 (dirty) | 88.8% (dirty) | 0.198 |
| FedLoRA Setting B smishing | **0.6721** | **69.2%** | **0.395** |
| **FedLoRA D_300 smishing (BEST)** | **0.6757** | **67.6%** | **0.411** |
| Personalized (own category only) | — | **5–25%** | 0.72–0.93 |

---

## 1. Dataset

**7 datasets merged → 30,846 rows (after exact deduplication)**

| Source | Rows |
|---|---|
| super_sms_dataset | 13,703 |
| combined_labeled_dataset | 4,832 |
| uci_sms_spam | 4,011 |
| mendeley_balanced_10191 | 3,870 |
| exais_sms_spam | 3,651 |
| mendeley_5971 | 728 |
| english_british_sms | 51 |

Labels: ham=15,767 (51.1%) | spam=7,816 (25.4%) | smishing=7,263 (23.6%)  
Exact duplicates removed: 17,004 rows (35.5% of raw combined)  
Splits: train=21,591 | val=3,085 | test=6,170 (70/10/20%, stratified)

**Near-duplicate leakage (MinHash LSH, Jaccard ≥ 0.8):** 2,233 train-test pairs found. 978 test rows removed → `test_clean.csv` (5,192 rows). Smishing most affected: 500/1,453 smishing test rows (34%) removed.

| Model | Dirty FNR | Clean FNR | Inflation |
|---|---|---|---|
| E4b D_300 smishing | 58.3% | 67.6% | +9.3pp |
| E4b D_300 balanced | 62.3% | 72.1% | +9.8pp |
| E4b Setting B smishing | 60.6% | 69.2% | +8.5pp |
| E4b Setting B balanced | 73.4% | 82.3% | +8.9pp |
| E4b Setting B sqrt | 73.8% | 82.4% | +8.6pp |
| E4b Setting A (bug) | 68.0% | 74.5% | +6.5pp |

---

## 2. Client Split Design

### Setting B (main corrected Non-IID setting)

| Client | Category | Ham | Spam | Smishing |
|---|---|---|---|---|
| client_1 | reward_prize | 2,207 | 1,094 | 1,172 |
| client_2 | delivery_package | 2,207 | 1,094 | 226 |
| client_3 | bank_payment | 2,207 | 1,094 | 1,052 |
| client_4 | government_tax | 2,207 | 1,094 | 72 |
| client_5 | other_smishing | 2,207 | 1,094 | 2,562 |

Ham/spam fully partitioned (zero overlap between clients). Smishing partitioned by category.

### Setting D_300 (best — smishing floor of 300 per client)

Clients 2 and 4 receive top-up from client_5's other_smishing pool:  
client_2: 226 delivery + 74 other = 300 | client_4: 72 gov_tax + 228 other = 300 | client_5: 2,562 → 2,260 (donated 302)

---

## 3. Baseline Results

### E0 — Classical (CPU, dirty test)

| Model | Accuracy | Macro F1 | Smishing F1 | Smishing FNR |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.8084 | 0.7533 | 0.656 | 35.7% |
| TF-IDF + Linear SVM | 0.8183 | 0.7613 | 0.658 | 36.1% |

### E1 — Centralized DistilBERT (dirty test, upper bound)

Accuracy=0.8335 | Macro F1=**0.7810** | Smishing F1=0.689 | FNR=**30.2%** | Trainable params: 66.9M

### E2 — Centralized DistilBERT + LoRA (dirty test)

Accuracy=0.8233 | Macro F1=**0.7681** | Smishing F1=0.678 | FNR=**30.4%** | Trainable params: **740K / 67.7M (1.09%)**

> LoRA achieves **98.3% of full fine-tune Macro F1** at only **1.09% of parameters**. Adapter size: 2.96MB vs 256MB.

### E3 — Local-Only (Setting A ⚠ ham/spam overlap bug, dirty test)

| Client | Smishing samples | Macro F1 | Smishing FNR |
|---|---|---|---|
| client_1 reward_prize | 1,172 | 0.664 | 77.4% |
| client_2 delivery_package | 226 | 0.567 | 95.8% |
| client_3 bank_payment | 1,052 | 0.638 | 78.7% |
| client_4 government_tax | 72 | 0.557 | 97.5% |
| client_5 other_smishing | 2,562 | 0.679 | 52.9% |
| **Average** | — | **0.621** | **80.5%** |

---

## 4. Federated Results

### E4a — Naive FedAvg (total-sample weighted, Setting A ⚠)

Macro F1=0.612 | Smishing F1=0.198 | FNR=**88.8%** | FPR=0.7%  
**Worse than local-only.** Total-sample weighting lets ham/spam-dominant clients erase smishing signal.

### E4b — All Aggregation Variants (dirty test)

| Setting | Aggregation | Rounds | Dirty Macro F1 | Dirty FNR | Clean Macro F1 | Clean FNR | Clean Smishing F1 |
|---|---|---|---|---|---|---|---|
| A (bug) | smishing | 10 | 0.6884 | 68.0% | 0.6580 | 74.5% | 0.350 |
| B | smishing | 10 | **0.7077** | 60.6% | **0.6721** | 69.2% | 0.395 |
| B | sqrt | 10 | 0.6751 | 73.8% | 0.6373 | 82.4% | 0.271 |
| B | balanced | 10 | 0.6780 | 73.4% | 0.6379 | 82.3% | 0.271 |
| **D_300** | **smishing** | **10** | **0.7135** | **58.3%** | **0.6757** | **67.6%** | **0.411** |
| D_300 | balanced | 10 | 0.7084 | 62.3% | 0.6672 | 72.1% | 0.375 |
| D_300 | smishing | 20 (final R20) | 0.7000 | 60.8% | 0.6732 | 66.8% | 0.409 |
| C_0.5 | smishing | 10 | 0.6003 | 2.8% | 0.5892 | 4.0% | 0.628 |
| C_1.0 | smishing | 10 | 0.6345 | 6.5% | 0.6137 | 8.4% | 0.631 |

> C_0.5 / C_1.0: FNR is very low but spam F1 collapsed (0.162 / 0.259) — model calls everything smishing.

**Val F1 trajectory — Setting B smishing (10 rounds):**  
0.631 → 0.680 → 0.688 → 0.695 → 0.700 → 0.711 → **0.722** → 0.714 → 0.713 → 0.705

**Val F1 trajectory — Setting D_300 smishing (20 rounds):**  
0.626 → 0.697 → 0.712 → 0.699 → 0.701 → 0.717 → 0.710 → 0.707 → 0.710 → 0.716 → 0.712 → 0.713 → 0.714 → 0.709 → 0.716 → 0.715 → **0.718** → 0.711 → 0.713 → 0.704

Best val checkpoint at round 17 (val FNR=54.7%). Script now saves `global_adapter_{setting}_best` automatically.

### FedSA-LoRA (A-matrix only, Setting B)

Macro F1=0.612 | FNR=83.7% | Smishing F1=0.341  
B matrices initialized to zero, kept local — model cannot represent any transformation. Full LoRA sharing required.

---

## 5. Personalization Results (E5)

**Warm-started from E4b Setting A global adapter. Evaluated on per-category test sets.**

| Client | Category | Cat Macro F1 | Cat Smishing F1 | **Cat FNR** | Global FNR (wrong eval) |
|---|---|---|---|---|---|
| client_1 | reward_prize | 0.930 | 0.926 | **5.25%** | 76.5% |
| client_2 | delivery_package | 0.934 | 0.923 | **12.9%** | 93.7% |
| client_3 | bank_payment | 0.911 | 0.893 | **7.72%** | 76.6% |
| client_4 | government_tax | 0.905 | 0.857 | **25.0%** | 93.7% |
| client_5 | other_smishing | 0.788 | 0.719 | **23.75%** | 56.4% |

**Key finding:** Specialists achieve FNR 5–25% on their own category — near centralized performance — but fail on the global test because they are category specialists, not generalists.

**Proposed architecture:** Category classifier (TF-IDF/SVM) → route to specialist adapter.  
Oracle FNR (perfect routing) ≈ **15.7%** weighted average.

**Limitation:** Adapters used Setting A (buggy) as warm-start. Re-running from Setting B/D_300 global may improve results further.

---

## 6. Aggregation Strategy Verdict

| Strategy | Formula | Setting B Clean FNR | D_300 Clean FNR | Verdict |
|---|---|---|---|---|
| **smishing** | weight ∝ smishing_count | **69.2%** | **67.6%** | **Use this** |
| balanced | 0.5×total + 0.5×smishing | 82.3% | 72.1% | Rejected |
| sqrt | weight ∝ √smishing | 82.4% | — | Rejected |
| total (naive) | weight ∝ total_samples | ~88.8% | — | Rejected |
