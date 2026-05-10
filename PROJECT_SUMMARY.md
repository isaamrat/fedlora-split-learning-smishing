# FedSmishGuard: Smishing-Aware Federated LoRA for Non-IID SMS Phishing Detection

## Problem

SMS phishing (smishing) is a growing mobile threat where attackers impersonate banks, delivery services, and government agencies. Centralized detection systems require collecting user SMS data on a server — this creates serious privacy risks because SMS messages often contain OTPs, personal conversations, and sensitive financial information.

## Goal

Design a privacy-preserving smishing detection system where mobile devices train locally and share only lightweight model updates (LoRA adapter weights), never raw SMS data. The system must handle real-world **Non-IID conditions** where different clients specialize in different smishing categories.

---

## Dataset

**7 English SMS datasets merged:**

## Client Simulation Design

5 federated clients, each specializing in one smishing category (Non-IID):

| Client | Category | Smishing samples (Setting B) |
|---|---|---|
| client_1 | reward_prize | 1,172 |
| client_2 | delivery_package | 226 |
| client_3 | bank_payment | 1,052 |
| client_4 | government_tax | 72 |
| client_5 | other_smishing | 2,562 |

**Setting D_300:** Clients 2 and 4 receive top-up smishing (from client_5's pool) to reach a floor of 300 smishing each. This prevents data-starvation collapse for low-resource clients.

---

## Methods Tried

| Method | Description |
|---|---|
| TF-IDF + Logistic Regression | Classical baseline (CPU) |
| TF-IDF + Linear SVM | Classical baseline (CPU) |
| Centralized DistilBERT | Full fine-tune, upper bound |
| Centralized DistilBERT + LoRA | Parameter-efficient fine-tune |
| Local-only client training | No federation, clients train independently |
| Naive FedAvg-LoRA | Standard FedAvg weighted by total samples |
| Smishing-weighted FedAvg-LoRA | FedAvg weighted by smishing count per client |
| Sqrt-weighted FedAvg | Weight by sqrt(smishing count) |
| Balanced-weighted FedAvg | 50% total + 50% smishing weight |
| Setting D_300 smishing floor | Enforce min 300 smishing per client via top-up |
| Personalized adapters (E5) | Per-client local fine-tune from global adapter |
| FedSA-LoRA inspired | Aggregate only A matrices, keep B local |
| Dirichlet Non-IID splits | Random heterogeneity via Dirichlet(α) |

---

## Best Results (Clean Test)

| Model | Clean Macro F1 | Clean FNR | Smishing F1 |
|---|---|---|---|
| TF-IDF + SVM (baseline) | ~0.75 (dirty only) | ~36% | 0.658 |
| Centralized DistilBERT (E1) | ~0.74 (est.) | ~31% | ~0.65 |
| Centralized LoRA (E2) | ~0.74 (est.) | ~36% | ~0.64 |
| **FedLoRA D_300 smishing (best)** | **0.6757** | **67.6%** | **0.411** |
| FedLoRA Setting B smishing | 0.6721 | 69.2% | 0.395 |
| Personalized (own category) | — | **5–25%** | 0.72–0.93 |

**FNR = False Negative Rate for smishing** — the percentage of smishing messages that are missed (not detected). Lower is better.

---

## Key Findings

1. **LoRA achieves 98.3% of full fine-tune performance at only 1.09% of parameters** — LoRA is highly parameter-efficient for SMS classification.

2. **Naive FedAvg fails under Non-IID smishing** — FNR=88.8%, worse than local-only (80.5%). Total-sample weighting lets ham/spam-heavy clients dominate and erase smishing signal.

3. **Smishing-weighted FedAvg is essential** — Weighting by each client's smishing count dramatically reduces FNR (60–69% on clean test vs 88.8%).

4. **D_300 smishing floor prevents data-starvation collapse** — Client_4 (government_tax, 72 samples) jumps from val F1=0.547 to 0.711 in round 1 when given a top-up to 300 smishing samples.

5. **Personalized adapters are specialists, not generalists** — Specialists achieve FNR 5–25% on their own category but fail on the global test. This motivates a category-routing inference architecture.

6. **Near-duplicate leakage inflates FNR by +8–10pp** — Without removing near-duplicates, the dirty test overestimates performance. All valid numbers use `test_clean.csv`.

7. **Best-checkpoint saving matters in federated training** — Val F1 oscillates; the final round adapter is often worse than the round-17 adapter. The training script now saves both.

---

## Limitations

- Federated model still misses ~68% of smishing on clean test vs ~36% for centralized — substantial privacy-utility gap remains.
- Client_4 (government_tax) has very few real smishing samples — top-up is borrowed from other categories, not genuine government_tax text.
- Near-duplicate detection may be incomplete — some leakage may remain even in the clean test.
- Personalized adapters were fine-tuned from the Setting A (buggy) global adapter — re-running from D_300 global could improve results.
- Only English SMS tested — multilingual smishing not addressed.
- No real federated network tested — simulation on one device.

---

## Future Work

1. **Warm-start FedLoRA from E2 centralized checkpoint** — Initialize federated training from the already-trained centralized LoRA instead of from scratch. Expected to cut early-round FNR by 20–30pp.

2. **Category-routing inference** — Classify smishing type first, then route to the appropriate specialist adapter. Oracle FNR ≈ 15% weighted average.

3. **FedProx / SCAFFOLD** — Principled Non-IID federated algorithms to reduce client drift and improve convergence stability.

4. **Synthetic minority augmentation** — Generate synthetic government_tax smishing locally to address client_4's data starvation.

5. **DoRA (Weight-Decomposed LoRA)** — Drop-in replacement for LoRA shown to outperform by 1–3pp F1.
