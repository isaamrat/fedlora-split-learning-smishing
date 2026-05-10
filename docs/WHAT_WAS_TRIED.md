# What Was Tried — FedSmishGuard

A beginner-friendly explanation of every method attempted, what happened, and why.

---

## 1. Centralized DistilBERT (E1)

**What it means:**  
All 21,591 training SMS messages are collected on one server. A pre-trained language model (DistilBERT, trained on Wikipedia and BookCorpus) is fine-tuned on this combined dataset. Every single parameter (66.9 million) gets updated during training.

**Why we used it:**  
This is the theoretical upper bound — the best possible result when privacy is not a concern and all data is available. Everything else will be compared against this.

**Result:**  
Macro F1 = 0.781 | Smishing FNR = 30.2%  
The model correctly identifies about 70% of smishing messages.

**Limitation:**  
Requires all user SMS data in one place. Privacy-violating and not deployable in the real world. Also uses 66.9M trainable parameters — expensive to store and update.

---

## 2. Centralized DistilBERT + LoRA (E2)

**What it means:**  
Same as E1 but instead of fine-tuning all 66.9M parameters, we freeze the base model and add small "adapter" matrices (A and B) to the attention layers. Only these 740K parameters are trained and stored.

LoRA (Low-Rank Adaptation) works by approximating the weight update ΔW as a product of two small matrices: ΔW = A × B, where the rank r=8 is much smaller than the full matrix size.

**Why it's useful:**  
- 98.3% of E1 performance at only 1.09% of the parameters
- Adapter file is 2.96MB instead of 256MB
- In federated learning, clients send adapter weights each round — smaller = faster

**Result:**  
Macro F1 = 0.768 | Smishing FNR = 30.4%  
Nearly matches full fine-tuning.

**Why 1.09% parameter result matters:**  
In federated learning, communication cost is a major bottleneck. Sending 2.96MB per client instead of 256MB per client makes federation practical.

---

## 3. Local-Only Client Training (E3)

**What it means:**  
Each of the 5 clients trains independently on only its own local data. There is no server aggregation — no federation. Client_1 only sees reward_prize smishing, client_4 only sees government_tax smishing.

**Why it fails:**  
Each client has very few smishing samples:
- client_4 (government_tax): only 72 smishing messages
- client_2 (delivery_package): only 226 smishing messages

With so little data, the model cannot learn a robust classifier. Client_4's smishing FNR = 97.5% — it misses nearly every smishing message. The average FNR across all clients = 80.5%.

**Key finding:**  
Local-only training is severely limited by data starvation. Federation is needed to share knowledge across clients.

---

## 4. Naive FedAvg-LoRA (E4a)

**What it means:**  
A simple federation protocol: each client trains locally, then sends adapter weights to the server. The server takes a weighted average of all adapters, where the weight for each client is proportional to its **total sample count** (ham + spam + smishing combined). The averaged adapter becomes the new global model for the next round.

**Why it fails:**  
Each client has: ham=2,207 + spam=1,094 + smishing=(72 to 2,562)  
When weighted by total samples, each client gets nearly equal weight (because ham and spam dominate).  
Client_4 (72 smishing, 3,373 total) gets weight = 3,373/total ≈ the same as client_1 (1,172 smishing, 4,473 total).

But client_4 learned almost nothing about smishing (72 samples!). Its adapter actively hurts the aggregation — it pushes the global model away from smishing detection.

**Result:**  
Macro F1 = 0.612 | Smishing FNR = 88.8%  
**Worse than local-only training** (80.5% FNR). Federation made things worse.

**Key insight:**  
The aggregation formula must account for how much smishing knowledge each client contributes, not just how much data they have.

---

## 5. Smishing-Weighted FedAvg-LoRA (E4b)

**What it means:**  
Same as E4a, but now the aggregation weight for each client is proportional to its **smishing sample count** only.

```
weight_client_i = smishing_count_i / sum(all smishing counts)
```

client_5 (2,562 smishing) → gets much more weight than client_4 (72 smishing).  
This means the global model learns more from clients who have strong smishing signal.

**Why it improved:**  
Clients with more smishing data have more reliable smishing adapters. Weighting by smishing count lets the global model learn from clients who actually know smishing well.

**Result (Setting B, clean test):**  
Macro F1 = 0.6721 | Smishing FNR = 69.2%  
Much better than naive FedAvg (88.8% → 69.2%).

---

## 6. Sqrt Weighting

**What it means:**  
Instead of linear smishing weighting, use the square root: `weight_i ∝ sqrt(smishing_i)`.  
This compresses the gap between high-smishing and low-smishing clients — client_5 (2,562 smishing, sqrt=50.6) gets less advantage over client_4 (72 smishing, sqrt=8.5) than with linear weighting.

**Why it failed:**  
The intent was to give client_2 and client_4 more influence so they could contribute category-specific knowledge. But in practice, those clients have weak smishing adapters (due to few samples). Giving them more weight introduces noise into the aggregation.

**Result (Setting B, clean test):**  
Macro F1 = 0.6373 | Smishing FNR = 82.4%  
Significantly worse than linear smishing weighting.

**Lesson:**  
More weight to weak clients = more noise. Data-poor clients need more data (D_300), not more aggregation influence.

---

## 7. Balanced Weighting

**What it means:**  
A compromise between total-sample weighting and smishing-only weighting:  
`weight_i = 0.5 × (total_i / total_all) + 0.5 × (smishing_i / smishing_all)`

This gives each client partial credit for their total data volume and partial credit for their smishing data.

**Why it underperformed:**  
Even with the smishing floor (Setting D_300), adding total-sample weight into the formula dilutes the smishing-focused signal. The total-sample component re-introduces the problem from naive FedAvg — it gives proportional weight to ham/spam data that doesn't help with smishing detection.

**Result (Setting D_300, dirty test):**  
Macro F1 = 0.7084 | FNR = 62.3%  
vs. smishing-only: Macro F1 = 0.7135 | FNR = 58.3%  
Worse across the board.

**Lesson:**  
For the smishing-detection objective, pure smishing-weighted aggregation is the right formula. Mixing in total-sample weight adds irrelevant signal.

---

## 8. D_300 Smishing Floor

**What it means:**  
Before training, any client with fewer than 300 smishing samples receives "top-up" smishing messages from client_5's other_smishing pool. This ensures all clients contribute meaningful smishing signal to the aggregation.

- client_2: 226 delivery_package smishing → tops up to 300 (74 from other_smishing)
- client_4: 72 government_tax smishing → tops up to 300 (228 from other_smishing)
- Others: already above 300, unchanged

**Why it helped:**  
With 72 smishing samples, client_4's adapter was essentially random for smishing detection. FNR = 97.5% locally. With 300 samples, the adapter learns real smishing patterns. In round 1 of federation, client_4's val F1 jumps from 0.547 to 0.711 — the floor immediately fixes data-starvation.

**Result (Setting D_300, clean test):**  
Macro F1 = 0.6757 | Smishing FNR = 67.6%  
Best federated result in the study.

**Key insight:**  
Data balance across clients is more impactful than aggregation formula choice. Fix the data first.

---

## 9. Dirichlet Non-IID Client Splits (Setting C)

**What it means:**  
Instead of assigning clients by smishing category, labels are distributed randomly using a Dirichlet distribution with concentration parameter α. Low α (0.1, 0.3) creates extreme heterogeneity — some clients see almost only smishing, others almost none. High α (1.0) approaches uniform distribution.

**Why spam collapsed:**  
With α=0.5, the smishing-weighted FedAvg gives overwhelming weight to clients who happen to have extreme smishing concentration. These clients' adapters dominate the aggregation. But those clients learned almost nothing about spam (because Dirichlet gave them very little spam data). The global model collapses spam detection.

**Result (α=0.5):**  
Macro F1 = 0.6003 | FNR = 2.82% | Spam F1 = 0.162  
FNR is very low (almost no smishing missed) but spam F1 collapses — the model calls everything smishing.

**Lesson:**  
Extreme Non-IID + smishing-weighted aggregation creates an imbalanced global model. The category-based partitioning (Settings B/D) is more controlled and produces better results.

---

## 10. Dirty vs Clean Test

**What they mean:**  
- **Dirty test** (`test.csv`, 6,170 rows): The original test split. Contains near-duplicate rows that also appear in the training set.
- **Clean test** (`test_clean.csv`, 5,192 rows): Dirty test with 978 rows removed — those that were near-duplicates (Jaccard ≥ 0.8) of training examples.

**Why clean test is more honest:**  
Near-duplicate smishing messages in both train and test inflate performance. The model has "seen" the test examples before (in a nearly identical form). This makes FNR look artificially low — the model appears better than it actually is on unseen data.

**The inflation is concentrated in smishing:** 500 of the 1,453 smishing test rows (34%) were near-duplicates. Removing them makes the smishing evaluation much harder.

| Model | Dirty FNR | Clean FNR | Inflation |
|---|---|---|---|
| D_300 smishing | 58.3% | 67.6% | +9.3pp |
| Setting B smishing | 60.6% | 69.2% | +8.5pp |

All published results from this project use the clean test.

---

## Final Project Story

**Starting point:** Naive FedAvg-LoRA is worse than local training (FNR=88.8% vs 80.5%). The aggregation formula kills smishing detection because it gives equal weight to data-poor clients with terrible smishing adapters.

**Fix 1 — Smishing-weighted aggregation:** Weight each client's adapter by their smishing count. FNR drops from 88.8% to 69.2% on clean test. The right clients (those with smishing knowledge) now dominate the aggregation.

**Fix 2 — D_300 smishing floor:** Enforce a minimum of 300 smishing per client. Client_4's local val F1 jumps from 0.547 to 0.711 in round 1. Global FNR drops to 67.6% clean.

**Hidden finding — personalization:** Specialist adapters achieve FNR 5–25% on their own category. The global model is not the final product — it's the foundation for category-specific specialists.

**Honesty fix — clean test:** Removing near-duplicates shows the true generalization gap. The federated model misses 68% of smishing, not 58%.

**The gap that remains:** Federated (67.6% FNR) vs centralized (~36% FNR). The next step is to warm-start federated training from the centralized LoRA checkpoint, and implement category-routing inference to approach the oracle FNR of ~15%.