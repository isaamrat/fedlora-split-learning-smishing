# Current Split Learning Architecture

This document describes the architecture currently implemented when running:

```bash
python src/train_fedlora.py --split --split_layer 3 ...
```

Important: in the current code, split learning does **not** use LoRA adapters.

```text
Current split mode = Split DistilBERT + FedAvg
Current split mode != Split LoRA
```

## One-Line Summary

The model is a full DistilBERT classifier split into two parts:

```text
Client side: embeddings + first N DistilBERT transformer layers
Server side: remaining DistilBERT layers + classification head
```

After each client trains locally, the server averages both:

```text
1. client-side model weights
2. server-side model weights
```

## Full Current Architecture

```mermaid
flowchart LR
  subgraph Client["Client Device / Client Simulation"]
    SMS["Raw SMS text"]
    TOK["Tokenizer\nDistilBERT tokenizer\nmax_len = 128"]
    IDS["input_ids\nattention_mask"]
    EMB["DistilBERT embeddings"]
    L1["Transformer layer 1"]
    L2["Transformer layer 2"]
    L3["Transformer layer 3\nif split_layer = 3"]
    H["Smashed data\nhidden states"]

    SMS --> TOK
    TOK --> IDS
    IDS --> EMB
    EMB --> L1
    L1 --> L2
    L2 --> L3
    L3 --> H
  end

  subgraph Server["Server-Side Model"]
    L4["Transformer layer 4"]
    L5["Transformer layer 5"]
    L6["Transformer layer 6"]
    CLS["CLS representation\nhidden[:, 0]"]
    PC["pre_classifier"]
    DROP["dropout"]
    HEAD["classifier\n3 output logits"]
    PRED["Prediction\nham | spam | smishing"]

    L4 --> L5
    L5 --> L6
    L6 --> CLS
    CLS --> PC
    PC --> DROP
    DROP --> HEAD
    HEAD --> PRED
  end

  H --> L4
```

For `--split_layer 3`, the first 3 transformer layers stay on the client side and the remaining 3 transformer layers stay on the server side.

## DistilBERT Layer Split

```mermaid
flowchart TB
  subgraph Full["Full DistilBERT Sequence Classifier"]
    E["Embeddings"]
    T1["Transformer 1"]
    T2["Transformer 2"]
    T3["Transformer 3"]
    T4["Transformer 4"]
    T5["Transformer 5"]
    T6["Transformer 6"]
    C["Classification head"]
  end

  subgraph ClientPart["Client-side partition"]
    CE["Embeddings"]
    CT1["Transformer 1"]
    CT2["Transformer 2"]
    CT3["Transformer 3"]
  end

  subgraph ServerPart["Server-side partition"]
    ST4["Transformer 4"]
    ST5["Transformer 5"]
    ST6["Transformer 6"]
    SC["Classification head"]
  end

  E -. copied to .-> CE
  T1 -. copied to .-> CT1
  T2 -. copied to .-> CT2
  T3 -. copied to .-> CT3
  T4 -. copied to .-> ST4
  T5 -. copied to .-> ST5
  T6 -. copied to .-> ST6
  C -. copied to .-> SC
```

## Five-Client Training Round

In one federated round, all five clients start from the same global split model.

```mermaid
flowchart TB
  subgraph GlobalStart["Global split model at start of round"]
    GC0["Global client-side weights"]
    GS0["Global server-side weights"]
  end

  subgraph Clients["Five Clients"]
    C1["client_1\ntrain local split pair"]
    C2["client_2\ntrain local split pair"]
    C3["client_3\ntrain local split pair"]
    C4["client_4\ntrain local split pair"]
    C5["client_5\ntrain local split pair"]
  end

  GC0 --> C1
  GS0 --> C1
  GC0 --> C2
  GS0 --> C2
  GC0 --> C3
  GS0 --> C3
  GC0 --> C4
  GS0 --> C4
  GC0 --> C5
  GS0 --> C5

  C1 --> CW["Client-side state dicts"]
  C2 --> CW
  C3 --> CW
  C4 --> CW
  C5 --> CW

  C1 --> SW["Server-side state dicts"]
  C2 --> SW
  C3 --> SW
  C4 --> SW
  C5 --> SW

  CW --> AVG_C["Weighted FedAvg\nclient-side weights"]
  SW --> AVG_S["Weighted FedAvg\nserver-side weights"]

  AVG_C --> GC1["Updated global client-side model"]
  AVG_S --> GS1["Updated global server-side model"]

  GC1 --> VAL["Validate global split model"]
  GS1 --> VAL
```

## Forward Pass During Local Split Training

```mermaid
sequenceDiagram
  participant D as Client Data
  participant C as Client-side Model
  participant S as Server-side Model
  participant L as Loss Function

  D->>C: input_ids + attention_mask
  C->>C: embeddings
  C->>C: first split_layer transformer blocks
  C->>S: hidden states / smashed data
  S->>S: remaining transformer blocks
  S->>S: classifier head
  S->>L: logits
  L->>L: weighted cross-entropy
```

## Backward Pass During Local Split Training

```mermaid
sequenceDiagram
  participant L as Loss Function
  participant S as Server-side Model
  participant C as Client-side Model
  participant O as Optimizer

  L-->>S: gradients for classifier and server layers
  S-->>C: gradients through split hidden states
  C-->>C: gradients for client layers
  O->>S: update server-side parameters
  O->>C: update client-side parameters
```

In this project, the split training is simulated on one machine/GPU, so the gradient transfer is conceptual rather than a real network call.

## What Is Updated?

Current split learning updates the full split DistilBERT model:

| Component | Location | Updated in split mode? |
|---|---|---|
| Embeddings | Client side | Yes |
| Early transformer layers | Client side | Yes |
| Later transformer layers | Server side | Yes |
| Pre-classifier | Server side | Yes |
| Classifier | Server side | Yes |
| LoRA adapters | Not used in split mode | No |

That is why the log shows something like:

```text
Split model params: 66,955,779 / 66,955,779 (100.00%)
```

It means all split model parameters are trainable.

## Aggregation Step

The current split mode uses `fedavg_state_dict()` twice:

```text
fedavg_state_dict(global_client_model, client_states, ...)
fedavg_state_dict(global_server_model, server_states, ...)
```

So aggregation happens separately for the two halves:

```mermaid
flowchart LR
  CSD["5 client-side state dicts"] --> CA["Weighted average"]
  SSD["5 server-side state dicts"] --> SA["Weighted average"]

  CA --> GCM["New global client-side model"]
  SA --> GSM["New global server-side model"]

  GCM --> GM["Global split model"]
  GSM --> GM
```

## Checkpoints

The final split model is saved to:

```text
models/split/{setting_name}/
```

The best validation checkpoint is saved to:

```text
models/split/{setting_name}_best/
```

Each split checkpoint contains:

```text
client_state.pt
server_state.pt
tokenizer files
```

## Current Split Mode vs FedLoRA

| Feature | FedLoRA mode | Current split mode |
|---|---|---|
| Command flag | no `--split` | `--split` |
| Model location | Full model on each client | Model divided into client/server |
| LoRA used? | Yes | No |
| Trainable parameters | LoRA adapter parameters | Full DistilBERT split parameters |
| Aggregated weights | LoRA adapter weights | Client-side and server-side state dicts |
| Main checkpoint folder | `models/fedlora/` | `models/split/` |

## Current Training Loop

```mermaid
flowchart TB
  Start["Start --split training"] --> Init["Create global split model"]
  Init --> Round["For each federated round"]
  Round --> Clone["Clone global client/server models for each client"]
  Clone --> Train["Train local split pair on client data"]
  Train --> Collect["Collect client_state and server_state"]
  Collect --> Avg["FedAvg client states\nFedAvg server states"]
  Avg --> Validate["Validate global split model"]
  Validate --> Best{"Validation macro F1 improved?"}
  Best -->|"Yes"| SaveBest["Save best split checkpoint"]
  Best -->|"No"| Next["Continue"]
  SaveBest --> Next
  Next --> Round
  Round --> Final["Evaluate final global split model\non test_clean.csv"]
```

## Practical Note

Because current split mode updates the full DistilBERT model, it is more sensitive than FedLoRA. A LoRA learning rate such as `2e-4` can be too high for split mode.

A safer split-learning starting command is:

```bash
python src/train_fedlora.py \
  --split --split_layer 1 \
  --rounds 5 \
  --local_epochs 1 \
  --lr 2e-5 \
  --clients_dir data/clients/setting_D_300 \
  --agg_weight total
```
