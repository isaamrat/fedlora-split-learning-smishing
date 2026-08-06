# Full Model Architecture: FedLoRA + Split Learning

This document shows the full architecture used in this project: five federated SMS clients, DistilBERT, LoRA adapters, split learning, server-side classification, aggregation, and evaluation.

The implementation is mainly in `src/train_fedlora.py`.

## High-Level System

```mermaid
flowchart TB
  subgraph DataLayer["Federated SMS Data"]
    C1["client_1.csv"]
    C2["client_2.csv"]
    C3["client_3.csv"]
    C4["client_4.csv"]
    C5["client_5.csv"]
  end

  subgraph Clients["Five Federated Clients"]
    CL1["Client 1\nlocal SMS rows"]
    CL2["Client 2\nlocal SMS rows"]
    CL3["Client 3\nlocal SMS rows"]
    CL4["Client 4\nlocal SMS rows"]
    CL5["Client 5\nlocal SMS rows"]
  end

  subgraph Coordinator["Federated Coordinator / Server Process"]
    G["Global model state"]
    A["Weighted FedAvg aggregation\nsmishing | total | sqrt | balanced | uniform"]
    V["Validation on data/splits/val.csv"]
    T["Final test on data/splits/test_clean.csv"]
  end

  C1 --> CL1
  C2 --> CL2
  C3 --> CL3
  C4 --> CL4
  C5 --> CL5

  G --> CL1
  G --> CL2
  G --> CL3
  G --> CL4
  G --> CL5

  CL1 --> A
  CL2 --> A
  CL3 --> A
  CL4 --> A
  CL5 --> A

  A --> G
  G --> V
  G --> T
```

Each client trains on its own CSV file. Raw SMS text stays inside the client simulation. The server/coordinator receives model updates, aggregates them, validates the global model, and saves checkpoints.

## Base DistilBERT Model

The base model is:

```text
distilbert-base-uncased
```

The classification task has 3 labels:

```text
ham      -> 0
spam     -> 1
smishing -> 2
```

DistilBERT contains:

```mermaid
flowchart LR
  I["input_ids + attention_mask"] --> E["Embeddings"]
  E --> L1["Transformer layer 1"]
  L1 --> L2["Transformer layer 2"]
  L2 --> L3["Transformer layer 3"]
  L3 --> L4["Transformer layer 4"]
  L4 --> L5["Transformer layer 5"]
  L5 --> L6["Transformer layer 6"]
  L6 --> CLS["[CLS] hidden state"]
  CLS --> PC["pre_classifier"]
  PC --> DO["dropout"]
  DO --> H["classifier head\n3 logits"]
  H --> Y["prediction:\nham | spam | smishing"]
```

## LoRA Architecture

Normal FedLoRA mode keeps the full DistilBERT model on each client, but only trains small LoRA adapter weights.

LoRA is applied to the DistilBERT attention projection layers:

```text
target_modules = ["q_lin", "v_lin"]
r = 8
lora_alpha = 16
lora_dropout = 0.1
```

```mermaid
flowchart TB
  subgraph ClientModel["Client-side FedLoRA Model"]
    X["SMS text"] --> Tok["Tokenizer\nmax_len=128"]
    Tok --> DB["DistilBERT encoder"]
    DB --> Head["Classification head"]
    Head --> Logits["3 logits"]
  end

  subgraph LoRA["Trainable LoRA Adapters"]
    Q["LoRA on q_lin\nquery projection"]
    V["LoRA on v_lin\nvalue projection"]
  end

  DB -. "frozen base weights" .-> Base["Base DistilBERT weights"]
  Q --> DB
  V --> DB
```

In this mode, clients send only LoRA adapter weights to the server. The base DistilBERT weights remain fixed.

## FedLoRA Training Flow

```mermaid
sequenceDiagram
  participant S as Server / Coordinator
  participant C1 as Client 1
  participant C2 as Client 2
  participant C3 as Client 3
  participant C4 as Client 4
  participant C5 as Client 5

  S->>C1: Send global LoRA adapter
  S->>C2: Send global LoRA adapter
  S->>C3: Send global LoRA adapter
  S->>C4: Send global LoRA adapter
  S->>C5: Send global LoRA adapter

  C1->>C1: Train LoRA locally
  C2->>C2: Train LoRA locally
  C3->>C3: Train LoRA locally
  C4->>C4: Train LoRA locally
  C5->>C5: Train LoRA locally

  C1->>S: Return LoRA adapter weights
  C2->>S: Return LoRA adapter weights
  C3->>S: Return LoRA adapter weights
  C4->>S: Return LoRA adapter weights
  C5->>S: Return LoRA adapter weights

  S->>S: Weighted FedAvg over adapters
  S->>S: Validate global model
```

## Split Learning Architecture

Split learning divides DistilBERT into two parts:

- **Client-side model**: embeddings + first `split_layer` transformer blocks
- **Server-side model**: remaining transformer blocks + classification head

With the common command:

```bash
python src/train_fedlora.py --split --split_layer 3 ...
```

the split is:

```text
Client side: embeddings + transformer layers 1, 2, 3
Server side: transformer layers 4, 5, 6 + classifier
```

```mermaid
flowchart LR
  subgraph ClientSide["Client Side"]
    X["Raw SMS text"] --> Tok["Tokenizer"]
    Tok --> IDs["input_ids\nattention_mask"]
    IDs --> Emb["DistilBERT embeddings"]
    Emb --> CLayer1["Transformer layer 1"]
    CLayer1 --> CLayer2["Transformer layer 2"]
    CLayer2 --> CLayer3["Transformer layer 3"]
    CLayer3 --> Smash["Smashed data\nhidden states"]
  end

  subgraph ServerSide["Server Side"]
    Smash --> SLayer4["Transformer layer 4"]
    SLayer4 --> SLayer5["Transformer layer 5"]
    SLayer5 --> SLayer6["Transformer layer 6"]
    SLayer6 --> CLS["Take hidden[:, 0]\nCLS token representation"]
    CLS --> PC["pre_classifier"]
    PC --> Drop["dropout"]
    Drop --> Head["classifier"]
    Head --> Logits["3 logits"]
  end

  Logits --> Loss["Cross entropy loss\nwith class weights"]
```

## Split Learning Forward and Backward Pass

```mermaid
sequenceDiagram
  participant C as Client Model
  participant S as Server Model

  C->>C: Token IDs -> embeddings
  C->>C: Run first N DistilBERT layers
  C->>S: Send hidden states
  S->>S: Run remaining DistilBERT layers
  S->>S: Classifier produces logits
  S->>S: Compute cross-entropy loss
  S-->>C: Backpropagate gradients through split point
  C->>C: Update client-side parameters
  S->>S: Update server-side parameters
```

In this repository, split learning is simulated on one machine/GPU. The code still keeps the architectural separation between the client model and server model.

## Five-Client Split-Fed Architecture

In split federated mode, each client receives a copy of the global client/server split model pair. After local split training, both sides are aggregated.

```mermaid
flowchart TB
  subgraph Global["Global Split Model"]
    GC["Global client-side weights"]
    GS["Global server-side weights"]
  end

  subgraph Round["One Federated Round"]
    direction TB
    P1["client_1\ntrain split pair locally"]
    P2["client_2\ntrain split pair locally"]
    P3["client_3\ntrain split pair locally"]
    P4["client_4\ntrain split pair locally"]
    P5["client_5\ntrain split pair locally"]
  end

  GC --> P1
  GS --> P1
  GC --> P2
  GS --> P2
  GC --> P3
  GS --> P3
  GC --> P4
  GS --> P4
  GC --> P5
  GS --> P5

  P1 --> CStates["Client-side state dicts"]
  P2 --> CStates
  P3 --> CStates
  P4 --> CStates
  P5 --> CStates

  P1 --> SStates["Server-side state dicts"]
  P2 --> SStates
  P3 --> SStates
  P4 --> SStates
  P5 --> SStates

  CStates --> CAvg["FedAvg client-side weights"]
  SStates --> SAvg["FedAvg server-side weights"]

  CAvg --> GC2["Updated global client model"]
  SAvg --> GS2["Updated global server model"]
  GC2 --> Eval["Validation macro F1\nand smishing FNR"]
  GS2 --> Eval
```

## Aggregation

The same weighting options are available for FedLoRA and split-fed training:

| Option | Meaning |
|---|---|
| `smishing` | Weight each client by its number of smishing samples |
| `total` | Weight each client by total number of samples |
| `sqrt` | Weight each client by square root of smishing count |
| `balanced` | Average of total-sample and smishing-sample weighting |
| `uniform` | Equal client weight |

For FedLoRA, aggregation averages LoRA adapter weights.

For split learning, aggregation averages:

```text
1. client-side model state dicts
2. server-side model state dicts
```

## Checkpoints and Outputs

FedLoRA adapter checkpoints:

```text
models/fedlora/global_adapter_{setting_name}/
models/fedlora/global_adapter_{setting_name}_best/
```

Split learning checkpoints:

```text
models/split/{setting_name}/
models/split/{setting_name}_best/
```

Split checkpoints contain:

```text
client_state.pt
server_state.pt
tokenizer files
```

Result files:

```text
reports/results_fedlora_rounds.csv
reports/results_split_final.csv
reports/RESULTS.md
reports/figures/confusion_matrix_split_final.png
```

## End-to-End Training Summary

```mermaid
flowchart TB
  Start["Start training command"] --> Load["Load tokenizer, val set,\ntest_clean set, and 5 clients"]
  Load --> Mode{"Training mode"}

  Mode -->|"FedLoRA"| FL["Build DistilBERT + LoRA"]
  FL --> FLTrain["Train LoRA locally on each client"]
  FLTrain --> FLAgg["FedAvg LoRA adapters"]
  FLAgg --> FLVal["Validate global LoRA model"]
  FLVal --> FLBest["Save best adapter checkpoint"]

  Mode -->|"Split learning"| SL["Build split DistilBERT pair"]
  SL --> SLTrain["Train client/server split pair\non each client"]
  SLTrain --> SLAgg["FedAvg client-side and server-side weights"]
  SLAgg --> SLVal["Validate global split model"]
  SLVal --> SLBest["Save best split checkpoint"]

  FLBest --> Test["Evaluate on clean test set"]
  SLBest --> Test
  Test --> Metrics["Accuracy, macro F1,\nweighted F1, smishing FNR/FPR"]
```

## Important Tuning Note

FedLoRA trains only a small number of adapter parameters, so a learning rate like `2e-4` can be reasonable.

Split learning in this implementation trains the full split DistilBERT model, so the trainable parameter count is much larger. For split learning, start with a smaller learning rate:

```bash
python src/train_fedlora.py \
  --split --split_layer 1 --rounds 5 --local_epochs 1 --lr 2e-5 \
  --clients_dir data/clients/setting_D_300 \
  --agg_weight total
```

Then increase rounds only if validation macro F1 remains stable.
