"""
shared/fedavg.py — Weighted FedAvg aggregation helpers (shared by train_fedlora and train_split)

Two variants:
  fedavg_lora        — aggregates LoRA adapter state dicts (requires peft)
  fedavg_state_dict  — aggregates generic model state dicts (used by split learning)

Supported agg_weight strategies:
  'smishing' — weight ∝ smishing sample count              (best; default)
  'sqrt'     — weight ∝ sqrt(smishing count)
  'balanced' — 0.5×total_weight + 0.5×smishing_weight
  'total'    — weight ∝ total sample count (naive FedAvg)
  'uniform'  — equal weights
"""

import math
from typing import List, Optional


def _compute_weights(
    client_sizes: List[int],
    smishing_sizes: Optional[List[int]],
    agg_weight: str,
) -> List[float]:
    """Return a normalised weight vector for FedAvg aggregation."""
    n = len(client_sizes)

    if agg_weight == "uniform":
        return [1.0 / n] * n

    if agg_weight == "total":
        total = sum(client_sizes)
        return [s / max(total, 1) for s in client_sizes]

    if agg_weight == "sqrt" and smishing_sizes:
        roots = [math.sqrt(max(s, 1)) for s in smishing_sizes]
        total = sum(roots)
        return [r / total for r in roots]

    if agg_weight == "balanced" and smishing_sizes:
        sum_total    = max(sum(client_sizes), 1)
        sum_smishing = max(sum(smishing_sizes), 1)
        w_total    = [s / sum_total    for s in client_sizes]
        w_smishing = [max(s, 1) / sum_smishing for s in smishing_sizes]
        return [0.5 * wt + 0.5 * ws for wt, ws in zip(w_total, w_smishing)]

    # default: smishing-count linear
    counts = smishing_sizes if smishing_sizes else client_sizes
    total  = sum(max(c, 1) for c in counts)
    return [max(c, 1) / total for c in counts]


def fedavg_lora(
    global_model,
    client_state_dicts: List[dict],
    client_sizes: List[int],
    smishing_sizes: Optional[List[int]] = None,
    agg_weight: str = "smishing",
):
    """
    Weighted FedAvg over LoRA adapter state dicts.
    Requires peft — call set_peft_model_state_dict on the result.
    """
    from peft import set_peft_model_state_dict

    weights   = _compute_weights(client_sizes, smishing_sizes, agg_weight)
    avg_state = {}
    for key in client_state_dicts[0]:
        tensors = [sd[key].float() for sd in client_state_dicts]
        avg_state[key] = sum(w * t for w, t in zip(weights, tensors))

    set_peft_model_state_dict(global_model, avg_state)
    return global_model


def fedavg_state_dict(
    reference_model,
    client_state_dicts: List[dict],
    client_sizes: List[int],
    smishing_sizes: Optional[List[int]] = None,
    agg_weight: str = "smishing",
):
    """
    Weighted FedAvg over generic model state dicts.
    Used by split learning where both client and server halves are plain nn.Modules.
    """
    weights   = _compute_weights(client_sizes, smishing_sizes, agg_weight)
    avg_state = {}
    for key in client_state_dicts[0]:
        tensors = [sd[key].float() for sd in client_state_dicts]
        avg_state[key] = sum(w * t for w, t in zip(weights, tensors))

    reference_model.load_state_dict(avg_state)
    return reference_model
