"""
evaluate.py — evaluation metrics for all experiments.
Saves confusion matrix figures and result CSVs.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)

from utils import LABEL2ID, ID2LABEL, REPORTS_DIR, append_result

FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(y_true, y_pred, label_names=None) -> dict:
    if label_names is None:
        label_names = [ID2LABEL[i] for i in sorted(ID2LABEL)]

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0
    )

    macro   = report["macro avg"]
    weighted = report["weighted avg"]

    # Smishing-specific rates
    sm_idx = LABEL2ID.get("smishing", 2)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    sm_mask = (y_true_arr == sm_idx)
    if sm_mask.sum() > 0:
        sm_tp  = ((y_pred_arr == sm_idx) & sm_mask).sum()
        sm_fn  = ((y_pred_arr != sm_idx) & sm_mask).sum()
        sm_fp  = ((y_pred_arr == sm_idx) & ~sm_mask).sum()
        sm_tn  = ((y_pred_arr != sm_idx) & ~sm_mask).sum()
        sm_fnr = sm_fn / (sm_tp + sm_fn) if (sm_tp + sm_fn) > 0 else 0.0
        sm_fpr = sm_fp / (sm_fp + sm_tn) if (sm_fp + sm_tn) > 0 else 0.0
    else:
        sm_fnr = sm_fpr = None

    per_class = {}
    for name in label_names:
        if name in report:
            per_class[name] = {
                "precision": report[name]["precision"],
                "recall":    report[name]["recall"],
                "f1":        report[name]["f1-score"],
                "support":   report[name]["support"],
            }

    return {
        "accuracy":          round(acc, 4),
        "macro_precision":   round(macro["precision"], 4),
        "macro_recall":      round(macro["recall"], 4),
        "macro_f1":          round(macro["f1-score"], 4),
        "weighted_f1":       round(weighted["f1-score"], 4),
        "smishing_fnr":      round(sm_fnr, 4) if sm_fnr is not None else None,
        "smishing_fpr":      round(sm_fpr, 4) if sm_fpr is not None else None,
        "per_class":         per_class,
    }


def save_confusion_matrix(y_true, y_pred, label_names, tag: str):
    """Save confusion matrix PNG."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {tag}")
    plt.tight_layout()
    out_path = FIGURES_DIR / f"confusion_matrix_{tag}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved -> {out_path.name}")


def report_metrics(metrics: dict, experiment: str, tag: str,
                   y_true=None, y_pred=None,
                   extra: Optional[dict] = None):

    label_names = [ID2LABEL[i] for i in sorted(ID2LABEL)]

    print(f"\n{'='*55}")
    print(f"  {experiment}")
    print(f"{'='*55}")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:     {metrics['weighted_f1']:.4f}")
    print(f"  Smishing FNR:    {metrics['smishing_fnr']}")
    print(f"  Smishing FPR:    {metrics['smishing_fpr']}")
    print(f"  Per-class:")
    for cls, vals in metrics["per_class"].items():
        print(f"    {cls:12s}  P={vals['precision']:.3f}  "
              f"R={vals['recall']:.3f}  F1={vals['f1']:.3f}  n={int(vals['support'])}")

    if y_true is not None and y_pred is not None:
        save_confusion_matrix(y_true, y_pred, label_names, tag)

    flat_row = {
        "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M"),
        "experiment":      experiment,
        "tag":             tag,
        "accuracy":        metrics["accuracy"],
        "macro_f1":        metrics["macro_f1"],
        "weighted_f1":     metrics["weighted_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall":    metrics["macro_recall"],
        "smishing_fnr":    metrics["smishing_fnr"],
        "smishing_fpr":    metrics["smishing_fpr"],
    }
    for cls, vals in metrics["per_class"].items():
        flat_row[f"{cls}_f1"] = vals["f1"]
    if extra:
        flat_row.update(extra)

    append_result(REPORTS_DIR / f"results_{tag}.csv", flat_row)

    results_md = REPORTS_DIR / "RESULTS.md"
    with open(results_md, "a", encoding="utf-8") as f:
        f.write(f"\n### {experiment} ({datetime.now():%Y-%m-%d %H:%M})\n")
        f.write(f"Accuracy={metrics['accuracy']:.4f} | Macro F1={metrics['macro_f1']:.4f} | "
                f"Weighted F1={metrics['weighted_f1']:.4f} | "
                f"Smishing FNR={metrics['smishing_fnr']} | "
                f"Smishing FPR={metrics['smishing_fpr']}\n")
        for cls, vals in metrics["per_class"].items():
            f.write(f"- {cls}: P={vals['precision']:.3f} R={vals['recall']:.3f} "
                    f"F1={vals['f1']:.3f} (n={int(vals['support'])})\n")
