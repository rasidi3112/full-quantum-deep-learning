from __future__ import annotations

from typing import Dict

import numpy as np # type: ignore
import torch # type: ignore
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score # type: ignore


def compute_classification_metrics(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    probs = torch.exp(log_probs).detach().cpu().numpy()
    y_true = targets.detach().cpu().numpy()
    y_pred = probs.argmax(axis=1)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average="weighted")),
    }
    try:
        metrics["nll"] = float(log_loss(y_true, probs, labels=list(range(probs.shape[1]))))
    except ValueError:
        metrics["nll"] = float("nan")

    try:
        if probs.shape[1] == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
        else:
            metrics["roc_auc"] = float(roc_auc_score(y_true, probs, multi_class="ovo"))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return metrics