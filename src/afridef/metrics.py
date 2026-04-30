"""Metrics: PR-AUC, recall@FPR=k, time-to-detection, robust-recall, drift index."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve


def pr_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    return float(average_precision_score(y_true, y_score))


def recall_at_fpr(y_true: Sequence[int], y_score: Sequence[float],
                  target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    # largest TPR with FPR <= target_fpr
    mask = fpr <= target_fpr
    return float(tpr[mask].max()) if mask.any() else 0.0


def time_to_detection(events: Sequence[bool], detection_window: int = 100) -> float:
    """Mean transactions between fraud onset and first detection. `events` is
    a boolean array indicating detected fraud per transaction."""
    raise NotImplementedError("TODO: implement onset-aligned detection lag")


def drift_index(metric_per_slice: Sequence[float]) -> float:
    """Concept-drift index: relative degradation of a metric across temporal
    slices, normalised by the first slice."""
    arr = np.asarray(metric_per_slice, dtype=float)
    if len(arr) == 0 or arr[0] == 0:
        return 0.0
    return float((arr[0] - arr[-1]) / arr[0])


def robust_recall(y_true_clean, y_score_clean,
                  y_true_adv, y_score_adv, target_fpr: float = 0.01) -> float:
    """Recall-at-FPR under adversarial perturbation, normalised against
    clean-evaluation recall."""
    clean = recall_at_fpr(y_true_clean, y_score_clean, target_fpr)
    adv = recall_at_fpr(y_true_adv, y_score_adv, target_fpr)
    return adv / max(clean, 1e-8)
