"""Scorecard performance metrics: ROC-AUC, KS, Gini, calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def gini(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Gini coefficient = 2*AUC - 1, the standard credit-scoring transform of AUC."""
    return 2 * roc_auc(y_true, y_score) - 1


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: max separation between the cumulative
    good-score and bad-score distributions across all thresholds."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


@dataclass
class ModelMetrics:
    label: str
    roc_auc: float
    ks: float
    gini: float


def summarize(label: str, y_true: np.ndarray, y_score: np.ndarray) -> ModelMetrics:
    return ModelMetrics(label, roc_auc(y_true, y_score), ks_statistic(y_true, y_score), gini(y_true, y_score))


def calibration_table(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Mean predicted probability vs. observed default rate, by score decile."""
    df = pd.DataFrame({"y": y_true, "score": y_score})
    df["bin"] = pd.qcut(df["score"], n_bins, duplicates="drop")
    table = df.groupby("bin", observed=True).agg(
        n=("y", "count"), mean_predicted=("score", "mean"), observed_rate=("y", "mean")
    )
    return table.reset_index(drop=True)
