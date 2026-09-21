"""Fairness audit: error-rate parity and calibration by protected group.

Method follows the C30 (COMPAS/ProPublica) framing: report both the
error-rate view (FPR/FNR by group) and the calibration view (predicted vs.
observed default rate by group), and name the tension between them rather
than picking one as "the" fairness metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def group_error_rates(
    y_true: np.ndarray, y_pred: np.ndarray, group: pd.Series
) -> pd.DataFrame:
    """False positive / false negative rate by group, at a fixed decision threshold.

    FPR here is the rate at which good accounts (y_true=0) are declined
    (y_pred=1); FNR is the rate at which bad accounts (y_true=1) are approved
    (y_pred=0).
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": group.to_numpy()})
    rows = []
    for g, sub in df.groupby("group"):
        negatives = sub[sub["y_true"] == 0]
        positives = sub[sub["y_true"] == 1]
        fpr = (negatives["y_pred"] == 1).mean() if len(negatives) else np.nan
        fnr = (positives["y_pred"] == 0).mean() if len(positives) else np.nan
        rows.append(
            {
                "group": g,
                "n": len(sub),
                "actual_default_rate": sub["y_true"].mean(),
                "predicted_default_rate": sub["y_pred"].mean(),
                "fpr": fpr,
                "fnr": fnr,
            }
        )
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def group_calibration(
    y_true: np.ndarray, y_score: np.ndarray, group: pd.Series, n_bins: int = 5
) -> pd.DataFrame:
    """Mean predicted probability vs. observed default rate, by group and score bin."""
    df = pd.DataFrame({"y": y_true, "score": y_score, "group": group.to_numpy()})
    rows = []
    for g, sub in df.groupby("group"):
        sub = sub.copy()
        sub["bin"] = pd.qcut(sub["score"], n_bins, duplicates="drop")
        table = sub.groupby("bin", observed=True).agg(
            n=("y", "count"), mean_predicted=("score", "mean"), observed_rate=("y", "mean")
        )
        table["group"] = g
        rows.append(table.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


def bucket_age(age: pd.Series) -> pd.Series:
    """AGE is continuous; bucket it into bands so it works as a fairness-audit group."""
    return pd.cut(
        age,
        bins=[0, 25, 35, 45, 55, 200],
        labels=["<=25", "26-35", "36-45", "46-55", "56+"],
    )
