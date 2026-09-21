"""Fairness audit: approval parity, error-rate parity, and calibration by group."""

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


def approval_fairness_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    group: pd.Series,
    reference: str | int | None = None,
) -> pd.DataFrame:
    """Approval rate, disparate-impact ratio, TPR, and calibration by group.

    Scores are probabilities of default. Accounts below the threshold are
    approved; accounts at or above the threshold are declined. TPR is measured
    for the adverse outcome detection task: the share of true defaults that
    are correctly declined.
    """

    df = pd.DataFrame({"y": y_true, "score": y_score, "group": group.to_numpy()})
    df["approved"] = df["score"] < threshold
    df["declined"] = ~df["approved"]
    rows = []
    for group_value, sub in df.groupby("group", observed=True):
        positives = sub[sub["y"] == 1]
        tpr = positives["declined"].mean() if len(positives) else np.nan
        rows.append(
            {
                "group": group_value,
                "n": int(len(sub)),
                "approval_rate": float(sub["approved"].mean()),
                "actual_default_rate": float(sub["y"].mean()),
                "mean_predicted_pd": float(sub["score"].mean()),
                "calibration_error": float(sub["score"].mean() - sub["y"].mean()),
                "true_positive_rate": float(tpr),
            }
        )

    table = pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
    if reference is None:
        reference_rate = float(table["approval_rate"].max())
    else:
        ref = table.loc[table["group"].astype(str).eq(str(reference)), "approval_rate"]
        if ref.empty:
            raise ValueError(f"Reference group {reference!r} is not present")
        reference_rate = float(ref.iloc[0])

    table["disparate_impact_ratio"] = table["approval_rate"] / reference_rate
    table["di_flag_below_0_8"] = table["disparate_impact_ratio"] < 0.8
    return table


def true_positive_rate_gap(table: pd.DataFrame) -> float:
    """Max-minus-min true positive rate gap, ignoring empty groups."""

    return float(table["true_positive_rate"].max() - table["true_positive_rate"].min())


def bucket_age(age: pd.Series) -> pd.Series:
    """AGE is continuous; bucket it into bands so it works as a fairness-audit group."""
    return pd.cut(
        age,
        bins=[0, 25, 35, 45, 55, 200],
        labels=["<=25", "26-35", "36-45", "46-55", "56+"],
    )
