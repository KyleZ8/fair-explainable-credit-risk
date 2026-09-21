"""Cost-based approval threshold: pick the score cutoff that minimizes expected
cost, not the one that maximizes accuracy.

Default costs assume missing a default (approving a bad account) costs 5x
more than wrongly declining a good account (lost margin, not lost principal)
- a standard illustrative ratio in credit-risk case studies, not fit to this
issuer's real loss data. See .ai/PROJECT_SPEC.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_COST_FN = 5.0  # cost of approving an account that defaults
DEFAULT_COST_FP = 1.0  # cost of declining an account that would not have defaulted


def expected_cost(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
) -> float:
    """Mean cost per account. y_pred=1 (score >= threshold) means decline."""
    y_pred = (y_score >= threshold).astype(int)
    n_fn = int(((y_pred == 0) & (y_true == 1)).sum())  # approved, defaulted
    n_fp = int(((y_pred == 1) & (y_true == 0)).sum())  # declined, would not have defaulted
    return (n_fn * cost_fn + n_fp * cost_fp) / len(y_true)


@dataclass
class ThresholdResult:
    cost_fn: float
    cost_fp: float
    best_threshold: float
    best_cost: float
    curve: pd.DataFrame  # threshold, expected_cost, approval_rate


def optimal_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    cost_fn: float = DEFAULT_COST_FN,
    cost_fp: float = DEFAULT_COST_FP,
    n_thresholds: int = 200,
) -> ThresholdResult:
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    costs = np.array([expected_cost(y_true, y_score, t, cost_fn, cost_fp) for t in thresholds])
    approval_rate = np.array([(y_score < t).mean() for t in thresholds])

    curve = pd.DataFrame({"threshold": thresholds, "expected_cost": costs, "approval_rate": approval_rate})
    best_idx = int(np.argmin(costs))
    return ThresholdResult(cost_fn, cost_fp, float(thresholds[best_idx]), float(costs[best_idx]), curve)
