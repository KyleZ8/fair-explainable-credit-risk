from __future__ import annotations

import numpy as np
import pytest

from credit_risk.threshold import expected_cost, optimal_threshold, threshold_outcomes


def test_expected_cost_and_profit_at_known_threshold() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    outcomes = threshold_outcomes(y_true, y_score, threshold=0.5, cost_fn=5, cost_fp=1)

    assert expected_cost(y_true, y_score, threshold=0.5, cost_fn=5, cost_fp=1) == 0.0
    assert outcomes["approval_rate"] == pytest.approx(0.5)
    assert outcomes["approved_default_rate"] == pytest.approx(0.0)
    assert outcomes["expected_profit"] == pytest.approx(0.5)


def test_optimal_threshold_finds_zero_cost_separator() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    result = optimal_threshold(y_true, y_score, n_thresholds=11)

    assert result.best_threshold == pytest.approx(0.3)
    assert result.best_cost == 0.0
