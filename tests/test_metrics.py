from __future__ import annotations

import numpy as np
import pytest

from credit_risk.metrics import gini, ks_statistic, roc_auc


def test_auc_gini_and_ks_match_hand_checkable_ranking() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])

    assert roc_auc(y_true, y_score) == pytest.approx(0.75)
    assert gini(y_true, y_score) == pytest.approx(0.50)
    assert ks_statistic(y_true, y_score) == pytest.approx(0.50)
