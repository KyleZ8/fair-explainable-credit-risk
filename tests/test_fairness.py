from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.fairness import approval_fairness_table, true_positive_rate_gap


def test_disparate_impact_flags_group_below_four_fifths_rule() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.6, 0.7, 0.1, 0.2, 0.3, 0.4])
    group = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"])

    table = approval_fairness_table(y_true, y_score, threshold=0.5, group=group)
    row_a = table.loc[table["group"].eq("A")].iloc[0]
    row_b = table.loc[table["group"].eq("B")].iloc[0]

    assert row_a["approval_rate"] == pytest.approx(0.5)
    assert row_b["approval_rate"] == pytest.approx(1.0)
    assert row_a["disparate_impact_ratio"] == pytest.approx(0.5)
    assert bool(row_a["di_flag_below_0_8"])
    assert true_positive_rate_gap(table) == pytest.approx(0.5)
