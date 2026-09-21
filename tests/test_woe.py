from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.woe import WoeBinner


def test_woe_and_iv_match_smoothed_known_answer() -> None:
    x = pd.DataFrame({"status": [0, 0, 0, 1, 1, 1]})
    y = pd.Series([0, 0, 1, 0, 1, 1])

    binner = WoeBinner().fit(x, y)
    table = binner.woe_tables_["status"]

    expected_woe = np.log(0.625 / 0.375)
    assert table.loc[0, "woe"] == pytest.approx(expected_woe)
    assert table.loc[1, "woe"] == pytest.approx(-expected_woe)
    assert binner.iv_["status"] == pytest.approx(2 * (0.625 - 0.375) * expected_woe)


def test_woe_transform_reuses_training_mapping() -> None:
    x = pd.DataFrame({"status": [0, 0, 0, 1, 1, 1]})
    y = pd.Series([0, 0, 1, 0, 1, 1])

    binner = WoeBinner().fit(x, y)
    transformed = binner.transform(pd.DataFrame({"status": [0, 1, 99]}))

    assert transformed.loc[0, "status"] > 0
    assert transformed.loc[1, "status"] < 0
    assert transformed.loc[2, "status"] == 0.0
