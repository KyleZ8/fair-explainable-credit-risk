"""Per-account reason codes for both models.

Scorecard: each feature's contribution to the log-odds is coefficient * WoE
value for the bin the account fell in - additive and exact by construction.
Gradient boosting: XGBoost's built-in `pred_contribs` (a SHAP-values
implementation, no extra dependency) gives the same kind of additive,
per-feature contribution for a tree ensemble. Both return the same shape:
top-N features by |contribution| per account, so the two models' reason
codes are directly comparable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb


def scorecard_reason_codes(
    woe_df: pd.DataFrame, coef: pd.Series, top_n: int = 3
) -> pd.DataFrame:
    """woe_df: WoE-transformed features (one row per account). coef: fitted
    logistic regression coefficients, indexed by feature name."""
    contributions = woe_df[coef.index] * coef.to_numpy()
    return _top_n_table(contributions, top_n)


def xgb_reason_codes(model: xgb.XGBClassifier, X: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X, feature_names=list(X.columns))
    contribs = booster.predict(dmatrix, pred_contribs=True)  # last column is the bias term
    contributions = pd.DataFrame(contribs[:, :-1], columns=X.columns, index=X.index)
    return _top_n_table(contributions, top_n)


def _top_n_table(contributions: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """For each row, the top-N features by |contribution|, as name/value pairs."""
    records = []
    values = contributions.to_numpy()
    cols = np.array(contributions.columns)
    order = np.argsort(-np.abs(values), axis=1)[:, :top_n]
    for i, idx_row in enumerate(order):
        row = {"row_index": contributions.index[i]}
        for rank, j in enumerate(idx_row, start=1):
            row[f"reason_{rank}_feature"] = cols[j]
            row[f"reason_{rank}_contribution"] = values[i, j]
        records.append(row)
    return pd.DataFrame(records)
