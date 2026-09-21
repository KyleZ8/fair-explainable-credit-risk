"""Weight-of-Evidence binning and Information Value, for the logistic scorecard.

Standard credit-scoring practice: bin each feature, compute WoE per bin from
the training set only, then transform both train and test onto those same
bin edges. IV per feature is a standard rule-of-thumb predictiveness ranking
(<0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, >0.5 suspicious).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

EPS = 0.5  # Laplace-style smoothing so an empty bin doesn't produce log(0).


@dataclass
class WoeBinner:
    n_bins: int = 10
    categorical_threshold: int = 10  # features with <= this many unique values are binned by value
    bin_edges_: dict[str, np.ndarray] = field(default_factory=dict)
    woe_tables_: dict[str, pd.DataFrame] = field(default_factory=dict)
    iv_: dict[str, float] = field(default_factory=dict)
    is_categorical_: dict[str, bool] = field(default_factory=dict)

    def _bin_labels(self, series: pd.Series, feature: str) -> pd.Series:
        if self.is_categorical_[feature]:
            return series.astype(float)
        edges = self.bin_edges_[feature]
        return pd.Series(np.digitize(series.to_numpy(dtype=float), edges[1:-1]), index=series.index)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> WoeBinner:
        n_good = int((y == 0).sum())
        n_bad = int((y == 1).sum())

        for feature in X.columns:
            series = X[feature].astype(float)
            categorical = series.nunique() <= self.categorical_threshold
            self.is_categorical_[feature] = categorical

            if categorical:
                bin_id = series
            else:
                edges = np.unique(
                    np.quantile(series, np.linspace(0, 1, self.n_bins + 1))
                )
                if len(edges) < 3:
                    edges = np.array([series.min(), series.median(), series.max()])
                edges[0], edges[-1] = -np.inf, np.inf
                self.bin_edges_[feature] = edges
                bin_id = pd.Series(np.digitize(series.to_numpy(), edges[1:-1]), index=series.index)

            table = (
                pd.DataFrame({"bin": bin_id, "y": y.to_numpy()})
                .groupby("bin")["y"]
                .agg(n="count", bad="sum")
                .assign(good=lambda t: t["n"] - t["bad"])
            )
            table["good_rate"] = (table["good"] + EPS) / (n_good + EPS * len(table))
            table["bad_rate"] = (table["bad"] + EPS) / (n_bad + EPS * len(table))
            table["woe"] = np.log(table["good_rate"] / table["bad_rate"])
            table["iv_contribution"] = (table["good_rate"] - table["bad_rate"]) * table["woe"]

            self.woe_tables_[feature] = table
            self.iv_[feature] = float(table["iv_contribution"].sum())

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for feature in X.columns:
            bin_id = self._bin_labels(X[feature], feature)
            woe_map = self.woe_tables_[feature]["woe"]
            out[feature] = bin_id.map(woe_map).fillna(0.0).to_numpy()
        return pd.DataFrame(out, index=X.index)

    def iv_table(self) -> pd.DataFrame:
        return (
            pd.Series(self.iv_, name="information_value")
            .sort_values(ascending=False)
            .rename_axis("feature")
            .reset_index()
        )
