"""Data loading, the feature/protected-attribute split, and the train/test split.

Protected attributes (SEX, AGE, MARRIAGE, EDUCATION) are excluded from model
features to avoid disparate treatment, but are kept alongside the test set
for the fairness audit in notebooks/02_models.ipynb. See ../../.ai/PROJECT_SPEC.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "default_of_credit_card_clients.csv"

TARGET = "default_payment_next_month"
PROTECTED = ("SEX", "AGE", "MARRIAGE", "EDUCATION")
ID_COL = "ID"

SEED = 20260101


def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model features: everything except ID, target, and protected attributes."""
    exclude = {ID_COL, TARGET, *PROTECTED}
    return [c for c in df.columns if c not in exclude]


def train_test_split_data(
    df: pd.DataFrame, test_size: float = 0.25, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split on the target, so the ~22% default rate holds in both sets."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[TARGET]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
