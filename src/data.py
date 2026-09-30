"""Data pipeline: load, validate and split the Wine dataset."""

from typing import Tuple

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2
EXPECTED_N_FEATURES = 13
EXPECTED_CLASSES = {0, 1, 2}


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the Wine dataset as a feature DataFrame and target Series."""
    wine = load_wine(as_frame=True)
    X = wine.data.copy()
    y = wine.target.copy()
    y.name = "target"
    return X, y


def validate_data(X: pd.DataFrame, y: pd.Series) -> None:
    """Raise ValueError if the dataset violates basic quality expectations."""
    if X.isnull().values.any():
        raise ValueError("Feature matrix contains null values.")
    if y.isnull().values.any():
        raise ValueError("Target vector contains null values.")
    if X.shape[1] != EXPECTED_N_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_N_FEATURES} features, got {X.shape[1]}."
        )
    if len(X) != len(y):
        raise ValueError("Features and target have different lengths.")
    unexpected = set(y.unique()) - EXPECTED_CLASSES
    if unexpected:
        raise ValueError(f"Unexpected class labels found: {sorted(unexpected)}")


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def get_train_test_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load, validate and split the dataset in one call."""
    X, y = load_data()
    validate_data(X, y)
    return split_data(X, y)
