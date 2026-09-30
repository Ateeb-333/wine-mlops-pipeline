"""Unit tests for src/data.py."""

import numpy as np
import pytest

from src.data import (
    EXPECTED_N_FEATURES,
    get_train_test_data,
    load_data,
    split_data,
    validate_data,
)


def test_load_data_shape():
    X, y = load_data()
    assert X.shape == (178, EXPECTED_N_FEATURES)
    assert len(y) == 178


def test_no_null_values():
    X, y = load_data()
    assert not X.isnull().values.any()
    assert not y.isnull().values.any()


def test_validate_passes_on_clean_data():
    X, y = load_data()
    validate_data(X, y)  # should not raise


def test_validate_rejects_nulls():
    X, y = load_data()
    X.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="null"):
        validate_data(X, y)


def test_validate_rejects_wrong_feature_count():
    X, y = load_data()
    with pytest.raises(ValueError, match="features"):
        validate_data(X.drop(columns=X.columns[0]), y)


def test_split_sizes_80_20():
    X_train, X_test, y_train, y_test = get_train_test_data()
    assert len(X_train) == 142
    assert len(X_test) == 36
    assert len(X_train) + len(X_test) == 178


def test_split_is_stratified():
    X, y = load_data()
    _, _, y_train, y_test = split_data(X, y)
    full = y.value_counts(normalize=True).sort_index()
    train = y_train.value_counts(normalize=True).sort_index()
    test = y_test.value_counts(normalize=True).sort_index()
    assert np.allclose(full, train, atol=0.03)
    assert np.allclose(full, test, atol=0.05)


def test_split_is_reproducible():
    first = get_train_test_data()
    second = get_train_test_data()
    assert first[0].index.equals(second[0].index)
    assert first[1].index.equals(second[1].index)
