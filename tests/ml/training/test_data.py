"""Preparing a raw training frame and splitting it."""

import pandas as pd
import pytest

from dropout_service.ml.schema import ALL_FEATURES, TARGET_COLUMN
from dropout_service.ml.training.data import prepare_dataset, split_dataset


def test_prepare_returns_features_and_binary_target(dataset):
    X, y = dataset
    assert len(X) == 3000
    assert list(X.columns) == ALL_FEATURES
    assert set(y.unique()) == {0, 1}
    assert len(X) == len(y)


def test_prepare_does_not_impute(dataset):
    """Imputation belongs to the pipeline, so missing values must survive preparation.

    If this starts failing because nulls were filled here, the median leak is back.
    """
    X, _ = dataset
    assert X["area_materia_critica"].isna().sum() == 1507


def test_target_column_is_not_a_feature(dataset):
    X, _ = dataset
    assert TARGET_COLUMN not in X.columns


def test_identifier_columns_are_excluded(dataset):
    """Leaking an identifier or the period would let the model memorise rows."""
    X, _ = dataset
    for column in ("id_estudiante", "periodo", "materia_mas_repetida"):
        assert column not in X.columns


def test_boolean_target_is_accepted(raw_frame):
    """The database stores the label as a boolean; the CSV stores Si/No. Both must work."""
    frame = raw_frame.head(20).copy()
    frame[TARGET_COLUMN] = [i % 2 == 0 for i in range(20)]
    _, y = prepare_dataset(frame)
    assert set(y.unique()) <= {0, 1}
    assert y.dtype.kind == "i"


def test_unmappable_target_is_reported(raw_frame):
    """A corrupt target must fail loudly rather than produce silent NaN labels."""
    frame = raw_frame.head(5).copy()
    frame[TARGET_COLUMN] = ["Si", "No", "Maybe", "Si", "No"]
    with pytest.raises(ValueError, match="Maybe"):
        prepare_dataset(frame)


def test_missing_target_column_is_reported(raw_frame):
    frame = raw_frame.head(5).drop(columns=[TARGET_COLUMN])
    with pytest.raises(ValueError, match=TARGET_COLUMN):
        prepare_dataset(frame)


def test_absent_feature_column_becomes_missing_not_an_error(raw_frame):
    """A frame lacking an optional column should still prepare; imputers handle the gap."""
    frame = raw_frame.head(20).drop(columns=["horas_estudio_semana"])
    X, _ = prepare_dataset(frame)
    assert list(X.columns) == ALL_FEATURES
    assert X["horas_estudio_semana"].isna().all()


def test_split_is_stratified_and_sized(dataset):
    X, y = dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.2, random_state=42)

    assert len(X_train) == 2400
    assert len(X_test) == 600
    assert y_train.mean() == pytest.approx(y.mean(), abs=0.01)
    assert y_test.mean() == pytest.approx(y.mean(), abs=0.01)


def test_split_is_disjoint(dataset):
    X, y = dataset
    X_train, X_test, _, _ = split_dataset(X, y, random_state=42)
    assert not set(X_train.index) & set(X_test.index)


def test_split_is_reproducible(dataset):
    X, y = dataset
    first, _, _, _ = split_dataset(X, y, random_state=7)
    second, _, _, _ = split_dataset(X, y, random_state=7)
    assert list(first.index) == list(second.index)


def test_dataset_has_no_duplicate_rows(dataset):
    X, _ = dataset
    assert not pd.DataFrame(X).duplicated().any()
