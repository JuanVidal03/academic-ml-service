"""Training-data helpers, without a database."""

import math

import pandas as pd
import pytest

from dropout_service.ml.schema import ALL_FEATURES
from dropout_service.training_data.service import _features_payload, load_training_frame


def test_features_payload_converts_nan_to_none(raw_frame):
    """JSONB has null, not NaN. Leaving NaN in would serialise as the float 'NaN'."""
    row = raw_frame.iloc[0]
    payload = _features_payload(row, [c for c in ALL_FEATURES if c in raw_frame.columns])

    for key, value in payload.items():
        assert not (isinstance(value, float) and math.isnan(value)), f"{key} is NaN"


def test_features_payload_converts_numpy_scalars_to_python(raw_frame):
    """psycopg cannot adapt numpy scalars into JSONB."""
    row = raw_frame.iloc[0]
    payload = _features_payload(row, [c for c in ALL_FEATURES if c in raw_frame.columns])

    for key, value in payload.items():
        assert value is None or type(value) in (int, float, str, bool), (
            f"{key} is {type(value).__name__}, which psycopg cannot adapt"
        )


def test_features_payload_keeps_the_missing_category_as_null(raw_frame):
    """~50% of students have no critical area; that must round-trip as null."""
    blank = raw_frame[raw_frame["area_materia_critica"].isna()].iloc[0]
    payload = _features_payload(blank, [c for c in ALL_FEATURES if c in raw_frame.columns])
    assert payload["area_materia_critica"] is None


def test_load_training_frame_expands_jsonb_into_feature_columns(fake_session, raw_frame):
    """The frame handed to prepare_dataset must have one column per feature, in order."""
    sample = raw_frame.head(3)
    columns = [c for c in ALL_FEATURES if c in raw_frame.columns]
    fake_session._scalars_result = []
    rows = [
        (_features_payload(row, columns), bool(i % 2))
        for i, (_, row) in enumerate(sample.iterrows())
    ]

    class RowsSession(type(fake_session)):
        def execute(self, statement):
            self.executed.append(statement)
            return self

        def all(self):
            return rows

    frame = load_training_frame(RowsSession())
    assert list(frame.columns)[: len(ALL_FEATURES)] == ALL_FEATURES
    assert len(frame) == 3
    assert "deserto" in frame.columns


def test_load_training_frame_refuses_an_empty_table(fake_session):
    """An unseeded database must say so rather than train on nothing."""

    class EmptySession(type(fake_session)):
        def execute(self, statement):
            return self

        def all(self):
            return []

    with pytest.raises(RuntimeError, match="seed_db"):
        load_training_frame(EmptySession())


def test_absent_feature_comes_back_as_nan(fake_session):
    """A stored row missing a feature must reindex to NaN, not drop the column."""

    class PartialSession(type(fake_session)):
        def execute(self, statement):
            return self

        def all(self):
            return [({"nota_promedio": 3.0}, False)]

    frame = load_training_frame(PartialSession())
    assert list(frame.columns)[: len(ALL_FEATURES)] == ALL_FEATURES
    assert pd.isna(frame["porcentaje_asistencia"].iloc[0])
