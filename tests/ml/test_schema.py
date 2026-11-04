"""The feature lists must stay in lockstep with the input contract."""

import pytest
from pydantic import ValidationError

from dropout_service.ml.schema import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    StudentFeatures,
)


def test_feature_lists_cover_every_field_exactly_once():
    """Regression guard: a new field cannot be silently skipped by the pipeline.

    This is the property that makes the single-source-of-truth design hold. If someone adds
    a field to StudentFeatures that the derivation does not classify, this fails.
    """
    assert set(ALL_FEATURES) == set(StudentFeatures.model_fields)
    assert len(ALL_FEATURES) == len(StudentFeatures.model_fields)
    assert not set(CATEGORICAL_FEATURES) & set(NUMERIC_FEATURES)


def test_categorical_and_numeric_split_is_correct():
    assert "programa" in CATEGORICAL_FEATURES
    assert "area_materia_critica" in CATEGORICAL_FEATURES  # Optional[Literal] is categorical
    assert "trabaja" in CATEGORICAL_FEATURES  # Literal is categorical
    assert "nota_promedio" in NUMERIC_FEATURES
    assert "semestre" in NUMERIC_FEATURES


def test_redundant_feature_is_not_in_the_contract():
    """creditos_perdidos was exactly 3 * materias_reprobadas, so it was removed."""
    assert "creditos_perdidos" not in ALL_FEATURES


def test_area_materia_critica_is_optional_and_defaults_to_none(student):
    del student["area_materia_critica"]
    assert StudentFeatures(**student).area_materia_critica is None


def test_invalid_category_is_rejected(student):
    """An unknown category must 422, not become a silent all-zero one-hot row."""
    student["genero"] = "Masculino"
    with pytest.raises(ValidationError):
        StudentFeatures(**student)


def test_unknown_field_is_rejected(student):
    student["campo_inventado"] = 1
    with pytest.raises(ValidationError):
        StudentFeatures(**student)


def test_out_of_range_value_is_rejected(student):
    student["nota_promedio"] = 9.9  # scale is 0-5
    with pytest.raises(ValidationError):
        StudentFeatures(**student)
