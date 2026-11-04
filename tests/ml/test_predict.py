"""Predictor behaviour."""

import pytest

from dropout_service.ml.predict import Predictor
from dropout_service.ml.schema import StudentFeatures


def test_probability_is_a_valid_probability(predictor, student):
    result = predictor.predict([StudentFeatures(**student)])[0]
    assert 0.0 <= result.probabilidad <= 1.0


def test_prediction_is_deterministic(predictor, student):
    features = StudentFeatures(**student)
    first = predictor.predict([features])[0]
    second = predictor.predict([features])[0]
    assert first.probabilidad == second.probabilidad
    assert first.deserta == second.deserta


def test_deserta_follows_the_calibrated_threshold_not_one_half(predictor, student):
    """The threshold is tuned for recall, so model.predict()'s hardcoded 0.5 is wrong."""
    result = predictor.predict([StudentFeatures(**student)])[0]
    assert result.deserta == (result.probabilidad >= predictor.threshold)


def test_batch_preserves_input_order(predictor, student):
    low = dict(student, nota_promedio=4.8, porcentaje_asistencia=99.0, promedio_acumulado=4.7)
    high = dict(student, nota_promedio=1.2, porcentaje_asistencia=40.0, promedio_acumulado=1.4)
    results = predictor.predict([StudentFeatures(**low), StudentFeatures(**high)])

    assert len(results) == 2
    singles = [predictor.predict([StudentFeatures(**s)])[0] for s in (low, high)]
    assert [r.probabilidad for r in results] == [s.probabilidad for s in singles]


def test_a_struggling_student_scores_higher_than_a_thriving_one(predictor, student):
    """Sanity check that the model learned the expected direction."""
    thriving = dict(
        student,
        nota_promedio=4.8,
        promedio_acumulado=4.7,
        porcentaje_asistencia=99.0,
        materias_reprobadas=0,
        inasistencias=0,
    )
    struggling = dict(
        student,
        nota_promedio=1.5,
        promedio_acumulado=1.6,
        porcentaje_asistencia=45.0,
        materias_reprobadas=4,
        inasistencias=20,
    )
    results = predictor.predict([StudentFeatures(**thriving), StudentFeatures(**struggling)])
    assert results[1].probabilidad > results[0].probabilidad


def test_empty_batch_returns_empty_list(predictor):
    assert predictor.predict([]) == []


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.0, "bajo"), (0.29, "bajo"), (0.3, "medio"), (0.59, "medio"), (0.6, "alto"), (1.0, "alto")],
)
def test_risk_bands(predictor, probability, expected):
    assert predictor.risk_level(probability) == expected


def test_dump_and_from_bytes_round_trip(predictor, model_artifact, student):
    """The artefact stored in BYTEA must rebuild an identical predictor."""
    artifact, metadata = model_artifact
    rebuilt = Predictor.from_bytes(Predictor.dump(predictor.pipeline), metadata)

    original = predictor.predict([StudentFeatures(**student)])[0]
    again = rebuilt.predict([StudentFeatures(**student)])[0]
    assert original.probabilidad == again.probabilidad


def test_from_bytes_defaults_to_a_half_threshold_without_metadata(predictor):
    """Metadata carries the calibrated threshold; without it, fall back to 0.5."""
    bare = Predictor.from_bytes(Predictor.dump(predictor.pipeline))
    assert bare.threshold == 0.5


def test_from_bytes_rejects_a_corrupt_artifact():
    with pytest.raises(Exception):  # noqa: B017 - joblib raises various unpickling errors
        Predictor.from_bytes(b"not a joblib payload")
