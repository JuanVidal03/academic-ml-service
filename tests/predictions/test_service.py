"""The predictions service, exercised without HTTP."""

import logging

from dropout_service.predictions.schemas import PredictionRequest
from dropout_service.predictions.service import log_predictions


def test_log_predictions_is_a_noop_without_a_database(predictor, student):
    """The database is optional, so this must do nothing rather than raise."""
    features = PredictionRequest(**student)
    results = predictor.predict([features])
    log_predictions(None, [features], results, 1)  # must not raise


def test_log_predictions_swallows_database_failures(predictor, student, caplog):
    """A logging failure must never surface to the caller.

    The prediction has already succeeded by the time this runs in a background task, so
    turning an audit-write error into a request failure would be strictly worse.
    """
    features = PredictionRequest(**student)
    results = predictor.predict([features])

    def exploding_factory():
        raise RuntimeError("database is down")

    with caplog.at_level(logging.ERROR):
        log_predictions(exploding_factory, [features], results, 1)

    assert "Failed to write" in caplog.text


def test_log_predictions_rejects_mismatched_lengths(predictor, student):
    """zip(strict=True) must catch a students/results mismatch rather than silently
    truncating, which would write the wrong input alongside a prediction."""
    features = PredictionRequest(**student)
    results = predictor.predict([features])

    class DummySession:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def add_all(self, rows):
            raise AssertionError("should not get this far")

        def commit(self):
            raise AssertionError("should not get this far")

    # Two students, one result -> strict zip raises, and the error is logged not raised.
    log_predictions(lambda: DummySession(), [features, features], results, 1)
