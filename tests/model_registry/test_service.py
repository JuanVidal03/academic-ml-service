"""Saving and loading model versions, without a database."""

from dropout_service.ml.predict import Predictor
from dropout_service.ml.schema import StudentFeatures
from dropout_service.model_registry.models import ModelVersion
from dropout_service.model_registry.service import load_active, save_version


def test_save_version_stores_the_artefact_and_metrics(fake_session, predictor, model_artifact):
    _, metadata = model_artifact
    version = save_version(fake_session, predictor.pipeline, metadata)

    assert fake_session.committed
    assert version.algorithm == "LogisticRegression"
    assert version.n_features == metadata["n_features"]
    assert version.decision_threshold == metadata["decision_threshold"]
    assert version.metrics == metadata
    # The artefact is bytes, not JSON: a fitted Pipeline is not JSON-serialisable.
    assert isinstance(version.artifact, bytes)
    assert len(version.artifact) > 1000


def test_save_version_deactivates_the_previous_one(fake_session, predictor, model_artifact):
    """Only one model may be active, and the UPDATE must precede the INSERT.

    Doing it the other way round would momentarily violate the partial unique index on
    is_active and abort the transaction.
    """
    _, metadata = model_artifact
    version = save_version(fake_session, predictor.pipeline, metadata)

    assert version.is_active is True
    assert fake_session.executed, "expected an UPDATE deactivating the previous version"
    statement = str(fake_session.executed[0]).lower()
    assert "update" in statement and "is_active" in statement


def test_save_version_can_store_without_activating(fake_session, predictor, model_artifact):
    _, metadata = model_artifact
    version = save_version(fake_session, predictor.pipeline, metadata, activate=False)
    assert version.is_active is False
    assert not fake_session.executed


def test_load_active_returns_none_when_there_is_no_model(fake_session):
    """A fresh database must leave the service degraded, not crash it."""
    assert load_active(fake_session) is None


def test_load_active_rebuilds_a_working_predictor(fake_session, model_artifact, student):
    """The BYTEA round-trip must reconstruct a pipeline that predicts identically."""
    artifact, metadata = model_artifact
    row = ModelVersion(
        id=7,
        version=metadata["model_version"],
        algorithm=metadata["model_name"],
        artifact=artifact,
        metrics=metadata,
        decision_threshold=metadata["decision_threshold"],
        n_features=metadata["n_features"],
        is_active=True,
    )
    fake_session._scalars_result = [row]

    loaded = load_active(fake_session)
    assert loaded is not None
    assert loaded.model_version_id == 7
    assert loaded.threshold == metadata["decision_threshold"]

    direct = Predictor.from_bytes(artifact, metadata)
    features = StudentFeatures(**student)
    assert loaded.predict([features])[0].probabilidad == direct.predict([features])[0].probabilidad
