"""Candidate persistence and loading, without a database."""

from dropout_service.ml.schema import StudentFeatures
from dropout_service.ml.training.train import CandidateResult
from dropout_service.model_registry.models import ModelCandidate, ModelVersion
from dropout_service.model_registry.service import load_candidates, save_version


def make_candidate(predictor, algorithm: str, threshold: float, is_best: bool) -> CandidateResult:
    return CandidateResult(
        algorithm=algorithm,
        pipeline=predictor.pipeline,
        cv_metrics={"roc_auc_mean": 0.77 if is_best else 0.70},
        test_metrics={"roc_auc": 0.76},
        decision_threshold=threshold,
        is_best=is_best,
    )


def test_save_version_persists_every_candidate(fake_session, predictor, model_artifact):
    _, metadata = model_artifact
    candidates = [
        make_candidate(predictor, "LogisticRegression", 0.35, True),
        make_candidate(predictor, "RandomForest", 0.38, False),
        make_candidate(predictor, "SVM", 0.34, False),
    ]

    save_version(fake_session, predictor.pipeline, metadata, candidates=candidates)

    stored = [obj for obj in fake_session.added if isinstance(obj, ModelCandidate)]
    assert len(stored) == 3
    assert {c.algorithm for c in stored} == {"LogisticRegression", "RandomForest", "SVM"}
    # Bytes, not JSON: a fitted pipeline is not serialisable as JSON.
    assert all(isinstance(c.artifact, bytes) and len(c.artifact) > 1000 for c in stored)


def test_exactly_one_candidate_is_marked_best(fake_session, predictor, model_artifact):
    _, metadata = model_artifact
    candidates = [
        make_candidate(predictor, "LogisticRegression", 0.35, True),
        make_candidate(predictor, "RandomForest", 0.38, False),
    ]

    save_version(fake_session, predictor.pipeline, metadata, candidates=candidates)

    stored = [obj for obj in fake_session.added if isinstance(obj, ModelCandidate)]
    assert sum(1 for c in stored if c.is_best) == 1


def test_candidates_keep_their_own_threshold(fake_session, predictor, model_artifact):
    """The point of storing a threshold per candidate.

    Probabilities from different algorithms are calibrated differently, so a single shared
    cut-point would make most verdicts misrepresent the model that produced them.
    """
    _, metadata = model_artifact
    save_version(
        fake_session,
        predictor.pipeline,
        metadata,
        candidates=[
            make_candidate(predictor, "LogisticRegression", 0.3505, True),
            make_candidate(predictor, "HistGradientBoosting", 0.2504, False),
        ],
    )

    stored = {c.algorithm: c for c in fake_session.added if isinstance(c, ModelCandidate)}
    assert stored["LogisticRegression"].decision_threshold == 0.3505
    assert stored["HistGradientBoosting"].decision_threshold == 0.2504


def test_load_candidates_rebuilds_working_predictors(fake_session, model_artifact, student):
    """Each loaded candidate must predict, and must use its stored threshold."""
    artifact, metadata = model_artifact
    rows = [
        ModelCandidate(
            id=1,
            model_version_id=9,
            algorithm="LogisticRegression",
            artifact=artifact,
            cv_metrics={"roc_auc_mean": 0.774},
            test_metrics={"roc_auc": 0.7695},
            decision_threshold=0.3505,
            is_best=True,
        ),
        ModelCandidate(
            id=2,
            model_version_id=9,
            algorithm="RandomForest",
            artifact=artifact,
            cv_metrics={"roc_auc_mean": 0.757},
            test_metrics={"roc_auc": 0.75},
            decision_threshold=0.9,
            is_best=False,
        ),
    ]
    fake_session._scalars_result = rows

    loaded = load_candidates(fake_session, 9)

    assert [c.algorithm for c in loaded] == ["LogisticRegression", "RandomForest"]
    assert loaded[0].is_best is True
    assert loaded[0].predictor.threshold == 0.3505
    assert loaded[1].predictor.threshold == 0.9

    features = StudentFeatures(**student)
    low, high = (c.predictor.predict([features])[0] for c in loaded)
    # Same artefact, same probability -- only the verdict differs, because the thresholds do.
    assert low.probabilidad == high.probabilidad
    assert high.deserta is False, "a 0.9 threshold must not flag this student"


def test_load_candidates_is_empty_for_a_version_without_any(fake_session):
    """A model trained before candidate storage existed must not crash the loader."""
    fake_session._scalars_result = []
    assert load_candidates(fake_session, 1) == []


def test_model_version_still_saves_without_candidates(fake_session, predictor, model_artifact):
    """Candidates are optional, so the existing single-model path keeps working."""
    _, metadata = model_artifact
    version = save_version(fake_session, predictor.pipeline, metadata)

    assert isinstance(version, ModelVersion)
    assert not [obj for obj in fake_session.added if isinstance(obj, ModelCandidate)]
