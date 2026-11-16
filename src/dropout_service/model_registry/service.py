"""Saving and loading trained models.

This is the only place that knows the artefact lives in a database. ``ml/predict.py`` takes
bytes, so the ML package stays free of SQLAlchemy -- see ``tests/test_layering.py``.
"""

import logging
from collections.abc import Sequence
from typing import Any, NamedTuple

from sklearn.pipeline import Pipeline
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from dropout_service.ml.predict import Predictor
from dropout_service.model_registry.models import ModelCandidate, ModelVersion

logger = logging.getLogger(__name__)


class LoadedCandidate(NamedTuple):
    """One stored algorithm, ready to predict."""

    algorithm: str
    predictor: Predictor
    cv_metrics: dict[str, Any]
    test_metrics: dict[str, Any]
    is_best: bool


def save_version(
    session: Session,
    pipeline: Pipeline,
    metadata: dict[str, Any],
    activate: bool = True,
    candidates: Sequence[Any] = (),
) -> ModelVersion:
    """Store a trained pipeline and, by default, make it the active one.

    Deactivating the previous row happens before the insert, in the same transaction, so the
    partial unique index on ``is_active`` is never momentarily violated.
    """
    if activate:
        session.execute(
            update(ModelVersion).where(ModelVersion.is_active.is_(True)).values(is_active=False)
        )

    version = ModelVersion(
        version=str(metadata["model_version"]),
        algorithm=str(metadata["model_name"]),
        artifact=Predictor.dump(pipeline),
        metrics=metadata,
        decision_threshold=float(metadata["decision_threshold"]),
        n_features=int(metadata["n_features"]),
        is_active=activate,
    )
    session.add(version)
    # Flush rather than commit, so the candidates land in the same transaction: a retrain
    # must not leave a version with a partial set of candidates.
    session.flush()

    for candidate in candidates:
        session.add(
            ModelCandidate(
                model_version_id=version.id,
                algorithm=candidate.algorithm,
                artifact=Predictor.dump(candidate.pipeline),
                cv_metrics=candidate.cv_metrics,
                test_metrics=candidate.test_metrics,
                decision_threshold=candidate.decision_threshold,
                is_best=candidate.is_best,
            )
        )

    session.commit()
    logger.info(
        "Saved model version %s (%s, %d bytes, active=%s) with %d candidate(s)",
        version.version,
        version.algorithm,
        len(version.artifact),
        activate,
        len(candidates),
    )
    return version


def load_active(session: Session) -> Predictor | None:
    """Return a predictor built from the active model version, or None if there is none."""
    row = session.scalars(
        select(ModelVersion).where(ModelVersion.is_active.is_(True)).limit(1)
    ).first()
    if row is None:
        logger.warning(
            "No active model version. Train one first: uv run scripts/train.py",
        )
        return None
    return Predictor.from_bytes(row.artifact, row.metrics, model_version_id=row.id)


def load_candidates(session: Session, model_version_id: int) -> list[LoadedCandidate]:
    """Load every stored algorithm for a version, best first.

    Ordered in SQL so the API does not have to re-sort, and so "best first" is the same
    wherever it is read from.
    """
    rows = session.scalars(
        select(ModelCandidate)
        .where(ModelCandidate.model_version_id == model_version_id)
        .order_by(ModelCandidate.is_best.desc(), ModelCandidate.id)
    ).all()

    loaded: list[LoadedCandidate] = []
    for row in rows:
        loaded.append(
            LoadedCandidate(
                algorithm=row.algorithm,
                # The threshold goes in via metadata, which Predictor already reads --
                # one deserialisation, and each candidate keeps the cut-point it was
                # actually calibrated with.
                predictor=Predictor.from_bytes(
                    row.artifact,
                    {
                        **row.test_metrics,
                        "model_name": row.algorithm,
                        "decision_threshold": row.decision_threshold,
                    },
                ),
                cv_metrics=row.cv_metrics,
                test_metrics=row.test_metrics,
                is_best=row.is_best,
            )
        )
    if loaded:
        logger.info("Loaded %d candidate model(s) for version %s", len(loaded), model_version_id)
    return loaded
