"""Saving and loading trained models.

This is the only place that knows the artefact lives in a database. ``ml/predict.py`` takes
bytes, so the ML package stays free of SQLAlchemy -- see ``tests/test_layering.py``.
"""

import logging
from typing import Any

from sklearn.pipeline import Pipeline
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from dropout_service.ml.predict import Predictor
from dropout_service.model_registry.models import ModelVersion

logger = logging.getLogger(__name__)


def save_version(
    session: Session, pipeline: Pipeline, metadata: dict[str, Any], activate: bool = True
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
    session.commit()
    logger.info(
        "Saved model version %s (%s, %d bytes, active=%s)",
        version.version,
        version.algorithm,
        len(version.artifact),
        activate,
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
