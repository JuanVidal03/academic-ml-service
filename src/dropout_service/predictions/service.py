"""Business logic for the predictions slice.

Kept out of ``router.py`` so it can be exercised without an HTTP request, and so the
handlers stay request/response plumbing.
"""

import logging
from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from dropout_service.ml.predict import PredictionResult
from dropout_service.ml.schema import StudentFeatures
from dropout_service.predictions.models import Prediction

logger = logging.getLogger(__name__)


def log_predictions(
    factory: sessionmaker | None,
    students: Sequence[StudentFeatures],
    results: Sequence[PredictionResult],
    model_version: str,
) -> None:
    """Persist one audit row per prediction, in a single transaction.

    Intended to run *after* the response is sent, so database latency never delays a
    prediction. A failure here is logged but never raised: the prediction itself already
    succeeded, and losing an audit row must not turn a successful request into an error.

    Does nothing when no database is configured.
    """
    if factory is None:
        return
    try:
        with factory() as session:
            session.add_all(
                [
                    Prediction(
                        input_data=student.model_dump(mode="json"),
                        probability=result.probabilidad,
                        prediction=result.deserta,
                        risk_level=result.riesgo,
                        model_version=model_version,
                    )
                    for student, result in zip(students, results, strict=True)
                ]
            )
            session.commit()
    except Exception:
        logger.exception("Failed to write %d prediction log row(s)", len(results))


def fetch_predictions_page(
    session: Session, page: int, page_size: int
) -> tuple[Sequence[Prediction], int]:
    """Return one page of logged predictions (newest first) and the total row count."""
    total = session.scalar(select(func.count()).select_from(Prediction)) or 0
    rows = session.scalars(
        select(Prediction)
        .order_by(Prediction.created_at.desc(), Prediction.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return rows, total
