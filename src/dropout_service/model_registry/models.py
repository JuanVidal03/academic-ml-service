"""The model_versions table: trained artefacts and their metrics."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Index, Integer, LargeBinary, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from dropout_service.database import SCHEMA, Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ModelVersion(Base):
    """One trained model.

    ``artifact`` holds the joblib bytes rather than JSON because a fitted scikit-learn
    pipeline is not JSON-serialisable: it carries numpy arrays, learned one-hot categories,
    scaler statistics and a reference to ``ml/schema.py:coerce_missing_to_nan``. Storing the
    bytes keeps the artefact byte-exact and works whichever algorithm wins the comparison.
    ``metrics`` *is* naturally JSON, so it gets a queryable JSONB column.
    """

    __tablename__ = "model_versions"
    __table_args__ = (
        # Exactly one active model, enforced by the database rather than by hope: a partial
        # unique index over the rows where is_active is true.
        Index(
            "uq_model_versions_single_active",
            "is_active",
            unique=True,
            postgresql_where=text("is_active"),
        ),
        {"schema": SCHEMA},
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    algorithm: Mapped[str] = mapped_column(String(64), nullable=False)
    artifact: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    decision_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    n_features: Mapped[int] = mapped_column(Integer, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
