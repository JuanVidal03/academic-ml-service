"""Tables for the training dataset and the analysis reports."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from dropout_service.database import SCHEMA, Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class TrainingRecord(Base):
    """One labelled student-term used for training.

    The 34 features live in a single JSONB column rather than 34 mirrored columns. That is
    deliberate: ``ml/schema.py:StudentFeatures`` is the single source of truth for the
    feature set, and a mirrored table would be a second copy of that list -- exactly the
    duplication that once let the training and serving defaults drift apart and silently
    corrupt half the predictions. It also means adding a feature needs no migration.
    Individual features stay queryable via ``features->>'nota_promedio'``.
    """

    __tablename__ = "training_records"
    __table_args__ = {"schema": SCHEMA}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    student_id: Mapped[int | None] = mapped_column(
        ForeignKey(f"{SCHEMA}.students.id", ondelete="SET NULL"), nullable=True, index=True
    )
    periodo: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    features: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    deserto: Mapped[bool] = mapped_column(Boolean, nullable=False)
    #: Where the row came from: "csv" for the seeded dataset, so verified outcomes added
    #: later can be told apart.
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="csv")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )


class AnalysisReport(Base):
    """One run of the exploratory analysis, kept so retrains can be compared."""

    __tablename__ = "analysis_reports"
    __table_args__ = {"schema": SCHEMA}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    n_records: Mapped[int] = mapped_column(Integer, nullable=False)
    dropout_rate: Mapped[float] = mapped_column(Float, nullable=False)
    report: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
