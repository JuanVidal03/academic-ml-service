"""Tables owned by the predictions slice.

Imported explicitly by ``alembic/env.py`` so the table is registered on
``Base.metadata`` before migrations are generated.
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from dropout_service.database import SCHEMA, Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Prediction(Base):
    """One prediction served by the API, kept for auditing and drift monitoring."""

    __tablename__ = "predictions"
    __table_args__ = {"schema": SCHEMA}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    input_data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    prediction: Mapped[bool] = mapped_column(Boolean, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(10), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    # Indexed because the history endpoint orders by it.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False, index=True
    )
