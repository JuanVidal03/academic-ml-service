"""The users table.

Storage only: login, tokens and route protection are a separate piece of work. Create the
first account with ``uv run scripts/create_user.py``.
"""

from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from dropout_service.database import SCHEMA, Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class User(Base):
    """Someone who may use the service."""

    __tablename__ = "users"
    __table_args__ = {"schema": SCHEMA}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    #: A bcrypt hash. Never a plaintext or SHA-family digest.
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="adviser")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
