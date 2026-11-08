"""The refresh_tokens table."""

from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from dropout_service.database import SCHEMA, Base


def _utcnow() -> datetime:
    return datetime.now(UTC)


class RefreshToken(Base):
    """One issued refresh token.

    Kept in the database rather than being a self-contained JWT so that logout can actually
    revoke: a stateless refresh token stays valid until it expires, no matter what the user
    clicks. Rows also make rotation auditable -- ``replaced_by_id`` chains each token to its
    successor, so replaying a revoked token is detectable and means the token leaked.

    ``token_hash`` is a SHA-256 digest; see security.py for why not bcrypt.
    """

    __tablename__ = "refresh_tokens"
    __table_args__ = {"schema": SCHEMA}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey(f"{SCHEMA}.users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    #: Set when the token is rotated away or explicitly logged out.
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    #: The token issued in its place, forming the rotation chain.
    replaced_by_id: Mapped[int | None] = mapped_column(
        ForeignKey(f"{SCHEMA}.refresh_tokens.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    @property
    def is_usable(self) -> bool:
        """True when the token is neither revoked nor expired."""
        return self.revoked_at is None and self.expires_at > _utcnow()
