"""Authentication logic: credentials, token issuing, and refresh rotation."""

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from dropout_service.auth.models import RefreshToken
from dropout_service.auth.security import (
    hash_refresh_token,
    new_refresh_token,
    spend_dummy_verification,
    verify_password,
)
from dropout_service.config import get_settings
from dropout_service.users.models import User
from dropout_service.users.service import get_by_email

logger = logging.getLogger(__name__)


def authenticate(session: Session, email: str, password: str) -> User | None:
    """Return the user for these credentials, or None.

    Returns None for every failure -- unknown email, wrong password, deactivated account --
    and the caller must not say which. The dummy verification on the unknown-email path
    matters: skipping it would make a missing account measurably faster to reject than a
    wrong password, which leaks whether an address is registered.
    """
    user = get_by_email(session, email)
    if user is None:
        spend_dummy_verification()
        return None
    if not verify_password(password, user.password_hash):
        return None
    if not user.is_active:
        logger.info("Rejected login for deactivated account %s", user.id)
        return None
    return user


def issue_refresh_token(session: Session, user: User, replaces: RefreshToken | None = None) -> str:
    """Create a refresh-token row and return the raw token.

    The raw value is returned once and never stored; only its hash is persisted.
    """
    settings = get_settings()
    raw = new_refresh_token()
    row = RefreshToken(
        user_id=user.id,
        token_hash=hash_refresh_token(raw),
        expires_at=datetime.now(UTC) + timedelta(days=settings.refresh_token_days),
    )
    session.add(row)
    session.flush()
    if replaces is not None:
        replaces.revoked_at = datetime.now(UTC)
        replaces.replaced_by_id = row.id
    session.commit()
    return raw


def find_refresh_token(session: Session, raw_token: str) -> RefreshToken | None:
    """Look a refresh token up by its hash."""
    return session.scalars(
        select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(raw_token))
    ).first()


def revoke_family(session: Session, token: RefreshToken) -> int:
    """Revoke a token and every descendant in its rotation chain.

    Called when a revoked token is presented again. A legitimate client never replays an
    old token, so a replay means the value leaked -- and since rotation already issued a
    successor, the safe assumption is that an attacker holds one of them. Killing the whole
    chain forces a fresh login rather than leaving the thief with a working credential.
    """
    revoked = 0
    cursor: RefreshToken | None = token
    while cursor is not None:
        if cursor.revoked_at is None:
            cursor.revoked_at = datetime.now(UTC)
            revoked += 1
        nxt = cursor.replaced_by_id
        cursor = session.get(RefreshToken, nxt) if nxt else None
    session.commit()
    if revoked:
        logger.warning(
            "Revoked %d refresh token(s) for user %s after a replayed token", revoked, token.user_id
        )
    return revoked


def revoke_one(session: Session, token: RefreshToken) -> None:
    """Revoke a single token. Used by logout."""
    if token.revoked_at is None:
        token.revoked_at = datetime.now(UTC)
        session.commit()
