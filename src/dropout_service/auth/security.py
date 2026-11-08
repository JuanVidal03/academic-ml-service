"""Password hashing, access tokens, and refresh-token secrets.

Two different hashes are used here on purpose, and it looks inconsistent until you see why:

* **Passwords use bcrypt.** They are low-entropy and human-chosen, so the hash has to be
  *slow* to make offline guessing expensive.
* **Refresh tokens use SHA-256.** They are 32 bytes from a CSPRNG, so there is nothing to
  guess and no need for a work factor -- and bcrypt would tax every single refresh for no
  security gain. (bcrypt also silently truncates at 72 bytes, which a token could exceed.)
"""

import hashlib
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt

from dropout_service.config import get_settings

#: A real bcrypt hash of a value nobody knows, used to spend the same time verifying a
#: password for an account that does not exist. See service.authenticate.
_DUMMY_HASH = bcrypt.hashpw(secrets.token_bytes(32), bcrypt.gensalt()).decode()

TOKEN_TYPE = "bearer"  # noqa: S105 - a scheme name, not a credential


def hash_password(plaintext: str) -> str:
    """Hash a password with bcrypt."""
    return bcrypt.hashpw(plaintext.encode(), bcrypt.gensalt()).decode()


def verify_password(plaintext: str, password_hash: str) -> bool:
    """Check a password against its hash. Never raises on a malformed hash."""
    try:
        return bcrypt.checkpw(plaintext.encode(), password_hash.encode())
    except (ValueError, TypeError):
        return False


def spend_dummy_verification() -> None:
    """Burn roughly one bcrypt verification.

    Called when the email is unknown so that a missing account and a wrong password take
    comparable time. Without it, response latency tells an attacker which emails exist.
    """
    verify_password("does-not-matter", _DUMMY_HASH)


def create_access_token(user_id: int, role: str, email: str) -> tuple[str, int]:
    """Mint a short-lived access token. Returns ``(token, expires_in_seconds)``."""
    settings = get_settings()
    expires_in = settings.access_token_minutes * 60
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "role": role,
        "email": email,
        "iat": now,
        "exp": now + timedelta(seconds=expires_in),
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, expires_in


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and verify an access token.

    ``algorithms`` is pinned deliberately. Letting PyJWT honour the token's own ``alg``
    header is the classic forgery route -- an attacker sets ``alg: none`` or downgrades to a
    weaker algorithm and signs their own token.

    Raises:
        jwt.InvalidTokenError: expired, tampered, or otherwise unacceptable.
    """
    settings = get_settings()
    return jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[settings.jwt_algorithm],
        options={"require": ["exp", "sub"]},
    )


def new_refresh_token() -> str:
    """Generate an opaque refresh token. Not a JWT: it carries no claims, only entropy."""
    return secrets.token_urlsafe(32)


def hash_refresh_token(token: str) -> str:
    """Hash a refresh token for storage, so a database leak does not yield usable tokens."""
    return hashlib.sha256(token.encode()).hexdigest()
