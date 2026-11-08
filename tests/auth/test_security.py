"""Password hashing and token handling.

These are the functions an attacker interacts with most directly, so the negative cases
matter more than the happy path.
"""

import time
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from dropout_service.auth import security
from dropout_service.config import get_settings


def test_password_round_trip():
    hashed = security.hash_password("a-correct-horse-battery-staple")
    assert hashed != "a-correct-horse-battery-staple"
    assert security.verify_password("a-correct-horse-battery-staple", hashed)


def test_wrong_password_is_rejected():
    hashed = security.hash_password("the-right-one")
    assert not security.verify_password("the-wrong-one", hashed)


def test_hashes_are_salted():
    """Two hashes of the same password must differ, or identical passwords are visible."""
    assert security.hash_password("same-input") != security.hash_password("same-input")


def test_verify_password_survives_a_malformed_hash():
    """A corrupt stored hash must be a failed login, not a 500."""
    assert not security.verify_password("anything", "not-a-bcrypt-hash")
    assert not security.verify_password("anything", "")


def test_access_token_round_trip():
    token, expires_in = security.create_access_token(user_id=7, role="admin", email="a@b.co")
    payload = security.decode_access_token(token)

    assert payload["sub"] == "7"
    assert payload["role"] == "admin"
    assert payload["email"] == "a@b.co"
    assert expires_in == get_settings().access_token_minutes * 60


def test_expired_token_is_rejected():
    """Minted in the past, so it is already expired when decoded."""
    settings = get_settings()
    payload = {
        "sub": "1",
        "role": "adviser",
        "email": "a@b.co",
        "iat": datetime.now(UTC) - timedelta(hours=2),
        "exp": datetime.now(UTC) - timedelta(hours=1),
    }
    expired = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)

    with pytest.raises(jwt.ExpiredSignatureError):
        security.decode_access_token(expired)


def test_tampered_token_is_rejected():
    token, _ = security.create_access_token(user_id=1, role="adviser", email="a@b.co")
    with pytest.raises(jwt.InvalidTokenError):
        security.decode_access_token(token[:-1] + ("A" if token[-1] != "A" else "B"))


def test_token_signed_with_another_key_is_rejected():
    """The signature is what makes the claims trustworthy."""
    forged = jwt.encode(
        {"sub": "1", "exp": datetime.now(UTC) + timedelta(hours=1)},
        "a-completely-different-signing-key",
        algorithm="HS256",
    )
    with pytest.raises(jwt.InvalidTokenError):
        security.decode_access_token(forged)


def test_unsigned_token_is_rejected():
    """The classic JWT forgery: alg=none with no signature.

    This only fails because decode_access_token pins `algorithms=[...]`. Letting PyJWT
    honour the token's own alg header would make this forgery succeed.
    """
    unsigned = jwt.encode(
        {"sub": "1", "role": "admin", "exp": datetime.now(UTC) + timedelta(hours=1)},
        key="",
        algorithm="none",
    )
    with pytest.raises(jwt.InvalidTokenError):
        security.decode_access_token(unsigned)


def test_token_without_required_claims_is_rejected():
    """A token with no exp would never expire, so it must not be accepted."""
    settings = get_settings()
    no_exp = jwt.encode({"sub": "1"}, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    with pytest.raises(jwt.MissingRequiredClaimError):
        security.decode_access_token(no_exp)


def test_refresh_tokens_are_unique_and_opaque():
    first, second = security.new_refresh_token(), security.new_refresh_token()
    assert first != second
    assert len(first) >= 40
    # Not a JWT: it carries entropy, not claims.
    assert first.count(".") == 0


def test_refresh_token_hash_is_deterministic_and_one_way():
    raw = security.new_refresh_token()
    assert security.hash_refresh_token(raw) == security.hash_refresh_token(raw)
    assert len(security.hash_refresh_token(raw)) == 64  # sha256 hex
    assert raw not in security.hash_refresh_token(raw)


def test_dummy_verification_costs_real_time():
    """Guards the no-enumeration property.

    If this ever became a no-op, an unknown email would return measurably faster than a
    wrong password and anyone could enumerate registered addresses.
    """
    started = time.perf_counter()
    security.spend_dummy_verification()
    assert time.perf_counter() - started > 0.005
