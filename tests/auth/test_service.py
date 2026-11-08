"""Authentication and refresh rotation, against a fake session."""

from datetime import UTC, datetime, timedelta

from dropout_service.auth import service
from dropout_service.auth.models import RefreshToken
from dropout_service.auth.security import hash_password, hash_refresh_token
from dropout_service.users.models import User

PASSWORD = "a-perfectly-fine-password"


def make_user(**overrides) -> User:
    defaults = {
        "id": 1,
        "email": "adviser@universidad.edu.co",
        "password_hash": hash_password(PASSWORD),
        "role": "adviser",
        "is_active": True,
    }
    return User(**{**defaults, **overrides})


class AuthSession:
    """Just enough Session for the auth service: lookups, adds, and get-by-id."""

    def __init__(self, user: User | None = None, tokens: list[RefreshToken] | None = None):
        self.user = user
        self.tokens = {t.id: t for t in (tokens or [])}
        self.added: list = []
        self.commits = 0
        self._next_id = max(self.tokens, default=0) + 1

    # -- query surface --------------------------------------------------
    def scalars(self, _statement):
        return self

    def first(self):
        # The auth service only ever selects a User or a RefreshToken by a unique value;
        # returning the configured one is enough for these tests.
        return self._pending_result

    def execute(self, _statement):
        return self

    # -- mutation surface ----------------------------------------------
    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = self._next_id
                self._next_id += 1
                if isinstance(obj, RefreshToken):
                    self.tokens[obj.id] = obj

    def commit(self):
        self.commits += 1
        self.flush()

    def get(self, _model, pk):
        return self.tokens.get(pk)

    _pending_result = None


def test_authenticate_accepts_correct_credentials(monkeypatch):
    user = make_user()
    session = AuthSession()
    monkeypatch.setattr(service, "get_by_email", lambda _s, _e: user)

    assert service.authenticate(session, user.email, PASSWORD) is user


def test_authenticate_rejects_a_wrong_password(monkeypatch):
    user = make_user()
    monkeypatch.setattr(service, "get_by_email", lambda _s, _e: user)

    assert service.authenticate(AuthSession(), user.email, "not-the-password") is None


def test_authenticate_rejects_a_deactivated_account(monkeypatch):
    """A disabled account must not be able to sign in even with the right password."""
    user = make_user(is_active=False)
    monkeypatch.setattr(service, "get_by_email", lambda _s, _e: user)

    assert service.authenticate(AuthSession(), user.email, PASSWORD) is None


def test_unknown_email_still_spends_a_hash(monkeypatch):
    """No user enumeration.

    The dummy verification is the whole mechanism: without it an unknown email returns
    without hashing and is measurably faster than a wrong password, which tells an attacker
    which addresses are registered.
    """
    calls: list[bool] = []
    monkeypatch.setattr(service, "get_by_email", lambda _s, _e: None)
    monkeypatch.setattr(service, "spend_dummy_verification", lambda: calls.append(True))

    assert service.authenticate(AuthSession(), "nobody@universidad.edu.co", "x") is None
    assert calls == [True], "an unknown email must still cost a verification"


def test_issue_refresh_token_stores_only_a_hash():
    """A database leak must not hand over usable refresh tokens."""
    user = make_user()
    session = AuthSession()

    raw = service.issue_refresh_token(session, user)
    stored = session.added[0]

    assert stored.token_hash == hash_refresh_token(raw)
    assert raw not in stored.token_hash
    assert stored.user_id == user.id
    assert stored.expires_at > datetime.now(UTC)


def test_rotation_revokes_the_previous_token_and_links_the_chain():
    user = make_user()
    session = AuthSession()
    first_raw = service.issue_refresh_token(session, user)
    first = session.added[0]

    second_raw = service.issue_refresh_token(session, user, replaces=first)
    second = session.added[1]

    assert second_raw != first_raw
    assert first.revoked_at is not None, "the rotated-away token must be revoked"
    assert first.replaced_by_id == second.id, "the chain must be linked"
    assert second.revoked_at is None


def test_revoke_family_walks_the_whole_chain():
    """Replaying a revoked token means it leaked, so every successor dies too."""
    user = make_user()
    session = AuthSession()
    first_raw = service.issue_refresh_token(session, user)
    first = session.added[0]
    service.issue_refresh_token(session, user, replaces=first)
    second = session.added[1]
    service.issue_refresh_token(session, user, replaces=second)
    third = session.added[2]

    # Un-revoke the head so there is something for revoke_family to do.
    first.revoked_at = None
    revoked = service.revoke_family(session, first)

    assert revoked >= 1
    assert third.revoked_at is not None, "the newest token in the chain must be revoked"
    assert first_raw is not None


def test_is_usable_rejects_expired_and_revoked_tokens():
    live = RefreshToken(
        id=1, user_id=1, token_hash="a", expires_at=datetime.now(UTC) + timedelta(days=1)
    )
    expired = RefreshToken(
        id=2, user_id=1, token_hash="b", expires_at=datetime.now(UTC) - timedelta(seconds=1)
    )
    revoked = RefreshToken(
        id=3,
        user_id=1,
        token_hash="c",
        expires_at=datetime.now(UTC) + timedelta(days=1),
        revoked_at=datetime.now(UTC),
    )

    assert live.is_usable
    assert not expired.is_usable
    assert not revoked.is_usable
