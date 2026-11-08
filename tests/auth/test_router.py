"""The authentication endpoints, with the session layer faked."""

import pytest
from fastapi.testclient import TestClient

from dropout_service.auth.router import REFRESH_COOKIE
from dropout_service.auth.security import hash_password
from dropout_service.main import create_app
from dropout_service.users.models import User

PASSWORD = "a-perfectly-fine-password"


@pytest.fixture
def auth_app(test_settings, predictor, monkeypatch):
    """An app whose auth service is backed by one in-memory user."""
    from dropout_service.auth import router as auth_router
    from dropout_service.auth import service as auth_service
    from dropout_service.dependencies import get_predictor, require_session

    user = User(
        id=1,
        email="adviser@universidad.edu.co",
        password_hash=hash_password(PASSWORD),
        role="adviser",
        is_active=True,
    )
    monkeypatch.setattr(
        auth_service,
        "get_by_email",
        lambda _s, email: user if email.strip().lower() == user.email else None,
    )
    # Patched on the *router* module, not the service: router.py did
    # `from ...service import issue_refresh_token`, so it holds its own reference and a
    # patch on the service module would not be seen.
    monkeypatch.setattr(auth_router, "issue_refresh_token", lambda *_a, **_k: "fake-refresh-token")

    app = create_app(test_settings)

    class _Session:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    app.dependency_overrides[get_predictor] = lambda: predictor
    app.dependency_overrides[require_session] = lambda: _Session()
    return app, user


def test_login_returns_a_token_and_sets_an_httponly_cookie(auth_app):
    app, user = auth_app
    with TestClient(app) as client:
        response = client.post("/auth/login", json={"email": user.email, "password": PASSWORD})

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] > 0
    assert body["user"]["email"] == user.email
    # The password hash must never leave the service.
    assert "password_hash" not in body["user"]

    cookie = response.headers["set-cookie"]
    assert REFRESH_COOKIE in cookie
    # The entire point of the design: JavaScript cannot read this.
    assert "HttpOnly" in cookie
    # The refresh token must not also be in the body, where JS could reach it.
    assert "fake-refresh-token" not in response.text


def test_login_rejects_a_wrong_password(auth_app):
    app, user = auth_app
    with TestClient(app) as client:
        response = client.post("/auth/login", json={"email": user.email, "password": "wrong"})
    assert response.status_code == 401


def test_unknown_email_and_wrong_password_are_indistinguishable(auth_app):
    """Different messages here would let anyone enumerate registered addresses."""
    app, user = auth_app
    with TestClient(app) as client:
        wrong = client.post("/auth/login", json={"email": user.email, "password": "wrong"})
        unknown = client.post(
            "/auth/login", json={"email": "nobody@universidad.edu.co", "password": "wrong"}
        )

    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def test_login_normalises_the_email_case(auth_app):
    """Nobody should fail to sign in because they capitalised their address."""
    app, user = auth_app
    with TestClient(app) as client:
        response = client.post(
            "/auth/login", json={"email": user.email.upper(), "password": PASSWORD}
        )
    assert response.status_code == 200


def test_login_rejects_a_malformed_email(auth_app):
    app, _ = auth_app
    with TestClient(app) as client:
        response = client.post("/auth/login", json={"email": "not-an-email", "password": "x"})
    assert response.status_code == 422


def test_me_requires_a_token(anon_client):
    assert anon_client.get("/auth/me").status_code == 401


def test_me_returns_the_signed_in_user(client):
    response = client.get("/auth/me")
    assert response.status_code == 200
    body = response.json()
    assert body["role"] == "adviser"
    assert "password_hash" not in body


def test_refresh_without_a_cookie_is_401(auth_app):
    app, _ = auth_app
    with TestClient(app) as client:
        response = client.post("/auth/refresh")
    assert response.status_code == 401


def test_logout_is_always_204(auth_app):
    """Signing out must work with an expired token, and must not reveal anything."""
    app, _ = auth_app
    with TestClient(app) as client:
        response = client.post("/auth/logout")
    assert response.status_code == 204
