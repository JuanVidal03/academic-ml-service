"""Application-level error handling."""

from fastapi.testclient import TestClient

from dropout_service.main import create_app

# A syntactically valid URL that cannot connect: port 1 on localhost.
UNREACHABLE_DB = "postgresql://nobody@127.0.0.1:1/absent"


def test_unreachable_database_returns_503_not_500(settings_factory):
    """A configured-but-down database must be reported as a service fault.

    Previously this surfaced as an opaque 500, which gave the caller no way to tell a
    broken dependency from a bad request -- and in production an unreachable database is
    by far the more likely cause. The admin dependency is overridden so the request gets
    past authentication and actually reaches the database.
    """
    from dropout_service.dependencies import get_current_user
    from dropout_service.users.models import User

    settings = settings_factory(database_url=UNREACHABLE_DB, log_level="CRITICAL")
    app = create_app(settings)
    app.dependency_overrides[get_current_user] = lambda: User(
        id=1, email="admin@universidad.edu.co", password_hash="x", role="admin", is_active=True
    )
    with TestClient(app) as client:
        response = client.get("/predictions")

    assert response.status_code == 503
    assert response.json()["detail"] == "Database unavailable."


def test_predictions_still_work_when_the_database_is_down(client, student):
    """Logging is best-effort, so a dead database must not break predicting.

    The client fixture already points at an unreachable database, with the predictor
    injected -- which is exactly the degraded-but-serving state this asserts.
    """
    response = client.post("/predict", json=student)

    assert response.status_code == 200
    assert 0.0 <= response.json()["probabilidad"] <= 1.0


def test_health_reports_a_broken_database_without_failing(settings_factory):
    """/health must stay 200 so operators can read *why* it is unhealthy."""
    settings = settings_factory(database_url=UNREACHABLE_DB, log_level="CRITICAL")
    with TestClient(create_app(settings)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["database"]["connected"] is False
    assert "error" in body["database"]
