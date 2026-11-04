"""Application-level error handling."""

from fastapi.testclient import TestClient

from dropout_service.config import Settings
from dropout_service.main import create_app
from dropout_service.router import API_V1_PREFIX

# A syntactically valid URL that cannot connect: port 1 on localhost.
UNREACHABLE_DB = "postgresql://nobody@127.0.0.1:1/absent"


def test_unreachable_database_returns_503_not_500(models_dir):
    """A configured-but-down database must be reported as a service fault.

    Previously this surfaced as an opaque 500, which gave the caller no way to tell a
    broken dependency from a bad request -- and in production an unreachable database is
    by far the more likely cause.
    """
    settings = Settings(models_dir=models_dir, database_url=UNREACHABLE_DB, log_level="CRITICAL")
    with TestClient(create_app(settings)) as client:
        response = client.get(f"{API_V1_PREFIX}/predictions")

    assert response.status_code == 503
    assert response.json()["detail"] == "Database unavailable."


def test_predictions_still_work_when_the_database_is_down(models_dir, student):
    """Logging is best-effort, so a dead database must not break predicting."""
    settings = Settings(models_dir=models_dir, database_url=UNREACHABLE_DB, log_level="CRITICAL")
    with TestClient(create_app(settings)) as client:
        response = client.post(f"{API_V1_PREFIX}/predict", json=student)

    assert response.status_code == 200
    assert 0.0 <= response.json()["probabilidad"] <= 1.0


def test_health_reports_a_broken_database_without_failing(models_dir):
    """/health must stay 200 so operators can read *why* it is unhealthy."""
    settings = Settings(models_dir=models_dir, database_url=UNREACHABLE_DB, log_level="CRITICAL")
    with TestClient(create_app(settings)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["database"]["connected"] is False
    assert "error" in body["database"]
