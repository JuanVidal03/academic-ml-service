"""Operational endpoints and model metadata."""

from dropout_service.router import API_V1_PREFIX


def test_health_reports_a_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["database"] is None  # no DATABASE_URL in tests


def test_health_stays_unprefixed(client):
    """Railway's healthcheckPath points at /health; moving it would break deploys."""
    assert client.get("/health").status_code == 200
    assert client.get(f"{API_V1_PREFIX}/health").status_code == 404


def test_root_advertises_the_api(client):
    body = client.get("/").json()
    assert body["health"] == "/health"
    assert body["api"] == API_V1_PREFIX


def test_model_info(client):
    response = client.get(f"{API_V1_PREFIX}/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["n_features"] == len(body["features"])
    assert 0.0 < body["decision_threshold"] <= 1.0
    assert set(body["risk_bands"]) == {"bajo", "medio", "alto"}


def test_openapi_schema_is_served_under_the_version_prefix(client):
    assert client.get(f"{API_V1_PREFIX}/openapi.json").status_code == 200
