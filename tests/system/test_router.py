"""Operational endpoints and model metadata."""


def test_health_reports_a_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["database"] is None  # no DATABASE_URL in tests


def test_health_is_served_at_the_root(client):
    """The container's HEALTHCHECK calls /health, so it must stay at the root."""
    assert client.get("/health").status_code == 200


def test_root_advertises_the_service(client):
    body = client.get("/").json()
    assert body["health"] == "/health"
    assert body["docs"] == "/docs"


def test_model_info(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["n_features"] == len(body["features"])
    assert 0.0 < body["decision_threshold"] <= 1.0
    assert set(body["risk_bands"]) == {"bajo", "medio", "alto"}


def test_openapi_schema_is_served(client):
    assert client.get("/openapi.json").status_code == 200
