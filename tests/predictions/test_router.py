"""HTTP contract for the prediction endpoints."""

from dropout_service.predictions.schemas import MAX_BATCH_SIZE
from dropout_service.router import API_V1_PREFIX

PREDICT = f"{API_V1_PREFIX}/predict"
BATCH = f"{API_V1_PREFIX}/predict/batch"
HISTORY = f"{API_V1_PREFIX}/predictions"


def test_predict_returns_a_complete_result(client, student):
    response = client.post(PREDICT, json=student)
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"deserta", "probabilidad", "riesgo"}
    assert isinstance(body["deserta"], bool)
    assert 0.0 <= body["probabilidad"] <= 1.0
    assert body["riesgo"] in {"bajo", "medio", "alto"}


def test_predict_works_without_the_optional_category(client, student):
    """~50% of real students have no critical area; this must not be an error."""
    del student["area_materia_critica"]
    assert client.post(PREDICT, json=student).status_code == 200


def test_predict_batch_preserves_order(client, student):
    good = dict(student, nota_promedio=4.9, porcentaje_asistencia=98.0)
    bad = dict(student, nota_promedio=1.1, porcentaje_asistencia=40.0)
    response = client.post(BATCH, json={"students": [good, bad]})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 2
    singles = [client.post(PREDICT, json=s).json()["probabilidad"] for s in (good, bad)]
    assert [p["probabilidad"] for p in body["predictions"]] == singles


def test_batch_size_is_capped(client, student):
    response = client.post(BATCH, json={"students": [student] * (MAX_BATCH_SIZE + 1)})
    assert response.status_code == 422


def test_empty_batch_is_rejected(client):
    assert client.post(BATCH, json={"students": []}).status_code == 422


def test_invalid_field_value_is_rejected(client, student):
    assert client.post(PREDICT, json=dict(student, semestre=-1)).status_code == 422


def test_unknown_category_is_rejected(client, student):
    assert client.post(PREDICT, json=dict(student, genero="Masculino")).status_code == 422


def test_unknown_field_is_rejected(client, student):
    assert client.post(PREDICT, json=dict(student, inventado=1)).status_code == 422


def test_missing_required_field_is_rejected(client, student):
    del student["nota_promedio"]
    assert client.post(PREDICT, json=student).status_code == 422


def test_history_requires_a_database(client):
    """Without DATABASE_URL this endpoint cannot work, and says so."""
    response = client.get(HISTORY)
    assert response.status_code == 503
    assert "DATABASE_URL" in response.json()["detail"]


def test_unprefixed_paths_are_not_served(client, student):
    """Business endpoints live under /api/v1 only -- guards the versioning decision."""
    assert client.post("/predict", json=student).status_code == 404
    assert client.get("/predictions").status_code == 404


def test_removed_records_endpoint_is_gone(client, student):
    assert client.post(f"{API_V1_PREFIX}/records", json=student).status_code == 404
