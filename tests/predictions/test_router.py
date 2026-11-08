"""HTTP contract for the prediction endpoints."""

from dropout_service.predictions.schemas import MAX_BATCH_SIZE

PREDICT = "/predict"
BATCH = "/predict/batch"
HISTORY = "/predictions"


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


def test_history_returns_a_page(admin_client):
    """Served from a faked session, so no database is involved."""
    response = admin_client.get(HISTORY)
    assert response.status_code == 200
    body = response.json()
    assert body["page"] == 1
    assert body["items"] == []
    assert body["total"] == 0


def test_predict_links_the_audit_row_to_a_student(client, student):
    """A supplied codigo must reach the audit row without altering the prediction."""
    without = client.post(PREDICT, json=student).json()
    with_code = client.post(PREDICT, json=dict(student, codigo="ABC123")).json()
    assert without["probabilidad"] == with_code["probabilidad"]


def test_codigo_is_not_a_model_feature():
    """If it leaked into the feature lists it would change the model."""
    from dropout_service.ml.schema import ALL_FEATURES

    assert "codigo" not in ALL_FEATURES


def test_removed_records_endpoint_is_gone(client, student):
    assert client.post("/records", json=student).status_code == 404


def test_prediction_endpoints_require_authentication(anon_client, student):
    """The whole point of the auth work: no token, no predictions."""
    assert anon_client.post(PREDICT, json=student).status_code == 401
    assert anon_client.post(BATCH, json={"students": [student]}).status_code == 401
    assert anon_client.get(HISTORY).status_code == 401


def test_history_is_admin_only(client, student):
    """An adviser may predict but must not read the whole audit log.

    403, not 401: the caller proved who they are and simply is not permitted, so telling
    them to re-authenticate would be misleading.
    """
    assert client.post(PREDICT, json=student).status_code == 200
    assert client.get(HISTORY).status_code == 403
