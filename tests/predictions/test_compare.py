"""The /predict/compare endpoint."""

from dropout_service.model_registry.service import LoadedCandidate

COMPARE = "/predict/compare"


def candidate(predictor, algorithm: str, threshold: float, is_best: bool) -> LoadedCandidate:
    """A LoadedCandidate sharing one fitted pipeline but with its own threshold.

    Sharing the pipeline is deliberate: it means every model returns the same probability,
    so any difference in verdict can only come from the threshold -- which is exactly the
    behaviour under test.
    """
    from dropout_service.ml.predict import Predictor

    return LoadedCandidate(
        algorithm=algorithm,
        predictor=Predictor(
            pipeline=predictor.pipeline,
            metadata={"model_name": algorithm},
            threshold=threshold,
        ),
        cv_metrics={"roc_auc_mean": 0.774 if is_best else 0.70},
        test_metrics={"roc_auc": 0.7695},
        is_best=is_best,
    )


def with_candidates(client, predictor, candidates):
    client.app.state.candidates = candidates
    return client


def test_requires_authentication(anon_client, student):
    assert anon_client.post(COMPARE, json=student).status_code == 401


def test_returns_every_model_best_first(client, predictor, student):
    with_candidates(
        client,
        predictor,
        [
            candidate(predictor, "LogisticRegression", 0.35, True),
            candidate(predictor, "RandomForest", 0.38, False),
            candidate(predictor, "SVM", 0.34, False),
        ],
    )

    response = client.post(COMPARE, json=student)
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["best_algorithm"] == "LogisticRegression"
    assert [m["algorithm"] for m in body["models"]] == [
        "LogisticRegression",
        "RandomForest",
        "SVM",
    ]
    assert body["models"][0]["is_best"] is True
    assert sum(1 for m in body["models"] if m["is_best"]) == 1


def test_each_model_applies_its_own_threshold(client, predictor, student):
    """The correctness point behind storing a threshold per candidate.

    All three share one pipeline, so the probability is identical; only the cut-points
    differ. A model whose threshold sits below the probability must flag the student and
    one whose threshold sits above must not.
    """
    with_candidates(
        client,
        predictor,
        [
            candidate(predictor, "Strict", 0.99, True),
            candidate(predictor, "Lenient", 0.01, False),
        ],
    )

    body = client.post(COMPARE, json=student).json()
    by_name = {m["algorithm"]: m for m in body["models"]}

    assert by_name["Strict"]["probabilidad"] == by_name["Lenient"]["probabilidad"]
    assert by_name["Strict"]["deserta"] is False
    assert by_name["Lenient"]["deserta"] is True
    assert by_name["Strict"]["decision_threshold"] == 0.99
    assert by_name["Lenient"]["decision_threshold"] == 0.01


def test_best_model_agrees_with_the_single_prediction_endpoint(client, predictor, student):
    """If these diverge, /predict/compare is not using the deployed model."""
    with_candidates(client, predictor, [candidate(predictor, "LogisticRegression", 0.4, True)])

    single = client.post("/predict", json=student).json()
    best = client.post(COMPARE, json=student).json()["models"][0]

    assert best["probabilidad"] == single["probabilidad"]
    assert best["riesgo"] == single["riesgo"]


def test_reports_clearly_when_no_candidates_are_stored(client, student):
    """A model trained before candidate storage must explain itself, not 500."""
    client.app.state.candidates = []

    response = client.post(COMPARE, json=student)
    assert response.status_code == 503
    assert "train.py" in response.json()["detail"]


def test_each_model_carries_its_metrics(client, predictor, student):
    with_candidates(client, predictor, [candidate(predictor, "LogisticRegression", 0.35, True)])

    model = client.post(COMPARE, json=student).json()["models"][0]
    assert model["cv_metrics"]["roc_auc_mean"] == 0.774
    assert model["test_metrics"]["roc_auc"] == 0.7695
