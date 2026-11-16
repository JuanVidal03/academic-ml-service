"""The /predict/batch/compare endpoint."""

from tests.predictions.test_compare import candidate, with_candidates

BATCH_COMPARE = "/predict/batch/compare"


def test_requires_authentication(anon_client, student):
    assert anon_client.post(BATCH_COMPARE, json={"students": [student]}).status_code == 401


def test_returns_a_comparison_per_student_in_order(client, predictor, student):
    with_candidates(
        client,
        predictor,
        [
            candidate(predictor, "LogisticRegression", 0.35, True),
            candidate(predictor, "RandomForest", 0.38, False),
            candidate(predictor, "SVM", 0.25, False),
        ],
    )
    # Three students, distinguishable by a field that moves the probability.
    students = [
        {**student, "nota_promedio": 4.6},
        {**student, "nota_promedio": 3.0},
        {**student, "nota_promedio": 1.4},
    ]

    response = client.post(BATCH_COMPARE, json={"students": students})

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert len(body["comparisons"]) == 3
    for comparison in body["comparisons"]:
        assert [m["algorithm"] for m in comparison["models"]] == [
            "LogisticRegression",
            "RandomForest",
            "SVM",
        ]
        assert sum(1 for m in comparison["models"] if m["is_best"]) == 1

    # Order is the submitted order, not sorted by risk: a failing student must not be
    # silently reordered to the top, or the results stop lining up with the input rows.
    probabilities = [c["models"][0]["probabilidad"] for c in body["comparisons"]]
    assert probabilities == sorted(probabilities), probabilities


def test_each_model_applies_its_own_threshold(client, predictor, student):
    # All three share a pipeline, so probabilities are identical; only the cut-points differ.
    with_candidates(
        client,
        predictor,
        [
            candidate(predictor, "Strict", 0.99, True),
            candidate(predictor, "Lenient", 0.01, False),
        ],
    )

    body = client.post(BATCH_COMPARE, json={"students": [student, student]}).json()

    for comparison in body["comparisons"]:
        by_name = {m["algorithm"]: m for m in comparison["models"]}
        assert by_name["Strict"]["probabilidad"] == by_name["Lenient"]["probabilidad"]
        assert by_name["Strict"]["deserta"] is False
        assert by_name["Lenient"]["deserta"] is True


def test_matches_single_compare_for_the_same_student(client, predictor, student):
    """A batch of one must equal /predict/compare, or the two screens disagree."""
    candidates = [
        candidate(predictor, "LogisticRegression", 0.35, True),
        candidate(predictor, "SVM", 0.25, False),
    ]
    with_candidates(client, predictor, candidates)

    single = client.post("/predict/compare", json=student).json()
    batched = client.post(BATCH_COMPARE, json={"students": [student]}).json()

    assert batched["comparisons"][0] == single


def test_503_when_no_candidates_are_stored(client, predictor, student):
    with_candidates(client, predictor, [])

    response = client.post(BATCH_COMPARE, json={"students": [student]})

    assert response.status_code == 503
    assert "Reentrena" in response.json()["detail"]


def test_rejects_more_than_one_hundred_students(client, predictor, student):
    with_candidates(client, predictor, [candidate(predictor, "LogisticRegression", 0.35, True)])

    response = client.post(BATCH_COMPARE, json={"students": [student] * 101})

    assert response.status_code == 422
