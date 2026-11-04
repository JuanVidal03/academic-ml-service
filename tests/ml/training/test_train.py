"""Training pipeline."""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from dropout_service.ml.schema import build_pipeline
from dropout_service.ml.training.train import PARAM_GRIDS, evaluate, select_threshold


def test_every_candidate_has_a_parameter_grid():
    """Keeps the open/closed property: a new algorithm is tuned, not silently defaulted."""
    from dropout_service.ml.training.train import CANDIDATES

    assert set(CANDIDATES) == set(PARAM_GRIDS)


def test_every_grid_key_is_a_real_pipeline_parameter():
    """Catches typos and wrong nesting depth in PARAM_GRIDS.

    A misspelled key makes GridSearchCV raise at fit time, i.e. minutes into a training
    run. Wrapped estimators need a deeper prefix (the calibrated SVC is
    classifier__estimator__C, not classifier__C), which is easy to get wrong.
    """
    from dropout_service.ml.training.train import CANDIDATES

    for name, grid in PARAM_GRIDS.items():
        available = build_pipeline(CANDIDATES[name]).get_params(deep=True)
        for key in grid:
            assert key in available, f"{name}: {key!r} is not a pipeline parameter"


def test_threshold_selection_hits_the_recall_target(dataset):
    X, y = dataset
    sample = X.sample(n=600, random_state=0)
    X_s, y_s = sample, y.loc[sample.index]

    pipeline = build_pipeline(LogisticRegression(max_iter=1000, random_state=0))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    threshold = select_threshold(pipeline, X_s, y_s, cv, target_recall=0.80)

    assert 0.0 < threshold <= 1.0
    # A recall target above the default 0.5 operating point implies a lower threshold.
    assert threshold < 0.5


def test_evaluate_uses_the_given_threshold(dataset):
    X, y = dataset
    sample = X.sample(n=400, random_state=1)
    X_s, y_s = sample, y.loc[sample.index]
    pipeline = build_pipeline(LogisticRegression(max_iter=1000, random_state=0)).fit(X_s, y_s)

    permissive = evaluate(pipeline, X_s, y_s, threshold=0.2)
    strict = evaluate(pipeline, X_s, y_s, threshold=0.8)

    assert permissive["recall"] > strict["recall"]
    # ROC-AUC is threshold independent, so it must not move.
    assert permissive["roc_auc"] == strict["roc_auc"]


def test_run_training_returns_a_fitted_pipeline_and_metadata(raw_frame, monkeypatch):
    """End-to-end on a reduced search space, so the test stays fast.

    run_training persists nothing: the caller stores the artefact. That is what keeps the
    ML package free of any database concern.
    """
    import dropout_service.ml.training.train as train_module

    monkeypatch.setattr(
        train_module, "CANDIDATES", {"LogisticRegression": LogisticRegression(max_iter=1000)}
    )
    monkeypatch.setattr(
        train_module, "PARAM_GRIDS", {"LogisticRegression": {"classifier__C": [1.0]}}
    )

    result = train_module.run_training(raw_frame.sample(n=600, random_state=0))

    assert hasattr(result.pipeline, "predict_proba")
    assert "decision_threshold" in result.metadata
    assert result.metadata["n_features"] == len(result.metadata["features"])
    assert 0.5 < result.metadata["test_metrics"]["roc_auc"] <= 1.0
