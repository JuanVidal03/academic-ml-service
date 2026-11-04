"""Model selection, tuning, threshold calibration and persistence."""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from dropout_service.config import get_settings
from dropout_service.ml.pipeline import build_pipeline
from dropout_service.ml.schema import ALL_FEATURES
from dropout_service.ml.training.data import load_dataset, split_dataset

logger = logging.getLogger(__name__)

SCORING = ["roc_auc", "f1", "recall", "accuracy"]

CANDIDATES: dict[str, BaseEstimator] = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    # SVC has no native probabilities; CalibratedClassifierCV is the supported way to get
    # them (SVC(probability=True) is deprecated as of scikit-learn 1.9).
    "SVM": CalibratedClassifierCV(SVC(), ensemble=False),
}

PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "LogisticRegression": {
        "classifier__C": [0.05, 0.1, 1.0, 10.0],
    },
    "RandomForest": {
        "classifier__n_estimators": [200, 400],
        "classifier__max_depth": [10, 20, None],
        "classifier__min_samples_leaf": [1, 4],
    },
    "HistGradientBoosting": {
        "classifier__max_iter": [100, 200],
        "classifier__max_depth": [5, 10, None],
        "classifier__learning_rate": [0.05, 0.1],
    },
    "GradientBoosting": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [2, 3],
    },
    # Nested one level deeper: the SVC sits inside the calibration wrapper.
    "SVM": {
        "classifier__estimator__C": [1.0, 10.0],
        "classifier__estimator__gamma": ["scale", "auto"],
    },
}


def _seeded(estimator: BaseEstimator, random_state: int) -> BaseEstimator:
    """Clone ``estimator`` with the seed applied to every nested random_state.

    Walks nested parameters so wrapped estimators (the calibrated SVC, for instance) are
    seeded too, keeping training runs reproducible.
    """
    candidate = clone(estimator)
    seeds = {
        name: random_state
        for name in candidate.get_params(deep=True)
        if name == "random_state" or name.endswith("__random_state")
    }
    if seeds:
        candidate.set_params(**seeds)
    return candidate


def compare_models(X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> pd.DataFrame:
    """Score every candidate with cross-validation, best ROC-AUC first.

    Uses a single :func:`cross_validate` call per algorithm with all four scorers, rather
    than one :func:`cross_val_score` call per metric, which refit every model four times
    over.
    """
    settings = get_settings()
    rows = []

    for name, estimator in CANDIDATES.items():
        logger.info("Evaluating %s...", name)
        pipeline = build_pipeline(_seeded(estimator, settings.random_state))
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
        row = {"model": name}

        for metric in SCORING:
            row[f"{metric}_mean"] = float(scores[f"test_{metric}"].mean())
            row[f"{metric}_std"] = float(scores[f"test_{metric}"].std())
        rows.append(row)

    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False, ignore_index=True)


def tune_model(
    X: pd.DataFrame, y: pd.Series, model_name: str, cv: StratifiedKFold
) -> tuple[Pipeline, dict[str, Any]]:
    """Grid-search the named algorithm and return the best fitted pipeline."""
    settings = get_settings()
    pipeline = build_pipeline(_seeded(CANDIDATES[model_name], settings.random_state))
    grid = PARAM_GRIDS.get(model_name, {})
    if not grid:
        logger.info("No parameter grid for %s, fitting with defaults", model_name)
        return pipeline.fit(X, y), {}

    logger.info("Tuning %s over %d parameter(s)...", model_name, len(grid))
    search = GridSearchCV(pipeline, grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
    search.fit(X, y)
    logger.info("Best params: %s (CV ROC-AUC %.4f)", search.best_params_, search.best_score_)
    return search.best_estimator_, search.best_params_


def select_threshold(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    target_recall: float,
) -> float:
    """Choose the decision threshold from out-of-fold training predictions.

    Deliberately *not* chosen on the test set: that would tune the threshold to the very
    rows used to report performance, making the reported recall optimistic. Out-of-fold
    predictions on the training set give an unbiased basis, leaving the test set untouched
    for reporting.

    Returns the highest threshold still achieving ``target_recall`` (the highest is the most
    precise among those that qualify), or 0.5 if the target is unreachable.
    """
    proba = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[
        :, 1
    ]
    precision, recall, thresholds = precision_recall_curve(y_train, proba)
    # precision_recall_curve returns len(thresholds) == len(recall) - 1
    qualifying = [t for t, r in zip(thresholds, recall[:-1], strict=True) if r >= target_recall]
    if not qualifying:
        logger.warning(
            "No threshold reaches recall %.2f out of fold; falling back to 0.50", target_recall
        )
        return 0.5
    threshold = float(max(qualifying))
    achieved = max(r for t, r in zip(thresholds, recall[:-1], strict=True) if t == threshold)
    logger.info(
        "Decision threshold %.4f (out-of-fold recall %.4f, target %.2f)",
        threshold,
        achieved,
        target_recall,
    )
    return threshold


def evaluate(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float
) -> dict[str, Any]:
    """Score the fitted pipeline on the held-out test set at ``threshold``."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    return {
        # Threshold-independent, so comparable across threshold choices.
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "classification_report": classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        ),
    }


def save_model(
    pipeline: Pipeline,
    metadata: dict[str, Any],
    models_dir: Path,
) -> tuple[Path, Path]:
    """Persist the pipeline and its metadata side by side."""
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "model.joblib"
    metrics_path = models_dir / "metrics.json"
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metadata, indent=2))
    return model_path, metrics_path


def run_training(models_dir: Path | None = None, csv_path: Path | None = None) -> dict[str, Any]:
    """Run the full training flow and persist the result."""
    settings = get_settings()
    models_dir = models_dir or settings.models_dir
    cv = StratifiedKFold(
        n_splits=settings.cv_folds, shuffle=True, random_state=settings.random_state
    )

    X, y = load_dataset(csv_path)
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    comparison = compare_models(X_train, y_train, cv)
    logger.info("Model comparison:\n%s", comparison.to_string(index=False))

    best_name = str(comparison.loc[0, "model"])
    logger.info("Best algorithm: %s", best_name)

    pipeline, best_params = tune_model(X_train, y_train, best_name, cv)
    threshold = select_threshold(pipeline, X_train, y_train, cv, settings.target_recall)
    metrics = evaluate(pipeline, X_test, y_test, threshold)

    logger.info(
        "Test metrics: ROC-AUC %.4f | recall %.4f | precision %.4f | f1 %.4f | accuracy %.4f",
        metrics["roc_auc"],
        metrics["recall"],
        metrics["precision"],
        metrics["f1"],
        metrics["accuracy"],
    )

    metadata = {
        "model_name": best_name,
        "model_version": datetime.now(UTC).isoformat(),
        "features": ALL_FEATURES,
        "n_features": len(ALL_FEATURES),
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "best_params": {k: str(v) for k, v in best_params.items()},
        "decision_threshold": threshold,
        "target_recall": settings.target_recall,
        "risk_bands": {
            "bajo": [0.0, settings.risk_threshold_medium],
            "medio": [settings.risk_threshold_medium, settings.risk_threshold_high],
            "alto": [settings.risk_threshold_high, 1.0],
        },
        "test_metrics": metrics,
        "comparison": comparison.to_dict(orient="records"),
    }

    model_path, metrics_path = save_model(pipeline, metadata, models_dir)
    logger.info("Saved model to %s and metrics to %s", model_path, metrics_path)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "best_model": best_name,
        "decision_threshold": threshold,
        "test_metrics": metrics,
    }
