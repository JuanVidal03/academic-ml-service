"""Exploratory analysis of the dropout dataset.

Produces a Markdown and a JSON report describing which features carry signal. This is a
reporting tool, not part of the serving path.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from dropout_service.ml.schema import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORICAL_FILL,
    NUMERIC_FEATURES,
    build_pipeline,
)
from dropout_service.ml.training.data import prepare_dataset, split_dataset

logger = logging.getLogger(__name__)

TOP_N = 12
REPORT_CATEGORIES = [
    "programa",
    "genero",
    "nivel_educativo_padres",
    "dificultad_economica",
    "beca_perdida_en_periodo",
    "area_materia_critica",
]


def describe_dataset(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    return {
        "total_records": int(len(X)),
        "dropout_count": int(y.sum()),
        "retained_count": int(len(y) - y.sum()),
        "dropout_rate": float(y.mean()),
        "n_features": len(ALL_FEATURES),
        "n_categorical": len(CATEGORICAL_FEATURES),
        "n_numeric": len(NUMERIC_FEATURES),
    }


def summarise_features(X: pd.DataFrame) -> dict[str, Any]:
    """Per-feature summary statistics."""
    summary: dict[str, Any] = {}
    for col in NUMERIC_FEATURES:
        series = pd.to_numeric(X[col], errors="coerce")
        summary[col] = {
            "type": "numeric",
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "median": float(series.median()),
            "max": float(series.max()),
            "missing": int(series.isna().sum()),
        }
    for col in CATEGORICAL_FEATURES:
        counts = X[col].value_counts(dropna=True)
        summary[col] = {
            "type": "categorical",
            "unique_values": int(counts.size),
            "value_counts": {str(k): int(v) for k, v in counts.items()},
            "missing": int(X[col].isna().sum()),
        }
    return summary


def dropout_rate_by_category(X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    """Dropout rate within each level of each categorical feature."""
    frame = X.copy()
    frame["_target"] = y.to_numpy()
    rates: dict[str, Any] = {}
    for col in CATEGORICAL_FEATURES:
        grouped = (
            frame.fillna({col: CATEGORICAL_FILL}).groupby(col)["_target"].agg(["mean", "count"])
        )
        rates[col] = {
            str(level): {"dropout_rate": float(row["mean"]), "count": int(row["count"])}
            for level, row in grouped.iterrows()
        }
    return rates


def numeric_correlations(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Pearson correlation of each numeric feature with the target, strongest first."""
    correlations = {
        col: float(pd.to_numeric(X[col], errors="coerce").corr(y.astype(float)) or 0.0)
        for col in NUMERIC_FEATURES
    }
    return dict(sorted(correlations.items(), key=lambda kv: abs(kv[1]), reverse=True))


def mutual_information(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> dict[str, float]:
    """Mutual information between each feature and the target, strongest first.

    Categorical features are ordinal-coded and flagged through ``discrete_features``. The
    coding is safe *because* of that flag: mutual information depends only on the partition
    a discrete variable induces, so relabelling its levels cannot change the result. Omitting
    the flag is what breaks the estimate -- sklearn would then apply its continuous
    nearest-neighbour estimator to arbitrary integer codes and read meaning into their
    numeric spacing.
    """
    encoded = X.copy()
    for col in CATEGORICAL_FEATURES:
        encoded[col] = encoded[col].fillna(CATEGORICAL_FILL).astype("category").cat.codes

    for col in NUMERIC_FEATURES:
        series = pd.to_numeric(encoded[col], errors="coerce")
        encoded[col] = series.fillna(series.median())

    discrete_mask = np.array([col in CATEGORICAL_FEATURES for col in ALL_FEATURES])
    scores = mutual_info_classif(
        encoded[ALL_FEATURES], y, discrete_features=discrete_mask, random_state=random_state
    )
    ranked = dict(zip(ALL_FEATURES, (float(s) for s in scores), strict=True))

    return dict(sorted(ranked.items(), key=lambda kv: kv[1], reverse=True))


def feature_importance(
    X: pd.DataFrame, y: pd.Series, random_state: int = 42, n_repeats: int = 10
) -> dict[str, dict[str, float]]:
    """Permutation importance of each feature, measured on held-out data.

    Replaces Random Forest ``feature_importances_``, which is computed from training-set
    impurity decrease and is biased toward high-cardinality features. Permutation importance
    instead measures how much held-out ROC-AUC degrades when a feature is shuffled, which is
    what "this feature matters" should mean. It runs against the full pipeline, so one score
    corresponds to one original feature rather than to a one-hot column.
    """
    X_train, X_test, y_train, y_test = split_dataset(X, y, random_state=random_state)
    pipeline = build_pipeline(LogisticRegression(max_iter=1000, random_state=random_state))
    pipeline.fit(X_train, y_train)

    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    ranked = {
        col: {"importance": float(mean), "std": float(std)}
        for col, mean, std in zip(
            X_test.columns, result.importances_mean, result.importances_std, strict=True
        )
    }
    return dict(sorted(ranked.items(), key=lambda kv: kv[1]["importance"], reverse=True))


def run_analysis(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute the full analysis for a training frame."""
    X, y = prepare_dataset(frame)
    logger.info("Analysing %d records...", len(X))
    return {
        "dataset": describe_dataset(X, y),
        "feature_summary": summarise_features(X),
        "dropout_rate_by_category": dropout_rate_by_category(X, y),
        "numeric_correlations": numeric_correlations(X, y),
        "mutual_information": mutual_information(X, y),
        "permutation_importance": feature_importance(X, y),
    }


def _table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return [*lines, ""]


def render_markdown(analysis: dict[str, Any]) -> str:
    """Render the analysis as a Markdown report."""
    info = analysis["dataset"]
    lines = [
        "# Academic Dropout - Data Analysis",
        "",
        "## Dataset",
        "",
        f"- **Records**: {info['total_records']:,}",
        f"- **Dropout**: {info['dropout_count']:,} ({info['dropout_rate']:.1%})",
        f"- **Retained**: {info['retained_count']:,} ({1 - info['dropout_rate']:.1%})",
        f"- **Features**: {info['n_features']} "
        f"({info['n_categorical']} categorical, {info['n_numeric']} numeric)",
        "",
        "The classes are close to balanced, so no resampling or class weighting is applied.",
        "",
        "## Permutation importance (held-out ROC-AUC drop)",
        "",
        "How much test-set ROC-AUC falls when each feature is shuffled. Measured on held-out",
        "data, so it reflects genuine predictive contribution.",
        "",
    ]
    lines += _table(
        ("Feature", "Importance", "Std"),
        [
            (name, f"{vals['importance']:.4f}", f"{vals['std']:.4f}")
            for name, vals in list(analysis["permutation_importance"].items())[:TOP_N]
        ],
    )

    lines += ["## Mutual information", ""]
    lines += _table(
        ("Feature", "MI"),
        [(k, f"{v:.4f}") for k, v in list(analysis["mutual_information"].items())[:TOP_N]],
    )

    lines += ["## Correlation with dropout (numeric features)", ""]
    lines += _table(
        ("Feature", "Pearson r"),
        [(k, f"{v:+.4f}") for k, v in list(analysis["numeric_correlations"].items())[:TOP_N]],
    )

    lines += ["## Dropout rate by category", ""]
    for category in REPORT_CATEGORIES:
        levels = analysis["dropout_rate_by_category"].get(category)
        if not levels:
            continue
        lines += [f"### {category}", ""]
        ordered = sorted(levels.items(), key=lambda kv: kv[1]["dropout_rate"], reverse=True)
        lines += _table(
            ("Value", "Dropout rate", "Students"),
            [(name, f"{v['dropout_rate']:.1%}", f"{v['count']:,}") for name, v in ordered],
        )

    return "\n".join(lines)


def save_analysis(analysis: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    """Write the JSON and Markdown reports."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "analysis.json"
    md_path = reports_dir / "analysis.md"
    json_path.write_text(json.dumps(analysis, indent=2))
    md_path.write_text(render_markdown(analysis))
    return json_path, md_path
