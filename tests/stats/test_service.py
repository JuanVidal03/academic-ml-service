"""The dashboard aggregates.

Covers the logic, not the SQL. Faking a session well enough to stand in for several
different aggregate queries would mostly test the fake; the queries themselves are
cross-checked against the real database in docs/WALKTHROUGH.md.
"""

from dropout_service.model_registry.models import ModelVersion
from dropout_service.stats.service import TOP_FEATURES, _roc_auc, _top_features
from dropout_service.training_data.models import AnalysisReport


def make_report(importances: dict[str, float]) -> AnalysisReport:
    return AnalysisReport(
        id=1,
        n_records=3000,
        dropout_rate=0.439,
        report={
            "permutation_importance": {
                name: {"importance": value, "std": 0.001} for name, value in importances.items()
            }
        },
    )


def test_top_features_are_ranked_strongest_first():
    report = make_report({"a": 0.01, "b": 0.09, "c": 0.05})
    ranked = _top_features(report)

    assert [f.feature for f in ranked] == ["b", "c", "a"]
    assert ranked[0].importance == 0.09


def test_top_features_is_capped():
    report = make_report({f"f{i}": i / 100 for i in range(20)})
    assert len(_top_features(report)) == TOP_FEATURES


def test_top_features_is_empty_without_an_analysis_run():
    """A database that has never been analysed must not break the dashboard."""
    assert _top_features(None) == []


def test_top_features_survives_a_report_without_importances():
    report = AnalysisReport(id=1, n_records=0, dropout_rate=0.0, report={})
    assert _top_features(report) == []


def test_roc_auc_is_read_from_the_active_version():
    version = ModelVersion(
        id=1,
        version="v",
        algorithm="LogisticRegression",
        artifact=b"x",
        metrics={"test_metrics": {"roc_auc": 0.7695}},
        decision_threshold=0.35,
        n_features=34,
        is_active=True,
    )
    assert _roc_auc(version) == 0.7695


def test_roc_auc_is_none_when_there_is_no_model():
    assert _roc_auc(None) is None


def test_roc_auc_is_none_when_metrics_lack_it():
    """An older version whose metadata predates the field must not raise."""
    version = ModelVersion(
        id=1,
        version="v",
        algorithm="X",
        artifact=b"x",
        metrics={},
        decision_threshold=0.5,
        n_features=34,
        is_active=True,
    )
    assert _roc_auc(version) is None
