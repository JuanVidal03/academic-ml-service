"""Aggregate queries for the dashboard.

Counts and shares only -- never individual predictions. That is what lets an adviser see
the dashboard while `/predictions` itself stays admin-only.
"""

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session

from dropout_service.model_registry.models import ModelVersion
from dropout_service.predictions.models import Prediction
from dropout_service.stats.schemas import (
    DailyCount,
    FeatureImportance,
    RiskBandCount,
    StatsOverview,
)
from dropout_service.training_data.models import AnalysisReport, TrainingRecord

logger = logging.getLogger(__name__)

TREND_DAYS = 30
TOP_FEATURES = 5


def build_overview(session: Session) -> StatsOverview:
    """Collect every dashboard figure in one pass."""
    total = session.scalar(select(func.count()).select_from(Prediction)) or 0

    by_risk = [
        RiskBandCount(riesgo=riesgo, count=count)
        for riesgo, count in session.execute(
            select(Prediction.risk_level, func.count())
            .group_by(Prediction.risk_level)
            .order_by(func.count().desc())
        ).all()
    ]

    flagged = (
        session.scalar(
            select(func.count()).select_from(Prediction).where(Prediction.prediction.is_(True))
        )
        or 0
    )

    since = datetime.now(UTC) - timedelta(days=TREND_DAYS)
    day = func.date(Prediction.created_at)
    per_day = [
        DailyCount(date=str(date), count=count)
        for date, count in session.execute(
            select(day, func.count())
            .where(Prediction.created_at >= since)
            .group_by(day)
            .order_by(day)
        ).all()
    ]

    training_total = session.scalar(select(func.count()).select_from(TrainingRecord)) or 0
    # Postgres refuses to cast boolean straight to a numeric type, so map it through a
    # CASE first. Casting directly raises CannotCoerce at query time.
    dropout_rate = session.scalar(
        select(func.avg(case((TrainingRecord.deserto.is_(True), 1.0), else_=0.0)))
    )

    report = session.scalars(
        select(AnalysisReport).order_by(AnalysisReport.id.desc()).limit(1)
    ).first()
    top_features = _top_features(report)

    version = session.scalars(
        select(ModelVersion).where(ModelVersion.is_active.is_(True)).limit(1)
    ).first()

    return StatsOverview(
        total_predictions=total,
        by_risk=by_risk,
        flagged_share=(flagged / total) if total else 0.0,
        per_day=per_day,
        training_records=training_total,
        training_dropout_rate=float(dropout_rate or 0.0),
        top_features=top_features,
        model_name=version.algorithm if version else None,
        model_roc_auc=_roc_auc(version),
    )


def _top_features(report: AnalysisReport | None) -> list[FeatureImportance]:
    """The strongest features from the newest analysis run, if one exists."""
    if report is None:
        return []
    importances = (report.report or {}).get("permutation_importance", {})
    ranked = sorted(importances.items(), key=lambda kv: kv[1].get("importance", 0.0), reverse=True)
    return [
        FeatureImportance(feature=name, importance=float(vals.get("importance", 0.0)))
        for name, vals in ranked[:TOP_FEATURES]
    ]


def _roc_auc(version: ModelVersion | None) -> float | None:
    if version is None:
        return None
    value = (version.metrics or {}).get("test_metrics", {}).get("roc_auc")
    return float(value) if value is not None else None
