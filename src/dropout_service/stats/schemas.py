"""Response shapes for the dashboard aggregates."""

from pydantic import BaseModel, Field


class RiskBandCount(BaseModel):
    """How many predictions fell in one risk band."""

    riesgo: str
    count: int


class DailyCount(BaseModel):
    """Predictions made on one day."""

    date: str
    count: int


class FeatureImportance(BaseModel):
    """One feature's contribution, from the newest analysis run."""

    feature: str
    importance: float


class StatsOverview(BaseModel):
    """Everything the dashboard needs, in one request.

    One endpoint rather than five, because the dashboard renders them together and five
    round-trips would show the cards filling in at different moments.
    """

    total_predictions: int
    by_risk: list[RiskBandCount]
    flagged_share: float = Field(description="Fraction of predictions where deserta is true")
    per_day: list[DailyCount] = Field(description="Last 30 days, oldest first")
    training_records: int
    training_dropout_rate: float
    top_features: list[FeatureImportance]
    model_name: str | None = None
    model_roc_auc: float | None = None
