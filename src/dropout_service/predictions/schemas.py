"""Request and response wrappers for the predictions slice.

The student input contract lives in :mod:`dropout_service.ml.schema` and the prediction
result in :mod:`dropout_service.ml.predict`; both are re-exported here so the HTTP layer
stays a thin wrapper and no field list is ever written twice.

``HealthResponse`` and ``ModelInfoResponse`` are used by the ``system`` slice and kept here
because they describe the same prediction contract (threshold, risk bands).
"""

from datetime import datetime

from pydantic import BaseModel, Field

from dropout_service.ml.predict import PredictionResult, RiskLevel
from dropout_service.ml.schema import StudentFeatures

__all__ = [
    "BatchPredictionRequest",
    "PredictionRequest",
    "BatchPredictionResponse",
    "HealthResponse",
    "ModelInfoResponse",
    "PaginatedPredictions",
    "PredictionRecord",
    "PredictionResult",
    "StudentFeatures",
]

MAX_BATCH_SIZE = 100


class PredictionRequest(StudentFeatures):
    """A single prediction request: the features, plus an optional student code.

    ``codigo`` is **not** a model feature -- it never reaches ``StudentFeatures``'
    derived feature lists, so it cannot influence the prediction. It only lets the audit
    row be linked to a student, which is what makes risk history answerable.
    """

    codigo: str | None = Field(
        None, max_length=32, description="Institutional student code; links the audit row"
    )


class BatchPredictionRequest(BaseModel):
    """Several students in one call."""

    students: list[PredictionRequest] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class BatchPredictionResponse(BaseModel):
    """Results in the same order as the submitted students."""

    predictions: list[PredictionResult]
    count: int


class ModelInfoResponse(BaseModel):
    """Metadata about the model currently serving traffic."""

    model_name: str
    model_version: str
    n_features: int
    features: list[str]
    decision_threshold: float
    risk_bands: dict[str, list[float]]
    test_metrics: dict


class HealthResponse(BaseModel):
    """Liveness and readiness detail."""

    status: str = Field(..., description="'healthy' when a model is loaded, else 'degraded'")
    model_loaded: bool
    database: dict | None = Field(None, description="Null when no database is configured")


class PredictionRecord(BaseModel):
    """A logged prediction."""

    id: int
    deserta: bool
    probabilidad: float
    riesgo: RiskLevel
    model_version_id: int | None
    student_id: int | None
    created_at: datetime


class PaginatedPredictions(BaseModel):
    """A page of logged predictions, newest first."""

    items: list[PredictionRecord]
    total: int
    page: int
    page_size: int
    pages: int
