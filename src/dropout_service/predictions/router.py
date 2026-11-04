"""Prediction endpoints.

Mounted under ``/api/v1`` by ``dropout_service.router``.
"""

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Query

from dropout_service.dependencies import PredictorDep, SessionDep, SessionFactoryDep
from dropout_service.ml.predict import PredictionResult
from dropout_service.ml.schema import StudentFeatures
from dropout_service.predictions.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PaginatedPredictions,
    PredictionRecord,
)
from dropout_service.predictions.service import fetch_predictions_page, log_predictions

router = APIRouter(tags=["predictions"])


@router.post("/predict", response_model=PredictionResult)
async def predict(
    student: StudentFeatures,
    background: BackgroundTasks,
    predictor: PredictorDep,
    factory: SessionFactoryDep,
) -> PredictionResult:
    """Assess a single student's dropout risk."""
    result = predictor.predict([student])[0]
    background.add_task(log_predictions, factory, [student], [result], predictor.model_version)
    return result


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    payload: BatchPredictionRequest,
    background: BackgroundTasks,
    predictor: PredictorDep,
    factory: SessionFactoryDep,
) -> BatchPredictionResponse:
    """Assess up to 100 students in one call. Results keep the submitted order."""
    results = predictor.predict(payload.students)
    background.add_task(
        log_predictions, factory, payload.students, results, predictor.model_version
    )
    return BatchPredictionResponse(predictions=results, count=len(results))


@router.get("/predictions", response_model=PaginatedPredictions)
async def list_predictions(
    session: SessionDep,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> PaginatedPredictions:
    """Page through the prediction audit log, newest first."""
    rows, total = fetch_predictions_page(session, page, page_size)
    return PaginatedPredictions(
        items=[
            PredictionRecord(
                id=row.id,
                deserta=row.prediction,
                probabilidad=row.probability,
                riesgo=row.risk_level,
                model_version=row.model_version,
                created_at=row.created_at,
            )
            for row in rows
        ],
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size,
    )
