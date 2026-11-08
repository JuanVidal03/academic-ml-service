"""Prediction endpoints."""

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Query

from dropout_service.dependencies import (
    AdminUserDep,
    CurrentUserDep,
    PredictorDep,
    SessionDep,
    SessionFactoryDep,
)
from dropout_service.ml.predict import PredictionResult
from dropout_service.predictions.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PaginatedPredictions,
    PredictionRecord,
    PredictionRequest,
)
from dropout_service.predictions.service import fetch_predictions_page, log_predictions

router = APIRouter(tags=["predictions"])


@router.post("/predict", response_model=PredictionResult)
async def predict(
    student: PredictionRequest,
    background: BackgroundTasks,
    predictor: PredictorDep,
    factory: SessionFactoryDep,
    user: CurrentUserDep,
) -> PredictionResult:
    """Assess a single student's dropout risk."""
    result = predictor.predict([student])[0]
    background.add_task(
        log_predictions, factory, [student], [result], predictor.model_version_id, user.id
    )
    return result


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    payload: BatchPredictionRequest,
    background: BackgroundTasks,
    predictor: PredictorDep,
    factory: SessionFactoryDep,
    user: CurrentUserDep,
) -> BatchPredictionResponse:
    """Assess up to 100 students in one call. Results keep the submitted order."""
    results = predictor.predict(payload.students)
    background.add_task(
        log_predictions, factory, payload.students, results, predictor.model_version_id, user.id
    )
    return BatchPredictionResponse(predictions=results, count=len(results))


@router.get("/predictions", response_model=PaginatedPredictions)
async def list_predictions(
    session: SessionDep,
    _admin: AdminUserDep,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> PaginatedPredictions:
    """Page through the prediction audit log, newest first.

    Administrators only: this returns every student's features and risk score, so an
    adviser working with a handful of students has no business reading all of it.
    """
    rows, total = fetch_predictions_page(session, page, page_size)
    return PaginatedPredictions(
        items=[
            PredictionRecord(
                id=row.id,
                deserta=row.prediction,
                probabilidad=row.probability,
                riesgo=row.risk_level,
                model_version_id=row.model_version_id,
                student_id=row.student_id,
                created_at=row.created_at,
            )
            for row in rows
        ],
        total=total,
        page=page,
        page_size=page_size,
        pages=(total + page_size - 1) // page_size,
    )
