"""Prediction endpoints."""

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status

from dropout_service.dependencies import (
    AdminUserDep,
    CandidatesDep,
    CurrentUserDep,
    PredictorDep,
    SessionDep,
    SessionFactoryDep,
)
from dropout_service.ml.predict import PredictionResult
from dropout_service.predictions.schemas import (
    BatchComparePredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ComparePredictionResponse,
    ModelPrediction,
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


@router.post("/predict/compare", response_model=ComparePredictionResponse)
async def predict_compare(
    student: PredictionRequest,
    background: BackgroundTasks,
    predictor: PredictorDep,
    candidates: CandidatesDep,
    factory: SessionFactoryDep,
    user: CurrentUserDep,
) -> ComparePredictionResponse:
    """Assess a student with every stored algorithm, best first.

    Each candidate applies its own calibrated threshold, so the verdicts are comparable as
    judgements even though the raw probabilities are not directly comparable across
    algorithms.

    Only the deployed model's answer is written to the audit log. Five rows per request
    would bury the one that was actually acted on.
    """
    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No hay modelos comparables almacenados. Reentrena con: uv run scripts/train.py"
            ),
        )

    models: list[ModelPrediction] = []
    for candidate in candidates:
        result = candidate.predictor.predict([student])[0]
        models.append(
            ModelPrediction(
                algorithm=candidate.algorithm,
                is_best=candidate.is_best,
                decision_threshold=candidate.predictor.threshold,
                cv_metrics=candidate.cv_metrics,
                test_metrics=candidate.test_metrics,
                **result.model_dump(),
            )
        )

    best = next((m for m in models if m.is_best), models[0])
    deployed = predictor.predict([student])[0]
    background.add_task(
        log_predictions, factory, [student], [deployed], predictor.model_version_id, user.id
    )
    return ComparePredictionResponse(best_algorithm=best.algorithm, models=models)


@router.post("/predict/batch/compare", response_model=BatchComparePredictionResponse)
async def predict_batch_compare(
    payload: BatchPredictionRequest,
    background: BackgroundTasks,
    predictor: PredictorDep,
    candidates: CandidatesDep,
    factory: SessionFactoryDep,
    user: CurrentUserDep,
) -> BatchComparePredictionResponse:
    """Assess up to 100 students with every stored algorithm, best first per student.

    Each candidate predicts the whole batch in one vectorised call, so this costs one
    prediction pass per algorithm rather than one per student -- the same five passes a
    single ``/predict/compare`` makes.

    As with ``/predict/compare``, only the deployed model's answer is written to the audit
    log: one row per student, not one per algorithm.
    """
    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No hay modelos comparables almacenados. Reentrena con: uv run scripts/train.py"
            ),
        )

    students = payload.students
    # algorithm -> that algorithm's result for every student, in submitted order
    by_candidate = [(candidate, candidate.predictor.predict(students)) for candidate in candidates]

    comparisons: list[ComparePredictionResponse] = []
    for index in range(len(students)):
        models = [
            ModelPrediction(
                algorithm=candidate.algorithm,
                is_best=candidate.is_best,
                decision_threshold=candidate.predictor.threshold,
                cv_metrics=candidate.cv_metrics,
                test_metrics=candidate.test_metrics,
                **results[index].model_dump(),
            )
            for candidate, results in by_candidate
        ]
        best = next((m for m in models if m.is_best), models[0])
        comparisons.append(ComparePredictionResponse(best_algorithm=best.algorithm, models=models))

    deployed = predictor.predict(students)
    background.add_task(
        log_predictions, factory, students, deployed, predictor.model_version_id, user.id
    )
    return BatchComparePredictionResponse(comparisons=comparisons, count=len(comparisons))


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
