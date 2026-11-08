"""Operational endpoints and deployed-model metadata."""

from fastapi import APIRouter, Request

from dropout_service.database import check_connection
from dropout_service.dependencies import CurrentUserDep, PredictorDep
from dropout_service.predictions.schemas import HealthResponse, ModelInfoResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Report liveness. Deliberately unauthenticated.

    The container's ``HEALTHCHECK`` calls this. It returns 200 even when the model is
    missing, reporting ``status: "degraded"`` instead: a missing model is a broken
    deployment that a restart cannot fix, so failing the health check would only produce a
    crash loop and hide the reason. Staying up lets an operator read this and see why.
    """
    engine = getattr(request.app.state, "engine", None)
    predictor = getattr(request.app.state, "predictor", None)
    return HealthResponse(
        status="healthy" if predictor is not None else "degraded",
        model_loaded=predictor is not None,
        database=check_connection(engine) if engine is not None else None,
    )


@router.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"service": "dropout-prediction-api", "docs": "/docs", "health": "/health"}


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info(predictor: PredictorDep, _user: CurrentUserDep) -> ModelInfoResponse:
    """Describe the model currently serving traffic. Requires authentication."""
    metadata = predictor.metadata
    return ModelInfoResponse(
        model_name=predictor.model_name,
        model_version=predictor.model_version,
        n_features=int(metadata.get("n_features", len(metadata.get("features", [])))),
        features=list(metadata.get("features", [])),
        decision_threshold=predictor.threshold,
        risk_bands=metadata.get(
            "risk_bands",
            {
                "bajo": [0.0, predictor.risk_medium],
                "medio": [predictor.risk_medium, predictor.risk_high],
                "alto": [predictor.risk_high, 1.0],
            },
        ),
        test_metrics=metadata.get("test_metrics", {}),
    )
