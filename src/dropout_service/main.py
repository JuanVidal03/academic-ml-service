"""FastAPI application factory and ASGI entry point.

Built as a factory rather than a module-level singleton so tests can construct an app with
their own :class:`Settings` and dependency overrides. Resources are created once in the
lifespan and attached to ``app.state``.

Run with: ``uvicorn dropout_service.main:app``
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from dropout_service import __version__
from dropout_service.config import Settings, configure_logging, get_settings
from dropout_service.database import create_db_engine, create_session_factory
from dropout_service.ml.predict import Predictor
from dropout_service.router import API_V1_PREFIX, api_router
from dropout_service.system.router import ops_router

logger = logging.getLogger(__name__)

DESCRIPTION = """
Predicts academic dropout risk for university students.

The model is trained offline from a historical dataset and loaded at startup. A PostgreSQL
database is optional: without one, predictions still work but are not logged.

Business endpoints live under `/api/v1`. `/health` is unprefixed, as an operational contract.

**Note:** these endpoints are currently unauthenticated. See `docs/WALKTHROUGH.md`.
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Pass ``settings`` to override environment configuration."""
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        app.state.settings = settings
        app.state.engine = None
        app.state.session_factory = None
        app.state.predictor = None

        try:
            app.state.predictor = Predictor.from_disk(settings.model_path, settings.metrics_path)
        except FileNotFoundError as exc:
            logger.error("Starting without a model: %s", exc)

        if settings.database_url:
            try:
                engine = create_db_engine(settings.database_url)
                app.state.engine = engine
                app.state.session_factory = create_session_factory(engine)
                logger.info("Database configured")
            except Exception:
                logger.exception("Database setup failed; continuing without logging")
        else:
            logger.info("No DATABASE_URL set; predictions will not be logged")

        yield

        if app.state.engine is not None:
            app.state.engine.dispose()

    app = FastAPI(
        title="Dropout Prediction API",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        openapi_url=f"{API_V1_PREFIX}/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(SQLAlchemyError)
    async def _handle_database_error(request: Request, exc: SQLAlchemyError) -> JSONResponse:
        """Turn an unreachable or misconfigured database into a clear 503.

        Without this, a database that is down surfaces as an opaque 500 and the caller
        cannot tell a service fault from a bad request. The real error is logged
        server-side; the response stays generic so connection details never reach a
        client.
        """
        logger.exception("Database error serving %s", request.url.path)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Database unavailable."},
        )

    # Ops endpoints at the root; everything else behind the version prefix.
    app.include_router(ops_router)
    app.include_router(api_router)
    return app


#: ASGI entry point: ``uvicorn dropout_service.main:app``
app = create_app()
