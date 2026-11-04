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
from dropout_service.model_registry.service import load_active
from dropout_service.predictions.router import router as predictions_router
from dropout_service.system.router import router as system_router

logger = logging.getLogger(__name__)

DESCRIPTION = """
Predicts academic dropout risk for university students.

The model is trained offline and stored in PostgreSQL; the service loads the active version
at startup. A database is required.

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

        # Startup stays tolerant of an unreachable database on purpose. The database is
        # required, but crash-looping through a brief outage would hide the reason: /health
        # reports it instead and the orchestrator decides what to do.
        try:
            engine = create_db_engine(settings.database_url)
            app.state.engine = engine
            app.state.session_factory = create_session_factory(engine)
            with app.state.session_factory() as session:
                app.state.predictor = load_active(session)
        except Exception as exc:  # noqa: BLE001 - any failure here means degraded, not dead
            logger.error(
                "Could not load the active model, starting degraded: %s: %s",
                type(exc).__name__,
                exc,
            )
            logger.debug("Model load failure detail", exc_info=True)

        yield

        if app.state.engine is not None:
            app.state.engine.dispose()

    app = FastAPI(
        title="Dropout Prediction API",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
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

    app.include_router(system_router)
    app.include_router(predictions_router)
    # When auth arrives, register its router here. A blanket requirement goes on the
    # include_router call for the protected routers -- not on system_router, which carries
    # /health and must stay reachable for the container's HEALTHCHECK.
    return app


#: ASGI entry point: ``uvicorn dropout_service.main:app``
app = create_app()
