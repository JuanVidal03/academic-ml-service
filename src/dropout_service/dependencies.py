"""Shared dependency providers.

Resources live on ``app.state``, populated once during the lifespan, and reach the routes
through ``Depends``. Tests swap them with ``app.dependency_overrides`` instead of reaching
into module-level globals.

The ``Annotated`` aliases at the bottom are what routes import, so the wiring is declared
once rather than repeated in every handler signature.
"""

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.orm import Session, sessionmaker

from dropout_service.config import Settings
from dropout_service.ml.predict import Predictor


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_predictor(request: Request) -> Predictor:
    """Return the loaded predictor, or fail the request if training never ran."""
    predictor: Predictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Train one with: uv run scripts/train.py",
        )
    return predictor


def get_session_factory(request: Request) -> sessionmaker | None:
    """Return the session factory, or None when the service runs without a database."""
    return getattr(request.app.state, "session_factory", None)


def require_session(request: Request) -> Iterator[Session]:
    """Yield a session for endpoints that cannot work without a database."""
    factory = getattr(request.app.state, "session_factory", None)
    if factory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="This endpoint requires a database. Set DATABASE_URL.",
        )
    with factory() as session:
        yield session


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
PredictorDep = Annotated[Predictor, Depends(get_predictor)]
SessionFactoryDep = Annotated[sessionmaker | None, Depends(get_session_factory)]
SessionDep = Annotated[Session, Depends(require_session)]

# When auth arrives, the protected-route dependency belongs here:
#     CurrentUserDep = Annotated[User, Depends(get_current_user)]
