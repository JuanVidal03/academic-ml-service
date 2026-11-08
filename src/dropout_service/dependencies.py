"""Shared dependency providers.

Resources live on ``app.state``, populated once during the lifespan, and reach the routes
through ``Depends``. Tests swap them with ``app.dependency_overrides`` instead of reaching
into module-level globals.

The ``Annotated`` aliases at the bottom are what routes import, so the wiring is declared
once rather than repeated in every handler signature.
"""

from collections.abc import Iterator
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session, sessionmaker

from dropout_service.config import Settings
from dropout_service.ml.predict import Predictor
from dropout_service.users.models import User


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
    """Return the session factory.

    Still typed optional because startup tolerates an unreachable database: the attribute
    is None only when the connection could not be established, which /health reports.
    """
    return getattr(request.app.state, "session_factory", None)


def require_session(request: Request) -> Iterator[Session]:
    """Yield a session for endpoints that cannot work without a database."""
    factory = getattr(request.app.state, "session_factory", None)
    if factory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database unavailable.",
        )
    with factory() as session:
        yield session


#: ``auto_error=False`` so a missing header reaches our own handler and returns a
#: WWW-Authenticate challenge with a consistent body, rather than FastAPI's 403.
_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> User:
    """Resolve the signed-in user from the Authorization header.

    Imported lazily inside the function body to keep the module-level import graph free of
    the auth slice, which imports this module for its own dependencies.
    """
    from dropout_service.auth.security import decode_access_token
    from dropout_service.users.service import get_by_id

    unauthorised = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No autenticado.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None or not credentials.credentials:
        raise unauthorised

    try:
        payload = decode_access_token(credentials.credentials)
    except jwt.InvalidTokenError:
        # Expired, tampered, or signed with the wrong key -- all indistinguishable to the
        # caller on purpose.
        raise unauthorised from None

    factory = getattr(request.app.state, "session_factory", None)
    if factory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable."
        )

    with factory() as session:
        user = get_by_id(session, int(payload["sub"]))
        if user is None or not user.is_active:
            raise unauthorised
        session.expunge(user)
    return user


def require_admin(user: Annotated[User, Depends(get_current_user)]) -> User:
    """Restrict a route to administrators.

    403 rather than 401: the caller proved who they are, they simply are not allowed. A 401
    would tell a client to re-authenticate, which would not help.
    """
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Requiere permisos de administrador."
        )
    return user


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
CurrentUserDep = Annotated[User, Depends(get_current_user)]
AdminUserDep = Annotated[User, Depends(require_admin)]
PredictorDep = Annotated[Predictor, Depends(get_predictor)]
SessionFactoryDep = Annotated[sessionmaker | None, Depends(get_session_factory)]
SessionDep = Annotated[Session, Depends(require_session)]
