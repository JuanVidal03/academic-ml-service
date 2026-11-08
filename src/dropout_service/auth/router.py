"""Authentication endpoints."""

import logging
from typing import Annotated

from fastapi import APIRouter, Cookie, HTTPException, Response, status

from dropout_service.auth.schemas import LoginRequest, TokenResponse, UserRead
from dropout_service.auth.security import create_access_token
from dropout_service.auth.service import (
    authenticate,
    find_refresh_token,
    issue_refresh_token,
    revoke_family,
    revoke_one,
)
from dropout_service.dependencies import CurrentUserDep, SessionDep, SettingsDep
from dropout_service.users.service import get_by_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

#: The cookie name is a constant, not a setting: FastAPI resolves the Cookie alias at
#: import time, so a configurable name could not reach the parameter declaration and the
#: two would silently disagree.
REFRESH_COOKIE = "dropout_refresh"

#: The cookie is scoped to /auth, so it is never sent to /predict. Narrower exposure, and
#: nothing outside these endpoints needs it.
REFRESH_COOKIE_PATH = "/auth"

#: One message for every credential failure. Saying "unknown email" or "wrong password"
#: would let anyone enumerate which addresses have accounts.
INVALID_CREDENTIALS = "Correo o contraseña incorrectos."

RefreshCookie = Annotated[str | None, Cookie(alias=REFRESH_COOKIE)]


def _set_refresh_cookie(response: Response, token: str, settings) -> None:
    """Attach the refresh cookie.

    ``httponly`` is the point of the whole design: JavaScript cannot read this value, so an
    XSS payload cannot steal a long-lived credential. ``secure`` defaults to True, which
    means a browser will silently drop the cookie over plain http -- local development must
    set COOKIE_SECURE=false.
    """
    response.set_cookie(
        key=REFRESH_COOKIE,
        value=token,
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        max_age=settings.refresh_token_days * 24 * 60 * 60,
        path=REFRESH_COOKIE_PATH,
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    credentials: LoginRequest, response: Response, session: SessionDep, settings: SettingsDep
) -> TokenResponse:
    """Exchange credentials for an access token and a refresh cookie."""
    user = authenticate(session, str(credentials.email), credentials.password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=INVALID_CREDENTIALS)

    raw_refresh = issue_refresh_token(session, user)
    _set_refresh_cookie(response, raw_refresh, settings)
    access_token, expires_in = create_access_token(user.id, user.role, user.email)
    logger.info("User %s signed in", user.id)
    return TokenResponse(
        access_token=access_token, expires_in=expires_in, user=UserRead.model_validate(user)
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    response: Response,
    session: SessionDep,
    settings: SettingsDep,
    dropout_refresh: RefreshCookie = None,
) -> TokenResponse:
    """Rotate the refresh token and issue a new access token.

    Every refresh invalidates the token it was called with. Presenting an already-revoked
    token therefore means the value leaked, so the whole rotation chain is killed and the
    real user has to sign in again -- unpleasant, but better than leaving a thief with a
    working credential.
    """
    if not dropout_refresh:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No hay sesión.")

    stored = find_refresh_token(session, dropout_refresh)
    if stored is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sesión inválida.")

    if not stored.is_usable:
        revoke_family(session, stored)
        response.delete_cookie(REFRESH_COOKIE, path=REFRESH_COOKIE_PATH)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sesión expirada.")

    user = get_by_id(session, stored.user_id)
    if user is None or not user.is_active:
        revoke_family(session, stored)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sesión inválida.")

    raw_refresh = issue_refresh_token(session, user, replaces=stored)
    _set_refresh_cookie(response, raw_refresh, settings)
    access_token, expires_in = create_access_token(user.id, user.role, user.email)
    return TokenResponse(
        access_token=access_token, expires_in=expires_in, user=UserRead.model_validate(user)
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    response: Response,
    session: SessionDep,
    settings: SettingsDep,
    dropout_refresh: RefreshCookie = None,
) -> None:
    """Revoke the refresh token and clear the cookie.

    Deliberately unauthenticated and always 204: signing out must work even with an expired
    access token, and telling a caller whether a token existed reveals nothing useful.
    """
    if dropout_refresh:
        stored = find_refresh_token(session, dropout_refresh)
        if stored is not None:
            revoke_one(session, stored)
    response.delete_cookie(REFRESH_COOKIE, path=REFRESH_COOKIE_PATH)


@router.get("/me", response_model=UserRead)
async def me(user: CurrentUserDep) -> UserRead:
    """The signed-in user."""
    return UserRead.model_validate(user)
