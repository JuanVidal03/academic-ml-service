"""Request and response shapes for authentication."""

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from dropout_service.auth.security import TOKEN_TYPE


class LoginRequest(BaseModel):
    """Credentials. ``extra="forbid"`` so a stray field is a 422, not a silent no-op."""

    model_config = ConfigDict(extra="forbid")

    email: EmailStr
    password: str = Field(min_length=1, max_length=256)


class UserRead(BaseModel):
    """A user as the API exposes them. Never includes the password hash."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    role: str
    is_active: bool


class TokenResponse(BaseModel):
    """What login and refresh return.

    The refresh token is deliberately absent: it travels only in an httpOnly cookie, so
    JavaScript cannot read it and therefore cannot leak it to an XSS payload.
    """

    access_token: str
    token_type: str = TOKEN_TYPE
    expires_in: int = Field(description="Access-token lifetime in seconds")
    user: UserRead
