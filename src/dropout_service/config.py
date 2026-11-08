"""Application settings and logging setup."""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Settings loaded from the environment.

    The tuneable values have **no code-side default**: they come from ``.env`` (or the
    process environment), and a missing one raises ``ValidationError`` naming the field.
    Since the app is built at import time, that means the service refuses to start rather
    than running with a value nobody chose -- so ``cp .env.example .env`` is a required
    setup step, not an optional one.

    Only ``log_level`` keeps a fallback; paths are derived from the project root. The
    database is required -- the model artefact, the training data and the audit log all
    live in it, so there is nothing to degrade to.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Database (required: the service stores its model, training data and audit
    # log here, so there is nothing to fall back to) ---
    database_url: str

    # --- Paths ---
    data_dir: Path = _PROJECT_ROOT / "data"
    reports_dir: Path = _PROJECT_ROOT / "reports"
    dataset_filename: str = "dataset_desercion_academica_3000.csv"

    # --- API ---
    # Required. Never set this to "*": the API is called with credentials, so a wildcard
    # origin would let any site call it. No code-side default is deliberate -- a stale
    # default here is a security problem, not a convenience.
    cors_origins: list[str]
    log_level: str = "INFO"

    # --- Training ---
    # All required. Every one of these changes the model that gets trained, so taking a
    # value nobody chose is worse than refusing to start.
    test_size: float
    cv_folds: int
    random_state: int
    target_recall: float

    # --- Risk banding (institutional policy, not a statistical result) ---
    risk_threshold_medium: float
    risk_threshold_high: float

    # --- Authentication (required) ---
    # No default, ever: a shipped signing key means anyone can mint a valid token for any
    # account. Generate one with `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
    jwt_secret: str = Field(min_length=32)
    access_token_minutes: int
    refresh_token_days: int
    #: Changing this is a code-level decision, not an operational knob.
    jwt_algorithm: str = "HS256"
    #: Defaults to the safe value. A browser silently REFUSES a Secure cookie over plain
    #: http://localhost, so local development must set COOKIE_SECURE=false -- otherwise
    #: login appears to work and every refresh fails.
    cookie_secure: bool = True
    cookie_samesite: Literal["lax", "strict", "none"] = "lax"

    @field_validator("database_url")
    @classmethod
    def _use_psycopg3_driver(cls, url: str) -> str:
        """Pin the bare ``postgresql://`` scheme to the driver actually installed.

        SQLAlchemy resolves ``postgresql://`` to psycopg2, which is not a dependency of
        this project (we ship psycopg 3), so such a URL fails with ModuleNotFoundError.
        Most hosted Postgres providers emit exactly that bare form, which would silently
        disable prediction logging in production. Normalising here means callers can paste a
        provider URL unchanged.
        """
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+psycopg://", 1)
        if url.startswith("postgres://"):  # some providers still emit this
            return url.replace("postgres://", "postgresql+psycopg://", 1)
        return url

    @property
    def csv_path(self) -> Path:
        return self.data_dir / self.dataset_filename


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()


def configure_logging(level: str | None = None) -> None:
    """Set up logging once, for scripts and the API alike."""
    logging.basicConfig(
        level=(level or get_settings().log_level).upper(),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
