"""Application settings and configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# project root:
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str | None = None

    # Paths
    data_dir: Path = _PROJECT_ROOT / "data"
    models_dir: Path = _PROJECT_ROOT / "models"
    reports_dir: Path = _PROJECT_ROOT / "reports"

    # Model
    model_path: Path | None = None

    # API
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    @property
    def csv_path(self) -> Path:
        return self.data_dir / "dataset_desercion_academica_3000.csv"

    @property
    def effective_model_path(self) -> Path:
        return self.model_path or (self.models_dir / "model.joblib")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
