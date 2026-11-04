"""Settings behaviour."""

import pytest
from pydantic import ValidationError

from dropout_service.config import Settings

#: Every field with no code-side default. A missing one must abort startup.
REQUIRED_FIELDS = {
    "database_url",
    "cors_origins",
    "test_size",
    "cv_folds",
    "random_state",
    "target_recall",
    "risk_threshold_medium",
    "risk_threshold_high",
}


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("postgresql://u:p@host:5432/db", "postgresql+psycopg://u:p@host:5432/db"),
        ("postgres://u:p@host:5432/db", "postgresql+psycopg://u:p@host:5432/db"),
        # Already explicit: left untouched.
        ("postgresql+psycopg://u:p@host:5432/db", "postgresql+psycopg://u:p@host:5432/db"),
    ],
)
def test_database_url_is_normalised_to_the_installed_driver(settings_factory, given, expected):
    """Providers inject bare postgresql:// URLs, which resolve to the absent psycopg2."""
    assert settings_factory(database_url=given).database_url == expected


def test_tuneable_settings_have_no_code_side_default(monkeypatch):
    """The whole point: a missing value must fail, not be invented.

    Ignores the on-disk .env and clears the environment, so this asserts the absence of
    defaults in the model rather than the presence of a local .env.
    """
    for field in REQUIRED_FIELDS:
        monkeypatch.delenv(field.upper(), raising=False)

    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)

    missing = {err["loc"][0] for err in excinfo.value.errors()}
    assert missing == REQUIRED_FIELDS


def test_database_url_is_required(settings_factory, monkeypatch):
    """The database holds the model, the training data and the audit log."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None, **dict.fromkeys(REQUIRED_FIELDS - {"database_url"}, 1))
    assert "database_url" in {err["loc"][0] for err in excinfo.value.errors()}


def test_log_level_keeps_its_fallback(settings_factory):
    """log_level is the one tuneable that is still optional."""
    assert settings_factory().log_level == "INFO"


def test_csv_path_follows_data_dir(settings_factory, tmp_path):
    settings = settings_factory(data_dir=tmp_path, dataset_filename="x.csv")
    assert settings.csv_path == tmp_path / "x.csv"


def test_model_file_paths_are_gone(settings_factory):
    """The model lives in the database, so there are no artefact paths to configure."""
    settings = settings_factory()
    assert not hasattr(settings, "model_path")
    assert not hasattr(settings, "models_dir")


def test_version_comes_from_package_metadata():
    """The version is declared only in pyproject.toml; everything else reads it."""
    import dropout_service

    assert dropout_service.__version__ != "0.0.0+unknown"
