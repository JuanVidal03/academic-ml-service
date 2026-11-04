"""Settings behaviour."""

import pytest
from pydantic import ValidationError

from dropout_service.config import Settings


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("postgresql://u:p@host:5432/db", "postgresql+psycopg://u:p@host:5432/db"),
        ("postgres://u:p@host:5432/db", "postgresql+psycopg://u:p@host:5432/db"),
        # Already explicit: left untouched.
        ("postgresql+psycopg://u:p@host:5432/db", "postgresql+psycopg://u:p@host:5432/db"),
        ("sqlite:///local.db", "sqlite:///local.db"),
    ],
)
def test_database_url_is_normalised_to_the_installed_driver(settings_factory, given, expected):
    """Providers inject bare postgresql:// URLs, which resolve to the absent psycopg2."""
    assert settings_factory(database_url=given).database_url == expected


def test_database_url_may_be_absent(settings_factory):
    assert settings_factory(database_url=None).database_url is None


def test_derived_paths_follow_models_dir(settings_factory, tmp_path):
    settings = settings_factory(models_dir=tmp_path)
    assert settings.model_path == tmp_path / "model.joblib"
    assert settings.metrics_path == tmp_path / "metrics.json"


def test_csv_path_follows_data_dir(settings_factory, tmp_path):
    settings = settings_factory(data_dir=tmp_path, dataset_filename="x.csv")
    assert settings.csv_path == tmp_path / "x.csv"


def test_version_comes_from_package_metadata():
    """The version is declared only in pyproject.toml; everything else reads it."""
    import dropout_service

    assert dropout_service.__version__ != "0.0.0+unknown"


REQUIRED_FIELDS = {
    "cors_origins",
    "test_size",
    "cv_folds",
    "random_state",
    "target_recall",
    "risk_threshold_medium",
    "risk_threshold_high",
}


def test_tuneable_settings_have_no_code_side_default(monkeypatch):
    """The whole point of the change: a missing value must fail, not be invented.

    Ignores the on-disk .env and clears the environment, so this asserts the absence of
    defaults in the model rather than the presence of a local .env.
    """
    for field in REQUIRED_FIELDS:
        monkeypatch.delenv(field.upper(), raising=False)

    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)

    missing = {err["loc"][0] for err in excinfo.value.errors()}
    assert missing == REQUIRED_FIELDS


def test_optional_settings_keep_their_fallbacks(settings_factory):
    """database_url and log_level are deliberately still optional."""
    settings = settings_factory()
    assert settings.database_url is None
    assert settings.log_level == "INFO"
