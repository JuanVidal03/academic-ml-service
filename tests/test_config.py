"""Settings behaviour."""

import pytest

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
def test_database_url_is_normalised_to_the_installed_driver(given, expected):
    """Providers inject bare postgresql:// URLs, which resolve to the absent psycopg2."""
    assert Settings(database_url=given).database_url == expected


def test_database_url_may_be_absent():
    assert Settings(database_url=None).database_url is None


def test_derived_paths_follow_models_dir(tmp_path):
    settings = Settings(models_dir=tmp_path)
    assert settings.model_path == tmp_path / "model.joblib"
    assert settings.metrics_path == tmp_path / "metrics.json"


def test_csv_path_follows_data_dir(tmp_path):
    settings = Settings(data_dir=tmp_path, dataset_filename="x.csv")
    assert settings.csv_path == tmp_path / "x.csv"


def test_version_comes_from_package_metadata():
    """The version is declared only in pyproject.toml; everything else reads it."""
    import dropout_service

    assert dropout_service.__version__ != "0.0.0+unknown"
