"""Shared fixtures.

A real (small) model is trained once per session so tests assert real behaviour. Without
it the old suite could only check ``status_code in [200, 503]``, which passes whether the
service works or not.
"""

import json
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from dropout_service.config import Settings
from dropout_service.main import create_app
from dropout_service.ml.predict import Predictor
from dropout_service.ml.schema import ALL_FEATURES, build_pipeline
from dropout_service.ml.training.data import load_dataset

SAMPLE_SIZE = 600


@pytest.fixture(scope="session")
def dataset():
    """The full dataset as ``(X, y)``."""
    return load_dataset()


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory, dataset) -> Path:
    """Train a small model once and return the directory holding it."""
    X, y = dataset
    sample = X.sample(n=SAMPLE_SIZE, random_state=0)
    X_sample, y_sample = sample, y.loc[sample.index]

    pipeline = build_pipeline(LogisticRegression(max_iter=1000, random_state=0))
    pipeline.fit(X_sample, y_sample)

    directory = tmp_path_factory.mktemp("models")
    joblib.dump(pipeline, directory / "model.joblib")
    (directory / "metrics.json").write_text(
        json.dumps(
            {
                "model_name": "LogisticRegression",
                "model_version": "test",
                "features": ALL_FEATURES,
                "n_features": len(ALL_FEATURES),
                "decision_threshold": 0.4,
                "risk_bands": {"bajo": [0.0, 0.3], "medio": [0.3, 0.6], "alto": [0.6, 1.0]},
                "test_metrics": {"roc_auc": 0.75},
            }
        )
    )
    return directory


#: The settings that now have no code-side default. Tests must supply them explicitly,
#: which also keeps the suite independent of whatever is in the developer's .env.
REQUIRED_SETTINGS: dict = {
    "cors_origins": ["http://testserver"],
    "test_size": 0.2,
    "cv_folds": 5,
    "random_state": 42,
    "target_recall": 0.80,
    "risk_threshold_medium": 0.30,
    "risk_threshold_high": 0.60,
}


def _make_settings(**overrides) -> Settings:
    """Build Settings with every required field supplied, ignoring the on-disk .env."""
    return Settings(_env_file=None, **{**REQUIRED_SETTINGS, **overrides})


@pytest.fixture(scope="session")
def test_settings(models_dir) -> Settings:
    """Settings isolated from the developer's .env (no database, fixture model)."""
    return _make_settings(models_dir=models_dir, database_url=None, log_level="WARNING")


@pytest.fixture
def settings_factory(models_dir):
    """Factory for ad-hoc Settings in tests that need their own configuration."""

    def _factory(**overrides) -> Settings:
        # Merge rather than pass both, so a test can override models_dir itself.
        return _make_settings(**{"models_dir": models_dir, **overrides})

    return _factory


@pytest.fixture(scope="session")
def predictor(test_settings) -> Predictor:
    return Predictor.from_disk(test_settings.model_path, test_settings.metrics_path)


@pytest.fixture
def client(test_settings) -> TestClient:
    """A client whose app has completed its lifespan (model loaded)."""
    with TestClient(create_app(test_settings)) as test_client:
        yield test_client


@pytest.fixture
def student() -> dict:
    """A valid request body."""
    return {
        "semestre": 3,
        "programa": "Ingenieria de Sistemas",
        "materias_matriculadas": 6,
        "materias_aprobadas": 4,
        "materias_reprobadas": 2,
        "materias_repetidas": 1,
        "materias_canceladas": 0,
        "semestres_repetidos": 0,
        "numero_materias_con_perdida_historica": 2,
        "nota_promedio": 3.2,
        "promedio_semestre_anterior": 3.0,
        "promedio_acumulado": 3.1,
        "area_materia_critica": "Matematicas y Ciencias Basicas",
        "porcentaje_asistencia": 85.0,
        "inasistencias": 5,
        "horas_estudio_semana": 10.0,
        "edad": 20,
        "genero": "M",
        "estrato": 3,
        "tipo_admision": "Regular",
        "modalidad": "Diurna",
        "distancia_a_universidad_km": 5.0,
        "trabaja": "No",
        "horas_trabajo_semana": 0,
        "ingreso_familiar": 2000000,
        "recibe_beca": "No",
        "beca_perdida_en_periodo": "No",
        "dificultad_economica": "No",
        "tiene_hijos": "No",
        "nivel_educativo_padres": "Secundaria",
        "apoyo_familiar_percibido": 4,
        "satisfaccion_con_programa": 4,
        "participacion_extracurricular": "No",
        "uso_tutorias": "Si",
    }
