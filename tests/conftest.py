"""Shared fixtures.

**No test touches PostgreSQL.** The session layer is faked, and the model is built in
memory from joblib bytes rather than read from a database. A real model is still trained
once per session, so assertions are about real behaviour: without it the suite could only
check `status_code in [200, 503]`, which passes whether the service works or not.

The trade-off is explicit: the schema, the migrations and the JSONB/BYTEA round-trip are
*not* covered here. Those are verified against the real database by the commands in
`docs/WALKTHROUGH.md`.
"""

import os
from contextlib import contextmanager
from datetime import UTC, datetime

# Set before any dropout_service import: main.py builds `app = create_app()` at module
# scope, which calls get_settings() and would otherwise read the developer's .env. These
# defaults make the suite independent of it -- previously the tests only *looked*
# independent because a complete .env happened to be present.
_TEST_ENV = {
    "DATABASE_URL": "postgresql://nobody@127.0.0.1:1/absent",
    "CORS_ORIGINS": '["http://testserver"]',
    "TEST_SIZE": "0.2",
    "CV_FOLDS": "5",
    "RANDOM_STATE": "42",
    "TARGET_RECALL": "0.80",
    "RISK_THRESHOLD_MEDIUM": "0.30",
    "RISK_THRESHOLD_HIGH": "0.60",
    "JWT_SECRET": "test-signing-key-that-is-at-least-32-characters-long",
    "ACCESS_TOKEN_MINUTES": "15",
    "REFRESH_TOKEN_DAYS": "7",
    # LOG_LEVEL is deliberately NOT pinned: it has a safe default, and pinning it here
    # leaks into Settings via os.environ even when _env_file=None, breaking the test that
    # asserts the fallback.
}
for _key, _value in _TEST_ENV.items():
    os.environ.setdefault(_key, _value)

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from dropout_service.config import Settings
from dropout_service.main import create_app
from dropout_service.ml.predict import Predictor
from dropout_service.ml.schema import ALL_FEATURES, TARGET_COLUMN, build_pipeline
from dropout_service.ml.training.data import prepare_dataset
from dropout_service.users.models import User

SAMPLE_SIZE = 600

#: A URL that parses but cannot connect. Startup tolerates an unreachable database, so the
#: app builds and the dependencies are overridden per test.
UNREACHABLE_DB = "postgresql://nobody@127.0.0.1:1/absent"

#: The settings with no code-side default. Supplying them explicitly also keeps the suite
#: independent of whatever is in the developer's .env.
REQUIRED_SETTINGS: dict = {
    "database_url": UNREACHABLE_DB,
    "cors_origins": ["http://testserver"],
    "test_size": 0.2,
    "cv_folds": 5,
    "random_state": 42,
    "target_recall": 0.80,
    "risk_threshold_medium": 0.30,
    "risk_threshold_high": 0.60,
    "jwt_secret": "test-signing-key-that-is-at-least-32-characters-long",
    "access_token_minutes": 15,
    "refresh_token_days": 7,
}


def _make_settings(**overrides) -> Settings:
    """Build Settings with every required field supplied, ignoring the on-disk .env."""
    return Settings(_env_file=None, **{**REQUIRED_SETTINGS, **overrides})


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    """The source dataset as a frame, standing in for what the database would return."""
    settings = _make_settings()
    return pd.read_csv(settings.csv_path)


@pytest.fixture(scope="session")
def dataset(raw_frame) -> tuple[pd.DataFrame, pd.Series]:
    return prepare_dataset(raw_frame)


@pytest.fixture(scope="session")
def model_artifact(dataset) -> tuple[bytes, dict]:
    """Train a small model once and return its joblib bytes plus metadata.

    This is what a `model_versions` row would hold, built without a database.
    """
    X, y = dataset
    sample = X.sample(n=SAMPLE_SIZE, random_state=0)
    pipeline = build_pipeline(LogisticRegression(max_iter=1000, random_state=0))
    pipeline.fit(sample, y.loc[sample.index])

    metadata = {
        "model_name": "LogisticRegression",
        "model_version": datetime.now(UTC).isoformat(),
        "features": ALL_FEATURES,
        "n_features": len(ALL_FEATURES),
        "decision_threshold": 0.4,
        "risk_bands": {"bajo": [0.0, 0.3], "medio": [0.3, 0.6], "alto": [0.6, 1.0]},
        "test_metrics": {"roc_auc": 0.75},
    }
    return Predictor.dump(pipeline), metadata


@pytest.fixture(scope="session")
def predictor(model_artifact) -> Predictor:
    artifact, metadata = model_artifact
    return Predictor.from_bytes(artifact, metadata, model_version_id=1)


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    return _make_settings(log_level="CRITICAL")


@pytest.fixture
def settings_factory():
    """Factory for ad-hoc Settings in tests that need their own configuration."""

    def _factory(**overrides) -> Settings:
        return _make_settings(**overrides)

    return _factory


class FakeSession:
    """Records what it was asked to persist; never talks to a database."""

    def __init__(self, scalars_result=None, scalar_result=0):
        self.added: list = []
        self.committed = False
        self.executed: list = []
        self._scalars_result = scalars_result or []
        self._scalar_result = scalar_result

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def add(self, obj):
        self.added.append(obj)

    def add_all(self, objs):
        self.added.extend(objs)

    def flush(self):
        for i, obj in enumerate(self.added, start=1):
            if getattr(obj, "id", None) is None:
                obj.id = i

    def commit(self):
        self.committed = True
        self.flush()

    def execute(self, statement):
        self.executed.append(statement)
        return self

    def scalars(self, statement=None):
        self.executed.append(statement)
        return self

    def scalar(self, statement=None):
        self.executed.append(statement)
        return self._scalar_result

    def first(self):
        return self._scalars_result[0] if self._scalars_result else None

    def all(self):
        return list(self._scalars_result)


@pytest.fixture
def fake_session():
    return FakeSession()


def _fake_user(role: str, user_id: int = 1) -> User:
    """A detached User instance: no session, no database."""
    return User(
        id=user_id,
        email=f"{role}@universidad.edu.co",
        password_hash="unused-in-tests",
        role=role,
        is_active=True,
    )


@contextmanager
def _client_for(test_settings, predictor, role: str | None):
    """Build a TestClient with the session layer faked and, optionally, a signed-in user.

    ``role=None`` leaves ``get_current_user`` real, so protected routes answer 401 -- that
    is how the unauthenticated cases are exercised. Supplying a role overrides *only*
    ``get_current_user``; ``require_admin`` stays real, so the adviser/admin distinction is
    genuinely tested rather than stubbed away.
    """
    from dropout_service.dependencies import (
        get_current_user,
        get_predictor,
        get_session_factory,
        require_session,
    )

    app = create_app(test_settings)
    session = FakeSession()
    app.dependency_overrides[get_predictor] = lambda: predictor
    app.dependency_overrides[get_session_factory] = lambda: lambda: session
    app.dependency_overrides[require_session] = lambda: session
    if role is not None:
        user = _fake_user(role)
        app.dependency_overrides[get_current_user] = lambda: user

    with TestClient(app) as test_client:
        # /health reads app.state directly -- it has to, since get_predictor raises 503
        # when the model is missing and health must still answer. The lifespan cleared it
        # (the test database is unreachable), so inject it after startup.
        app.state.predictor = predictor
        test_client.fake_session = session  # type: ignore[attr-defined]
        yield test_client


@pytest.fixture
def client(test_settings, predictor) -> TestClient:
    """Signed in as an adviser -- the ordinary case for the prediction endpoints."""
    with _client_for(test_settings, predictor, "adviser") as test_client:
        yield test_client


@pytest.fixture
def admin_client(test_settings, predictor) -> TestClient:
    """Signed in as an administrator."""
    with _client_for(test_settings, predictor, "admin") as test_client:
        yield test_client


@pytest.fixture
def anon_client(test_settings, predictor) -> TestClient:
    """Not signed in. Protected routes must answer 401."""
    with _client_for(test_settings, predictor, None) as test_client:
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


TARGET = TARGET_COLUMN
