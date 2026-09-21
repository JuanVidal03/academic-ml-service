"""Database models and engine configuration."""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from dropout_ml.config import get_settings

SCHEMA = "ml"


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


class TrainingRecord(Base):
    """Training data records (seeded from CSV + new verified records)."""

    __tablename__ = "training_records"
    __table_args__ = {"schema": SCHEMA}

    id = Column(Integer, primary_key=True, autoincrement=True)
    id_estudiante = Column(Integer, nullable=True)
    periodo = Column(String(20), nullable=True)
    semestre = Column(Integer, nullable=True)
    programa = Column(String(100), nullable=True)
    edad = Column(Integer, nullable=True)
    genero = Column(String(10), nullable=True)
    estrato = Column(Integer, nullable=True)
    tipo_admision = Column(String(50), nullable=True)
    modalidad = Column(String(20), nullable=True)
    distancia_a_universidad_km = Column(Float, nullable=True)
    trabaja = Column(String(5), nullable=True)
    horas_trabajo_semana = Column(Integer, nullable=True)
    ingreso_familiar = Column(Float, nullable=True)
    recibe_beca = Column(String(5), nullable=True)
    beca_perdida_en_periodo = Column(String(5), nullable=True)
    dificultad_economica = Column(String(5), nullable=True)
    tiene_hijos = Column(String(5), nullable=True)
    nivel_educativo_padres = Column(String(50), nullable=True)
    apoyo_familiar_percibido = Column(Integer, nullable=True)
    satisfaccion_con_programa = Column(Integer, nullable=True)
    participacion_extracurricular = Column(String(5), nullable=True)
    uso_tutorias = Column(String(5), nullable=True)
    materias_matriculadas = Column(Integer, nullable=True)
    materias_aprobadas = Column(Integer, nullable=True)
    materias_reprobadas = Column(Integer, nullable=True)
    materias_repetidas = Column(Integer, nullable=True)
    materia_mas_repetida = Column(String(100), nullable=True)
    area_materia_critica = Column(String(100), nullable=True)
    nota_promedio = Column(Float, nullable=True)
    promedio_semestre_anterior = Column(Float, nullable=True)
    promedio_acumulado = Column(Float, nullable=True)
    creditos_perdidos = Column(Integer, nullable=True)
    materias_canceladas = Column(Integer, nullable=True)
    semestres_repetidos = Column(Integer, nullable=True)
    numero_materias_con_perdida_historica = Column(Integer, nullable=True)
    porcentaje_asistencia = Column(Float, nullable=True)
    inasistencias = Column(Integer, nullable=True)
    horas_estudio_semana = Column(Float, nullable=True)
    deserto = Column(String(5), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    """Log of predictions made by the API."""

    __tablename__ = "predictions"
    __table_args__ = {"schema": SCHEMA}

    id = Column(Integer, primary_key=True, autoincrement=True)
    input_data = Column(JSONB, nullable=False)
    probability = Column(Float, nullable=False)
    prediction = Column(Boolean, nullable=False)
    risk_level = Column(String(10), nullable=False)
    model_version = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)


def get_engine(database_url: str | None = None):
    """Create SQLAlchemy engine."""
    url = database_url or get_settings().database_url
    if not url:
        return None
    return create_engine(url, pool_pre_ping=True)


def get_session_factory(engine):
    """Create session factory."""
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)


def create_schema_and_tables(engine) -> None:
    """Create the ml schema and all tables."""
    with engine.connect() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
        conn.commit()
    Base.metadata.create_all(engine)


def check_db_connection(engine) -> dict[str, Any]:
    """Check database connection health."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy", "connected": True}
    except Exception as e:
        return {"status": "unhealthy", "connected": False, "error": str(e)}
