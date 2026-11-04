"""Reading the training dataset out of the database, and seeding it in."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from dropout_service.ml.schema import ALL_FEATURES, TARGET_COLUMN, TARGET_MAP
from dropout_service.students.service import upsert_by_codigo
from dropout_service.training_data.models import AnalysisReport, TrainingRecord

logger = logging.getLogger(__name__)

#: CSV columns that identify a row rather than describe it. Never features.
STUDENT_CODE_COLUMN = "id_estudiante"
PERIOD_COLUMN = "periodo"


def count_records(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(TrainingRecord)) or 0


def load_training_frame(session: Session) -> pd.DataFrame:
    """Return every training record as a frame with one column per feature.

    Expands the JSONB payload back into columns, which is what
    ``ml/training/data.py:prepare_dataset`` expects. Features absent from a stored row come
    back as NaN and the pipeline's imputers handle them.
    """
    rows = session.execute(
        select(TrainingRecord.features, TrainingRecord.deserto).order_by(TrainingRecord.id)
    ).all()
    if not rows:
        raise RuntimeError("No training records. Seed the dataset first: uv run scripts/seed_db.py")

    frame = pd.DataFrame([dict(features) for features, _ in rows])
    # Reindex so the column set and order match the feature contract exactly.
    frame = frame.reindex(columns=ALL_FEATURES)
    frame[TARGET_COLUMN] = [bool(deserto) for _, deserto in rows]
    logger.info("Loaded %d training records from the database", len(frame))
    return frame


def seed_from_csv(session: Session, csv_path: Path) -> int:
    """Load the CSV into ``training_records`` and ``students``. Idempotent.

    Returns the number of rows inserted; 0 means the table was already populated.
    """
    existing = count_records(session)
    if existing:
        logger.info("training_records already holds %d rows; skipping seed", existing)
        return 0

    if not csv_path.exists():
        raise FileNotFoundError(f"Seed dataset not found at {csv_path}")

    df = pd.read_csv(csv_path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Seed dataset is missing the target column {TARGET_COLUMN!r}")

    target = df[TARGET_COLUMN].map(TARGET_MAP)
    if target.isna().any():
        unexpected = sorted(set(df.loc[target.isna(), TARGET_COLUMN].astype(str)))
        raise ValueError(f"Target has values outside {set(TARGET_MAP)}: {unexpected}")

    present = [c for c in ALL_FEATURES if c in df.columns]
    missing = sorted(set(ALL_FEATURES) - set(present))
    if missing:
        logger.warning("Seed dataset lacks %d feature column(s): %s", len(missing), missing)

    inserted = 0
    for (_, row), label in zip(df.iterrows(), target, strict=True):
        student = None
        if STUDENT_CODE_COLUMN in df.columns and pd.notna(row[STUDENT_CODE_COLUMN]):
            student = upsert_by_codigo(
                session,
                codigo=str(row[STUDENT_CODE_COLUMN]),
                programa=row.get("programa") if pd.notna(row.get("programa")) else None,
            )
        session.add(
            TrainingRecord(
                student_id=student.id if student else None,
                periodo=str(row[PERIOD_COLUMN]) if PERIOD_COLUMN in df.columns else None,
                features=_features_payload(row, present),
                deserto=bool(label),
                source="csv",
            )
        )
        inserted += 1

    session.commit()
    logger.info("Seeded %d training records from %s", inserted, csv_path.name)
    return inserted


def _features_payload(row: pd.Series, columns: list[str]) -> dict[str, Any]:
    """Build the JSONB payload, turning pandas NaN into JSON null."""
    payload: dict[str, Any] = {}
    for col in columns:
        value = row[col]
        if pd.isna(value):
            payload[col] = None
        elif hasattr(value, "item"):  # numpy scalar -> native Python
            payload[col] = value.item()
        else:
            payload[col] = value
    return payload


def save_analysis(session: Session, analysis: dict[str, Any]) -> AnalysisReport:
    """Persist one analysis run."""
    dataset = analysis.get("dataset", {})
    report = AnalysisReport(
        n_records=int(dataset.get("total_records", 0)),
        dropout_rate=float(dataset.get("dropout_rate", 0.0)),
        report=analysis,
    )
    session.add(report)
    session.commit()
    logger.info("Saved analysis report %d", report.id)
    return report
