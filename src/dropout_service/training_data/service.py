"""Reading the training dataset out of the database, and seeding it in."""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import bindparam, func, insert, select, update
from sqlalchemy.orm import Session

from dropout_service.ml.schema import ALL_FEATURES, TARGET_COLUMN, TARGET_MAP
from dropout_service.students.models import Student
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


#: Rows per round trip when resolving or writing students. The seed is routinely run
#: against a remote database, where a per-row SELECT/INSERT costs a full network
#: round trip -- 3,000 of them took half an hour against a managed Postgres in another
#: region. Everything below therefore works in batches.
BATCH_SIZE = 1000


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

    student_ids = _resolve_students(session, df)

    records = []
    for (_, row), label in zip(df.iterrows(), target, strict=True):
        codigo = _codigo_of(row) if STUDENT_CODE_COLUMN in df.columns else None
        records.append(
            {
                "student_id": student_ids.get(codigo) if codigo else None,
                "periodo": str(row[PERIOD_COLUMN]) if PERIOD_COLUMN in df.columns else None,
                "features": _features_payload(row, present),
                "deserto": bool(label),
                "source": "csv",
            }
        )

    if records:
        # One executemany rather than 3,000 ORM inserts: SQLAlchemy batches these into a
        # handful of multi-row INSERTs, so the cost stops scaling with network latency.
        session.execute(insert(TrainingRecord), records)

    session.commit()
    logger.info("Seeded %d training records from %s", len(records), csv_path.name)
    return len(records)


def _codigo_of(row: pd.Series) -> str | None:
    value = row[STUDENT_CODE_COLUMN]
    return str(value) if pd.notna(value) else None


def _resolve_students(session: Session, df: pd.DataFrame) -> dict[str, int]:
    """Return ``{codigo: student_id}`` for every student in the frame, creating the new ones.

    Does the whole frame in a fixed number of statements regardless of row count. The
    per-row equivalent (:func:`students.service.upsert_by_codigo`) is still the right
    choice for a single prediction; it is not for a bulk load.
    """
    if STUDENT_CODE_COLUMN not in df.columns:
        return {}

    # First programa seen for each code wins, matching the old row-by-row behaviour.
    wanted: dict[str, str | None] = {}
    for _, row in df.iterrows():
        codigo = _codigo_of(row)
        if codigo is None or codigo in wanted:
            continue
        programa = row.get("programa")
        wanted[codigo] = programa if pd.notna(programa) else None

    if not wanted:
        return {}

    codes = list(wanted)
    found: dict[str, tuple[int, str | None]] = {}
    for chunk in _chunks(codes, BATCH_SIZE):
        rows = session.execute(
            select(Student.codigo, Student.id, Student.programa).where(Student.codigo.in_(chunk))
        ).all()
        found.update({codigo: (sid, programa) for codigo, sid, programa in rows})

    resolved = {codigo: sid for codigo, (sid, _) in found.items()}

    # Existing rows whose programa the CSV disagrees with, updated in one statement.
    changes = [
        {"student_id": found[codigo][0], "new_programa": programa}
        for codigo, programa in wanted.items()
        if codigo in found and programa and found[codigo][1] != programa
    ]
    if changes:
        session.execute(
            update(Student)
            .where(Student.id == bindparam("student_id"))
            .values(programa=bindparam("new_programa")),
            changes,
        )

    new_rows = [
        {"codigo": codigo, "programa": programa}
        for codigo, programa in wanted.items()
        if codigo not in found
    ]
    if new_rows:
        inserted = session.execute(
            insert(Student).returning(Student.codigo, Student.id), new_rows
        ).all()
        resolved.update(dict(inserted))

    logger.info(
        "Resolved %d students (%d new, %d existing)", len(resolved), len(new_rows), len(found)
    )
    return resolved


def _chunks(items: list[str], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


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
