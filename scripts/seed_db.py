#!/usr/bin/env python3
"""Load the CSV dataset into the database.

Idempotent: running it twice inserts nothing the second time.

Usage:
    uv run scripts/seed_db.py
"""

from dropout_service.config import configure_logging, get_settings
from dropout_service.database import create_db_engine, create_session_factory
from dropout_service.training_data.service import count_records, seed_from_csv


def main() -> None:
    configure_logging()
    settings = get_settings()
    engine = create_db_engine(settings.database_url)
    session_factory = create_session_factory(engine)

    with session_factory() as session:
        inserted = seed_from_csv(session, settings.csv_path)
        total = count_records(session)

    print("\n" + "=" * 62)
    print("Seed complete" if inserted else "Already seeded; nothing to do")
    print("=" * 62)
    print(f"  Inserted : {inserted:,}")
    print(f"  Total    : {total:,} training records")


if __name__ == "__main__":
    main()
