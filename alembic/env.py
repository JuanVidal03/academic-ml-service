"""Alembic migration environment.

Reads the connection string from the application settings so there is exactly one place
that knows how to reach the database.
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool, text

from alembic import context
from dropout_service.config import get_settings
from dropout_service.database import SCHEMA, Base

# LOAD-BEARING: a table is only registered on Base.metadata once its module is imported.
# Without these imports `alembic revision --autogenerate` would see the tables in the
# database but not in the metadata, and emit a migration that DROPS them. Add a line here
# for every new slice that owns tables.
from dropout_service.predictions import models as _predictions_models  # noqa: E402,F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    url = get_settings().database_url
    if not url:
        raise RuntimeError("DATABASE_URL is not set; cannot run migrations.")
    return url


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of executing it."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        include_schemas=True,
        version_table_schema=SCHEMA,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live database."""
    config.set_main_option("sqlalchemy.url", _database_url())
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        # Alembic keeps its version table inside the same schema, and it creates that
        # table before running any migration -- so the schema has to exist first or the
        # very first upgrade fails with InvalidSchemaName. Creating it here (idempotently)
        # breaks that chicken-and-egg.
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
        connection.commit()

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
            version_table_schema=SCHEMA,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
