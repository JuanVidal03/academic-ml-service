"""Database engine, session factory, and the declarative base.

Table definitions do **not** live here -- each feature slice owns its own tables (see
``predictions/models.py``). This module only knows how to connect.

The database is optional: without ``DATABASE_URL`` the service predicts normally and simply
does not persist an audit log.
"""

import logging
from typing import Any

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

logger = logging.getLogger(__name__)

SCHEMA = "ml"


class Base(DeclarativeBase):
    """Declarative base for every table in the service.

    Anything that subclasses this is only registered on ``Base.metadata`` once its module
    has been imported. ``alembic/env.py`` therefore imports every slice's ``models`` module
    explicitly -- see the comment there before adding a new slice with tables.
    """


def create_db_engine(database_url: str) -> Engine:
    """Create an engine that survives idle connection drops.

    Managed Postgres services and connection proxies close idle connections, so
    ``pool_pre_ping`` and a recycle window keep the pool from handing out dead ones.
    """
    return create_engine(database_url, pool_pre_ping=True, pool_recycle=1800)


def create_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def check_connection(engine: Engine) -> dict[str, Any]:
    """Probe the database for the health endpoint. Never raises."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"connected": True}
    except Exception as exc:  # noqa: BLE001 - health checks must report, not raise
        logger.warning("Database health check failed: %s", exc)
        return {"connected": False, "error": str(exc)}
