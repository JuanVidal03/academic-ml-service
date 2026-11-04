"""Create the ml schema and the predictions audit table.

Revision ID: 0001
Revises:
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "ml"


def upgrade() -> None:
    # The schema itself is created by alembic/env.py before migrations run, because
    # Alembic's own version table lives in it.
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("input_data", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("prediction", sa.Boolean(), nullable=False),
        sa.Column("risk_level", sa.String(length=10), nullable=False),
        sa.Column("model_version", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        schema=SCHEMA,
    )
    # /predictions orders by created_at, so it needs an index to stay cheap as the log grows.
    op.create_index("ix_ml_predictions_created_at", "predictions", ["created_at"], schema=SCHEMA)


def downgrade() -> None:
    op.drop_index("ix_ml_predictions_created_at", table_name="predictions", schema=SCHEMA)
    op.drop_table("predictions", schema=SCHEMA)
    # The schema is deliberately left in place: alembic_version still lives there.
