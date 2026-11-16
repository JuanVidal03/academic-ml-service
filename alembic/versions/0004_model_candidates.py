"""Store every compared algorithm, not only the winner.

So the UI can show what each model would say about a given student.

Revision ID: 0004
Revises: 0003
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "ml"


def upgrade() -> None:
    op.create_table(
        "model_candidates",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=False),
        sa.Column("algorithm", sa.String(length=64), nullable=False),
        # joblib bytes, as for the parent version: a fitted pipeline is not JSON.
        sa.Column("artifact", sa.LargeBinary(), nullable=False),
        sa.Column("cv_metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("test_metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        # Per candidate: probabilities across algorithms are not comparably calibrated.
        sa.Column("decision_threshold", sa.Float(), nullable=False),
        sa.Column("is_best", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["model_version_id"], [f"{SCHEMA}.model_versions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "model_version_id", "algorithm", name="uq_model_candidates_version_algo"
        ),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_ml_model_candidates_model_version_id",
        "model_candidates",
        ["model_version_id"],
        schema=SCHEMA,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_ml_model_candidates_model_version_id", table_name="model_candidates", schema=SCHEMA
    )
    op.drop_table("model_candidates", schema=SCHEMA)
