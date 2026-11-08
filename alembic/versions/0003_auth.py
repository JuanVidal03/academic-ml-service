"""Refresh tokens, and record which user requested each prediction.

Revision ID: 0003
Revises: 0002
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "ml"


def upgrade() -> None:
    # Refresh tokens are rows rather than self-contained JWTs so that logout can actually
    # revoke, and so a replayed token can be detected via the rotation chain.
    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        # A SHA-256 hex digest: a leaked database yields no usable tokens.
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("replaced_by_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], [f"{SCHEMA}.users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["replaced_by_id"], [f"{SCHEMA}.refresh_tokens.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_ml_refresh_tokens_token_hash",
        "refresh_tokens",
        ["token_hash"],
        unique=True,
        schema=SCHEMA,
    )
    op.create_index("ix_ml_refresh_tokens_user_id", "refresh_tokens", ["user_id"], schema=SCHEMA)

    # An audit log of access to student data should record who did the accessing.
    op.add_column("predictions", sa.Column("user_id", sa.Integer(), nullable=True), schema=SCHEMA)
    op.create_foreign_key(
        "fk_ml_predictions_user_id",
        "predictions",
        "users",
        ["user_id"],
        ["id"],
        source_schema=SCHEMA,
        referent_schema=SCHEMA,
        ondelete="SET NULL",
    )
    op.create_index("ix_ml_predictions_user_id", "predictions", ["user_id"], schema=SCHEMA)


def downgrade() -> None:
    op.drop_index("ix_ml_predictions_user_id", table_name="predictions", schema=SCHEMA)
    op.drop_constraint(
        "fk_ml_predictions_user_id", "predictions", schema=SCHEMA, type_="foreignkey"
    )
    op.drop_column("predictions", "user_id", schema=SCHEMA)

    op.drop_index("ix_ml_refresh_tokens_user_id", table_name="refresh_tokens", schema=SCHEMA)
    op.drop_index("ix_ml_refresh_tokens_token_hash", table_name="refresh_tokens", schema=SCHEMA)
    op.drop_table("refresh_tokens", schema=SCHEMA)
