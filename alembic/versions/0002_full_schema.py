"""Users, students, training data, the model registry, and analysis reports.

Moves the service's state off disk: the trained artefact and the training dataset now live
in the database, and predictions gain real foreign keys.

Revision ID: 0002
Revises: 0001
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCHEMA = "ml"


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        schema=SCHEMA,
    )
    # A unique index, matching `unique=True, index=True` on the ORM column. Declaring a
    # separate UniqueConstraint as well would leave autogenerate perpetually "drifting".
    op.create_index("ix_ml_users_email", "users", ["email"], unique=True, schema=SCHEMA)

    op.create_table(
        "students",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("codigo", sa.String(length=32), nullable=False),
        sa.Column("programa", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        schema=SCHEMA,
    )
    op.create_index("ix_ml_students_codigo", "students", ["codigo"], unique=True, schema=SCHEMA)

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("algorithm", sa.String(length=64), nullable=False),
        # The joblib bytes. A fitted sklearn pipeline is not JSON-serialisable.
        sa.Column("artifact", sa.LargeBinary(), nullable=False),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("decision_threshold", sa.Float(), nullable=False),
        sa.Column("n_features", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("version"),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_ml_model_versions_is_active", "model_versions", ["is_active"], schema=SCHEMA
    )
    # Exactly one active model, guaranteed by the database.
    op.create_index(
        "uq_model_versions_single_active",
        "model_versions",
        ["is_active"],
        unique=True,
        schema=SCHEMA,
        postgresql_where=sa.text("is_active"),
    )

    op.create_table(
        "training_records",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("student_id", sa.Integer(), nullable=True),
        sa.Column("periodo", sa.String(length=20), nullable=True),
        # The 34 features as one payload: ml/schema.py stays the single source of truth,
        # and adding a feature needs no migration.
        sa.Column("features", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("deserto", sa.Boolean(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["student_id"], [f"{SCHEMA}.students.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        schema=SCHEMA,
    )
    op.create_index(
        "ix_ml_training_records_student_id", "training_records", ["student_id"], schema=SCHEMA
    )
    op.create_index(
        "ix_ml_training_records_periodo", "training_records", ["periodo"], schema=SCHEMA
    )

    op.create_table(
        "analysis_reports",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("n_records", sa.Integer(), nullable=False),
        sa.Column("dropout_rate", sa.Float(), nullable=False),
        sa.Column("report", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        schema=SCHEMA,
    )

    # predictions: replace the free-text model version with a real foreign key, and allow
    # linking to a student. The table is empty in every environment, so no backfill.
    op.drop_column("predictions", "model_version", schema=SCHEMA)
    op.add_column(
        "predictions", sa.Column("model_version_id", sa.Integer(), nullable=True), schema=SCHEMA
    )
    op.add_column(
        "predictions", sa.Column("student_id", sa.Integer(), nullable=True), schema=SCHEMA
    )
    op.create_foreign_key(
        "fk_ml_predictions_model_version_id",
        "predictions",
        "model_versions",
        ["model_version_id"],
        ["id"],
        source_schema=SCHEMA,
        referent_schema=SCHEMA,
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_ml_predictions_student_id",
        "predictions",
        "students",
        ["student_id"],
        ["id"],
        source_schema=SCHEMA,
        referent_schema=SCHEMA,
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_ml_predictions_model_version_id", "predictions", ["model_version_id"], schema=SCHEMA
    )
    op.create_index("ix_ml_predictions_student_id", "predictions", ["student_id"], schema=SCHEMA)


def downgrade() -> None:
    op.drop_index("ix_ml_predictions_student_id", table_name="predictions", schema=SCHEMA)
    op.drop_index("ix_ml_predictions_model_version_id", table_name="predictions", schema=SCHEMA)
    op.drop_constraint(
        "fk_ml_predictions_student_id", "predictions", schema=SCHEMA, type_="foreignkey"
    )
    op.drop_constraint(
        "fk_ml_predictions_model_version_id", "predictions", schema=SCHEMA, type_="foreignkey"
    )
    op.drop_column("predictions", "student_id", schema=SCHEMA)
    op.drop_column("predictions", "model_version_id", schema=SCHEMA)
    op.add_column(
        "predictions",
        sa.Column("model_version", sa.String(length=64), nullable=True),
        schema=SCHEMA,
    )

    op.drop_table("analysis_reports", schema=SCHEMA)
    op.drop_index("ix_ml_training_records_periodo", table_name="training_records", schema=SCHEMA)
    op.drop_index("ix_ml_training_records_student_id", table_name="training_records", schema=SCHEMA)
    op.drop_table("training_records", schema=SCHEMA)
    op.drop_index("uq_model_versions_single_active", table_name="model_versions", schema=SCHEMA)
    op.drop_index("ix_ml_model_versions_is_active", table_name="model_versions", schema=SCHEMA)
    op.drop_table("model_versions", schema=SCHEMA)
    op.drop_index("ix_ml_students_codigo", table_name="students", schema=SCHEMA)
    op.drop_table("students", schema=SCHEMA)
    op.drop_index("ix_ml_users_email", table_name="users", schema=SCHEMA)
    op.drop_table("users", schema=SCHEMA)
