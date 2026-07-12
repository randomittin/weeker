"""initial schema — full Atlas DDL

Creates every table from the declarative metadata. On Postgres it additionally
enables the ``vector`` extension (needed for the embedding columns) and builds
HNSW cosine indexes on the embedding columns; both are guarded to the Postgres
dialect so the migration also runs on SQLite for unit tests.

Revision ID: 0001
Revises:
Create Date: 2026-07-12
"""

from __future__ import annotations

from alembic import op

from atlas.core.models import Base

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

_HNSW = [
    ("chunks", "chunks_embedding_hnsw"),
    ("concepts", "concepts_embedding_hnsw"),
]


def upgrade() -> None:
    bind = op.get_bind()
    is_pg = bind.dialect.name == "postgresql"

    if is_pg:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    Base.metadata.create_all(bind=bind)

    if is_pg:
        for table, name in _HNSW:
            op.execute(
                f"CREATE INDEX IF NOT EXISTS {name} ON {table} "
                f"USING hnsw (embedding vector_cosine_ops)"
            )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        for _table, name in _HNSW:
            op.execute(f"DROP INDEX IF EXISTS {name}")
    Base.metadata.drop_all(bind=bind)
