"""Schema introspection gate.

Asserts every expected table exists and, on Postgres, that the pgvector-backed
HNSW indexes and the ``one_correct_per_question`` partial unique index are
present. Runnable against SQLite too — the pgvector-specific index assertions
are skipped there (SQLite has no ``vector`` type), but the partial unique index
still exists and is checked.

Run: ``python -m atlas.verify.schema`` — exit 0 on success, non-zero on any gap.
"""

from __future__ import annotations

import sys

from sqlalchemy import Engine, inspect

from atlas.core.db import get_engine

EXPECTED_TABLES = {
    "courses",
    "sources",
    "pages",
    "chapters",
    "sections",
    "chunks",
    "concepts",
    "concept_chunks",
    "objectives",
    "blueprint_weights",
    "case_groups",
    "questions",
    "attempts",
    "mastery",
    "flashcards",
    "misconceptions",
    "misconception_flags",
    "sessions",
    "mock_exams",
    "analytics_events",
    "review_queue",
}

PG_VECTOR_INDEXES = {"chunks_embedding_hnsw", "concepts_embedding_hnsw"}


def check(engine: Engine | None = None) -> list[str]:
    """Return a list of failures ([] means the schema is complete)."""
    engine = engine or get_engine()
    insp = inspect(engine)
    failures: list[str] = []

    tables = set(insp.get_table_names())
    missing = EXPECTED_TABLES - tables
    if missing:
        failures.append(f"missing tables: {sorted(missing)}")

    # Partial unique index on questions — required on all dialects.
    if "questions" in tables:
        q_idx = {ix["name"] for ix in insp.get_indexes("questions")}
        if "one_correct_per_question" not in q_idx:
            failures.append("missing partial unique index one_correct_per_question")

    # pgvector HNSW indexes — Postgres only.
    if engine.dialect.name == "postgresql":
        present = set()
        for table in ("chunks", "concepts"):
            if table in tables:
                present |= {ix["name"] for ix in insp.get_indexes(table)}
        missing_vec = PG_VECTOR_INDEXES - present
        if missing_vec:
            failures.append(f"missing pgvector HNSW indexes: {sorted(missing_vec)}")

    return failures


def main() -> int:
    engine = get_engine()
    failures = check(engine)
    dialect = engine.dialect.name
    if failures:
        print(f"SCHEMA FAIL ({dialect}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    n = len(EXPECTED_TABLES)
    print(f"SCHEMA OK ({dialect}): {n} tables + one_correct_per_question present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
