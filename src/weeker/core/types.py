"""Cross-dialect column types.

Postgres+pgvector is the production path, but unit tests run on SQLite. These
types emit native Postgres constructs (``vector(dim)``, ``uuid[]``, ``jsonb``)
on Postgres and degrade to JSON on SQLite so models import and tests run without
Postgres.
"""

from __future__ import annotations

import numpy as np
from sqlalchemy import ARRAY, JSON, Integer, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TypeDecorator


class Vector(TypeDecorator):
    """Embedding column: ``vector(dim)`` on Postgres, JSON list on SQLite."""

    impl = JSON
    cache_ok = True

    def __init__(self, dim: int):
        self.dim = dim
        super().__init__()

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector as PGVector

            return dialect.type_descriptor(PGVector(self.dim))
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            value = value.tolist()
        else:
            value = list(value)
        if dialect.name == "postgresql":
            return value  # pgvector's own bind processor takes it from here
        return value

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return np.asarray(value, dtype=np.float32)


# JSONB on Postgres, JSON elsewhere.
JSONType = JSON().with_variant(JSONB(), "postgresql")

# Native arrays on Postgres, JSON lists elsewhere.
UuidArray = JSON().with_variant(ARRAY(Uuid()), "postgresql")
TextArray = JSON().with_variant(ARRAY(Text()), "postgresql")
IntArray = JSON().with_variant(ARRAY(Integer()), "postgresql")
