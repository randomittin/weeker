"""verify.schema.check passes against a freshly-created SQLite schema."""

from __future__ import annotations

from sqlalchemy import create_engine

from weeker.core.models import Base
from weeker.verify import schema


def test_check_passes_on_created_schema():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    assert schema.check(engine) == []


def test_check_flags_empty_db():
    engine = create_engine("sqlite://")  # nothing created
    failures = schema.check(engine)
    assert any("missing tables" in f for f in failures)
