"""Engine and session factory.

Reads ``DATABASE_URL`` from the environment. Defaults to a local SQLite file so
the CLI and tests run without Postgres; production sets a ``postgresql+psycopg``
URL. Use :func:`get_engine` / :func:`get_session` for the shared handles or
:func:`session_scope` for a transactional block.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

DEFAULT_DATABASE_URL = "sqlite:///./weeker.db"


def database_url() -> str:
    """The active database URL (env ``DATABASE_URL`` or the local SQLite default)."""
    return os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)


@cache
def get_engine(url: str | None = None) -> Engine:
    """Return a cached engine for ``url`` (or the configured default)."""
    url = url or database_url()
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(url, future=True, connect_args=connect_args)


@cache
def _sessionmaker(url: str) -> sessionmaker:
    return sessionmaker(bind=get_engine(url), class_=Session, expire_on_commit=False, future=True)


def get_session(url: str | None = None) -> Session:
    """Open a new session bound to the configured engine."""
    return _sessionmaker(url or database_url())()


@contextmanager
def session_scope(url: str | None = None) -> Iterator[Session]:
    """Transactional session block: commit on success, rollback on error."""
    session = get_session(url)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
