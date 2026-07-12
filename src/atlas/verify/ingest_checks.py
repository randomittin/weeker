"""Ingestion gates as code (016 §T-11/§S2/§T-14 — the M1 gate).

Each function returns a list of human-readable failure strings; ``[]`` means the
stage passed. They read persisted rows via an injected session so they run
identically on SQLite (tests / dev) and Postgres (production).
"""

from __future__ import annotations

import uuid

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from atlas.core.config import (
    CHUNK_MAX,
    CHUNK_MIN,
    OCR_MIN_CHARS,
)
from atlas.core.models import Chapter, Chunk, Page, Source
from atlas.ingest.structure import check_structure

PARSE_COVERAGE_MIN = 0.98
CHUNK_COVERAGE_MIN = 0.97


def check_parse(session: Session, source_id: uuid.UUID) -> list[str]:
    """Parse gate: coverage ≥ 0.98, zero silent-empty pages."""
    pages = list(
        session.scalars(select(Page).where(Page.source_id == source_id).order_by(Page.page_number))
    )
    failures: list[str] = []
    if not pages:
        return [f"source {source_id}: no pages parsed"]
    silent_empty = [p.page_number for p in pages if not (p.text or "").strip() and not p.ocr_used]
    if silent_empty:
        failures.append(f"silent-empty pages: {silent_empty}")
    good = sum(1 for p in pages if len(p.text or "") >= OCR_MIN_CHARS)
    coverage = good / len(pages)
    if coverage < PARSE_COVERAGE_MIN:
        failures.append(f"parse coverage {coverage:.3f} < {PARSE_COVERAGE_MIN}")
    return failures


def check_structure_db(session: Session, course_id: uuid.UUID) -> list[str]:
    """Structure gate: chapters non-overlapping, cover ≥95% pages, depth ≤ 2."""
    chapters = list(
        session.scalars(select(Chapter).where(Chapter.course_id == course_id))
    )
    if not chapters:
        return [f"course {course_id}: no chapters"]
    page_count = session.scalar(
        select(func.max(Page.page_number)).join(Source, Page.source_id == Source.id).where(
            Source.course_id == course_id
        )
    ) or 0
    spans = [(c.page_start, c.page_end) for c in chapters]
    return check_structure(spans, page_count)


def check_chunks(session: Session, course_id: uuid.UUID) -> list[str]:
    """Chunk gate: every chunk token count in [CHUNK_MIN, CHUNK_MAX]; coverage ≥ 0.97."""
    chunks = list(session.scalars(select(Chunk).where(Chunk.course_id == course_id)))
    failures: list[str] = []
    if not chunks:
        return [f"course {course_id}: no chunks"]
    out_of_range = [
        c.token_count for c in chunks if not (CHUNK_MIN <= c.token_count <= CHUNK_MAX)
    ]
    if out_of_range:
        failures.append(
            f"{len(out_of_range)} chunk(s) outside [{CHUNK_MIN},{CHUNK_MAX}]: e.g. {out_of_range[:5]}"
        )
    # Coverage: chunk tokens vs. parsed page tokens for the course.
    from atlas.ingest.textutil import count_tokens

    page_tokens = 0
    for txt in session.scalars(
        select(Page.text).join(Source, Page.source_id == Source.id).where(
            Source.course_id == course_id
        )
    ):
        page_tokens += count_tokens(txt or "")
    chunk_tokens = sum(c.token_count for c in chunks)
    # Overlap inflates chunk tokens, so coverage is chunk/page bounded below by 1.
    if page_tokens:
        coverage = min(1.0, chunk_tokens / page_tokens)
        if coverage < CHUNK_COVERAGE_MIN:
            failures.append(f"chunk coverage {coverage:.3f} < {CHUNK_COVERAGE_MIN}")
    return failures


def check_embeddings(session: Session, course_id: uuid.UUID) -> list[str]:
    """Embedding gate: every chunk has an embedding of the right dimension."""
    from atlas.core.config import EMBED_DIM

    chunks = list(session.scalars(select(Chunk).where(Chunk.course_id == course_id)))
    if not chunks:
        return [f"course {course_id}: no chunks to embed"]
    missing = [c.id for c in chunks if c.embedding is None]
    failures: list[str] = []
    if missing:
        failures.append(f"{len(missing)} chunk(s) without embedding")
    for c in chunks:
        if c.embedding is not None and len(np.asarray(c.embedding).ravel()) != EMBED_DIM:
            failures.append(f"chunk {c.id}: embedding dim != {EMBED_DIM}")
            break
    return failures


def verify_ingest(session: Session, course_id: uuid.UUID) -> list[str]:
    """Full M1 ingestion gate: parse + structure + chunks + embeddings."""
    failures: list[str] = []
    for src_id in session.scalars(select(Source.id).where(Source.course_id == course_id)):
        failures += [f"[parse] {f}" for f in check_parse(session, src_id)]
    failures += [f"[structure] {f}" for f in check_structure_db(session, course_id)]
    failures += [f"[chunk] {f}" for f in check_chunks(session, course_id)]
    failures += [f"[embed] {f}" for f in check_embeddings(session, course_id)]
    return failures
