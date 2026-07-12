"""S4 — chunk embeddings + retrieval (016 §T-14).

Embeds every chunk via the shared :func:`weeker.core.embed.embed` client and
stores the vector on ``Chunk.embedding`` (pgvector on Postgres, JSON on SQLite).
The HNSW index build is guarded to Postgres; retrieval uses a numpy cosine
fallback so it works identically on SQLite for tests and dev.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session

from weeker.core.embed import EmbedTransport, embed
from weeker.core.models import Chunk, Course
from weeker.ingest.textutil import cosine_matrix

_HNSW = {
    "chunks": "chunks_embedding_hnsw",
    "concepts": "concepts_embedding_hnsw",
}


def embed_chunks(
    session: Session,
    course: Course,
    transport: EmbedTransport | None = None,
    cache_dir: str | Path | None = None,
) -> int:
    """Embed all chunks of ``course`` and persist their vectors. Returns count."""
    chunks = list(session.scalars(select(Chunk).where(Chunk.course_id == course.id)))
    if not chunks:
        return 0
    vectors = embed([c.content for c in chunks], transport=transport, cache_dir=cache_dir)
    for c, vec in zip(chunks, vectors, strict=True):
        c.embedding = np.asarray(vec, dtype=np.float32)
    session.flush()
    return len(chunks)


def _chapter_matrix(
    session: Session, chapter_id: uuid.UUID
) -> tuple[list[Chunk], np.ndarray]:
    chunks = [
        c
        for c in session.scalars(select(Chunk).where(Chunk.chapter_id == chapter_id))
        if c.embedding is not None
    ]
    if not chunks:
        return [], np.empty((0, 0), dtype=np.float32)
    matrix = np.vstack([np.asarray(c.embedding, dtype=np.float32) for c in chunks])
    return chunks, matrix


def smoke_retrieval(
    session: Session,
    chapter_id: uuid.UUID,
    query: str,
    topk: int = 6,
    transport: EmbedTransport | None = None,
    cache_dir: str | Path | None = None,
) -> list[tuple[Chunk, float]]:
    """Cosine top-k chunks within a chapter for a text query (numpy fallback)."""
    chunks, matrix = _chapter_matrix(session, chapter_id)
    if not chunks:
        return []
    qvec = embed([query], transport=transport, cache_dir=cache_dir)[0]
    sims = cosine_matrix(qvec, matrix)
    order = np.argsort(-sims)[:topk]
    return [(chunks[i], float(sims[i])) for i in order]


def retrieve_by_vector(
    session: Session,
    course_id: uuid.UUID,
    query_vec: np.ndarray,
    topk: int,
    min_cos: float,
) -> list[tuple[Chunk, float]]:
    """Course-wide cosine retrieval from a precomputed query vector."""
    chunks = [
        c
        for c in session.scalars(select(Chunk).where(Chunk.course_id == course_id))
        if c.embedding is not None
    ]
    if not chunks:
        return []
    matrix = np.vstack([np.asarray(c.embedding, dtype=np.float32) for c in chunks])
    sims = cosine_matrix(np.asarray(query_vec, dtype=np.float32), matrix)
    order = np.argsort(-sims)
    hits = [(chunks[i], float(sims[i])) for i in order if sims[i] >= min_cos]
    return hits[:topk]


def build_hnsw_index(engine: Engine) -> bool:
    """Create pgvector HNSW indexes on Postgres; no-op (False) elsewhere."""
    if engine.dialect.name != "postgresql":
        return False
    with engine.begin() as conn:
        for table, name in _HNSW.items():
            conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS {name} ON {table} "
                    f"USING hnsw (embedding vector_cosine_ops)"
                )
            )
    return True
