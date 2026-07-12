"""Real OpenAI-compatible embeddings response + EMBED_DIM enforcement + cache.

The embeddings HTTP path must parse ``data[].embedding`` (re-ordered by
``index``), reject a dim that doesn't match ``EMBED_DIM``, and surface a >=400
body clearly. Also confirms the on-disk cache short-circuits a second identical
call, and that SQLite JSON-vector storage retrieves the correct cosine top-k.
"""

from __future__ import annotations

import hashlib

import httpx
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from weeker.core import config
from weeker.core.embed import EmbedError, HTTPTransport, embed
from weeker.core.models import Base, Chapter, Chunk, Course, Source
from weeker.ingest import embeddings as E

DIM = config.EMBED_DIM


def _mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _emb_response(vectors):
    # OpenAI-compatible: data[] with per-row embedding + index (intentionally
    # returned out of order to exercise the sort-by-index).
    data = [
        {"object": "embedding", "index": i, "embedding": list(map(float, v))}
        for i, v in reversed(list(enumerate(vectors)))
    ]
    return {"object": "list", "data": data, "model": "m"}


def test_parses_data_embedding_and_reorders():
    calls = {"n": 0}
    v0 = np.arange(DIM, dtype=np.float32)
    v1 = np.arange(DIM, dtype=np.float32) + 1000.0

    def handler(_req):
        calls["n"] += 1
        return httpx.Response(200, json=_emb_response([v0, v1]))

    t = HTTPTransport(base_url="https://x/v1", api_key="k", client=_mock_client(handler))
    out = t.embed(["first", "second"])
    assert out.shape == (2, DIM)
    # index-sorted: row 0 is v0 even though the body listed it last
    assert np.array_equal(out[0], v0)
    assert np.array_equal(out[1], v1)
    assert calls["n"] == 1


def test_dim_mismatch_raises():
    def handler(_req):
        return httpx.Response(200, json=_emb_response([np.ones(8, dtype=np.float32)]))

    t = HTTPTransport(base_url="https://x/v1", api_key="k", client=_mock_client(handler))
    with pytest.raises(EmbedError, match=f"dim 8 != EMBED_DIM {DIM}"):
        t.embed(["x"])


def test_empty_data_envelope_raises():
    def handler(_req):
        return httpx.Response(200, json={"object": "list", "data": []})

    t = HTTPTransport(base_url="https://x/v1", api_key="k", client=_mock_client(handler))
    with pytest.raises(EmbedError, match="unexpected embeddings envelope"):
        t.embed(["x"])


def test_http_error_surfaces_body():
    def handler(_req):
        return httpx.Response(400, json={"error": {"message": "bad model"}})

    t = HTTPTransport(base_url="https://x/v1", api_key="k", client=_mock_client(handler))
    with pytest.raises(EmbedError, match="400.*bad model"):
        t.embed(["x"])


def test_cache_hit_via_http_transport(tmp_path):
    calls = {"n": 0}
    v = np.arange(DIM, dtype=np.float32)

    def handler(_req):
        calls["n"] += 1
        return httpx.Response(200, json=_emb_response([v]))

    t = HTTPTransport(base_url="https://x/v1", api_key="k", client=_mock_client(handler))
    a = embed(["cache me"], transport=t, cache_dir=tmp_path)
    b = embed(["cache me"], transport=t, cache_dir=tmp_path)
    assert np.array_equal(a, b)
    assert calls["n"] == 1  # second call served from disk cache


# ── SQLite (no pgvector): JSON vectors round-trip + correct cosine top-k ───────
class _OrthoEmbed:
    """Deterministic near-orthogonal vectors so top-k ordering is unambiguous."""

    def embed(self, texts):
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            slot = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16) % DIM
            out[i, slot] = 1.0
        return out


def test_sqlite_json_vectors_retrieve_correct_topk(tmp_path):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    fake = _OrthoEmbed()
    with Session(engine) as s:
        course = Course(title="c")
        s.add(course)
        s.flush()
        src = Source(course_id=course.id, filename="f", sha256="z" * 64)
        s.add(src)
        ch = Chapter(course_id=course.id, ordinal=1, title="C1", page_start=1, page_end=1)
        s.add(ch)
        s.flush()
        texts = ["alpha topic", "beta topic", "gamma topic", "delta topic"]
        for txt in texts:
            s.add(
                Chunk(
                    course_id=course.id,
                    source_id=src.id,
                    chapter_id=ch.id,
                    content=txt,
                    content_hash=hashlib.sha256(txt.encode()).hexdigest(),
                    token_count=2,
                    page_start=1,
                    page_end=1,
                )
            )
        s.flush()
        E.embed_chunks(s, course, transport=fake, cache_dir=tmp_path)
        s.commit()

        # stored as JSON on SQLite, read back as a vector
        stored = s.query(Chunk).all()
        assert all(c.embedding is not None for c in stored)
        assert len(np.asarray(stored[0].embedding).ravel()) == DIM

        qvec = fake.embed(["gamma topic"])[0]
        hits = E.retrieve_by_vector(s, course.id, qvec, topk=2, min_cos=0.0)
        assert hits, "no chunks retrieved"
        assert hits[0][0].content == "gamma topic"
        assert hits[0][1] > 0.99  # its own orthonormal vector → cosine ~1
        coss = [c for _c, c in hits]
        assert coss == sorted(coss, reverse=True)
