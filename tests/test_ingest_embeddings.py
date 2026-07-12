"""T-14 gate: chunk embeddings, cosine smoke retrieval, full `verify ingest` (M1)."""

from __future__ import annotations

import hashlib

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from weeker.core.config import EMBED_DIM
from weeker.core.models import Base, Chapter, Chunk, Course, Page, Section, Source
from weeker.ingest import chunk as C
from weeker.ingest import embeddings as E
from weeker.verify import ingest_checks

_SENTENCE = (
    "A mutual fund pools money from investors to buy a diversified portfolio "
    "managed by a professional asset management company under regulation. "
)


class FakeEmbed:
    """Deterministic pseudo-embeddings: seeded by text hash, unit-scaled."""

    def __init__(self):
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            out.append(rng.standard_normal(EMBED_DIM).astype(np.float32))
        return np.vstack(out)


def _seed_course(s):
    course = Course(title="c")
    s.add(course)
    s.flush()
    source = Source(course_id=course.id, filename="f", sha256="z" * 64)
    s.add(source)
    s.flush()
    ch = Chapter(course_id=course.id, ordinal=1, title="Ch1", page_start=1, page_end=6)
    s.add(ch)
    s.flush()
    s.add(Section(chapter_id=ch.id, ordinal=1, title="S1", page_start=1, page_end=6))
    for pg in range(1, 7):
        s.add(Page(source_id=source.id, page_number=pg, text=_SENTENCE * 60, char_count=1))
    s.flush()
    C.chunk_course(s, course)
    s.flush()
    return course, ch


def test_embed_chunks_stores_vectors(tmp_path):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        course, _ = _seed_course(s)
        n = E.embed_chunks(s, course, transport=FakeEmbed(), cache_dir=tmp_path)
        s.commit()
        assert n >= 2
        for c in s.query(Chunk).all():
            assert c.embedding is not None
            assert len(np.asarray(c.embedding).ravel()) == EMBED_DIM


def test_smoke_retrieval_ranks_by_cosine(tmp_path):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        course, ch = _seed_course(s)
        fake = FakeEmbed()
        E.embed_chunks(s, course, transport=fake, cache_dir=tmp_path)
        s.commit()
        first = s.query(Chunk).first()
        hits = E.smoke_retrieval(
            s, ch.id, first.content, topk=3, transport=fake, cache_dir=tmp_path
        )
        assert hits
        # a chunk's own text retrieves itself at the top with cosine ~1.
        assert hits[0][0].id == first.id
        assert hits[0][1] > 0.99
        coss = [c for _chunk, c in hits]
        assert coss == sorted(coss, reverse=True)


def test_verify_ingest_full_gate(tmp_path):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        course, _ = _seed_course(s)
        # parse rows already stand in as pages; give the source a page_count.
        src = s.query(Source).first()
        src.page_count = 6
        E.embed_chunks(s, course, transport=FakeEmbed(), cache_dir=tmp_path)
        s.commit()
        failures = ingest_checks.verify_ingest(s, course.id)
        assert failures == [], failures


def test_build_hnsw_index_skips_sqlite():
    engine = create_engine("sqlite://")
    assert E.build_hnsw_index(engine) is False
