"""T-15 gate: concept extraction, embedding merge, DAG cycle check, concept→chunk map."""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from weeker.core.config import CONCEPT_MERGE_COS, EMBED_DIM
from weeker.core.models import (
    Base,
    Chapter,
    Chunk,
    Concept,
    ConceptChunk,
    Course,
    Page,
    Section,
    Source,
)
from weeker.ingest import chunk as C
from weeker.ingest import concepts as K
from weeker.ingest import embeddings as E

_WORD = re.compile(r"[a-z]+")


class BowEmbed:
    """Hashing bag-of-words embeddings so texts sharing words correlate."""

    def embed(self, texts):
        out = []
        for t in texts:
            v = np.zeros(EMBED_DIM, dtype=np.float32)
            for w in _WORD.findall(t.lower()):
                v[int(hashlib.sha256(w.encode()).hexdigest()[:8], 16) % EMBED_DIM] += 1.0
            out.append(v)
        return np.vstack(out)


def test_merge_concepts_collapses_near_dupes():
    drafts = [
        K.ConceptDraft(title="Net Asset Value", description="price per unit", difficulty=1,
                       keywords=["nav"], misconceptions=[], source_pages=[3], chapter_id=None),
        K.ConceptDraft(title="NAV", description="price per unit of a fund", difficulty=2,
                       keywords=["price"], misconceptions=["market price"], source_pages=[4],
                       chapter_id=None),
        K.ConceptDraft(title="Expense Ratio", description="annual cost", difficulty=2,
                       keywords=["ter"], misconceptions=[], source_pages=[9], chapter_id=None),
    ]
    base = np.zeros((3, EMBED_DIM), dtype=np.float32)
    base[0, 0] = 1.0
    base[1, 0] = 0.999  # ~identical to draft 0 → merge
    base[1, 1] = 0.001
    base[2, 5] = 1.0     # distinct
    kept = K.merge_concepts(drafts, base, threshold=CONCEPT_MERGE_COS)
    assert len(kept) == 2
    merged = next(k for k in kept if k.title in ("Net Asset Value", "NAV"))
    # union of keywords / source pages across the merged pair
    assert set(merged.keywords) >= {"nav", "price"}
    assert set(merged.source_pages) >= {3, 4}


def test_detect_cycle():
    assert K.detect_cycle({"A": ["B"], "B": ["C"], "C": []}) == []
    cyc = K.detect_cycle({"A": ["B"], "B": ["A"]})
    assert cyc and set(cyc) == {"A", "B"}


CONCEPTS_JSON = {
    "concepts": [
        {"title": "Mutual Fund", "description": "A mutual fund pools investor money.",
         "difficulty": 1, "prerequisites": [], "keywords": ["fund", "pool"],
         "misconceptions": ["stock"], "source_pages": [1]},
        {"title": "Net Asset Value", "description": "NAV is fund value per unit price.",
         "difficulty": 2, "prerequisites": ["Mutual Fund"], "keywords": ["nav", "value"],
         "misconceptions": [], "source_pages": [2]},
    ]
}


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def request(self, model, system, user):
        self.calls += 1
        return json.dumps(CONCEPTS_JSON)


def _seed(s):
    course = Course(title="NISM")
    s.add(course)
    s.flush()
    source = Source(course_id=course.id, filename="f", sha256="q" * 64)
    s.add(source)
    s.flush()
    ch = Chapter(course_id=course.id, ordinal=1, title="Funds", page_start=1, page_end=6)
    s.add(ch)
    s.flush()
    s.add(Section(chapter_id=ch.id, ordinal=1, title="S1", page_start=1, page_end=6))
    body = (
        "A mutual fund pools investor money to buy securities. "
        "Net asset value NAV is the fund value per unit price computed daily. "
    )
    for pg in range(1, 7):
        s.add(Page(source_id=source.id, page_number=pg, text=body * 40, char_count=1))
    s.flush()
    C.chunk_course(s, course)
    s.flush()
    return course, ch


def test_run_concepts_end_to_end_leaves_pending(tmp_path):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        course, _ = _seed(s)
        bow = BowEmbed()
        E.embed_chunks(s, course, transport=bow, cache_dir=tmp_path / "e")
        s.commit()
        made = K.run_concepts(
            s, course, transport=FakeLLM(), embed_transport=bow, cache_dir=tmp_path / "c"
        )
        s.commit()
        assert made
        rows = s.query(Concept).all()
        assert rows
        assert all(c.review_status == "pending" for c in rows)
        assert all(c.embedding is not None for c in rows)
        # concept→chunk mapping produced (bag-of-words overlap clears min cosine)
        assert s.query(ConceptChunk).count() > 0


def test_map_concept_chunks_uses_embedding(tmp_path):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        course, _ = _seed(s)
        E.embed_chunks(s, course, transport=BowEmbed(), cache_dir=tmp_path)
        s.commit()
        target = s.query(Chunk).first()
        concept = Concept(course_id=course.id, title="t", description="d")
        concept.embedding = np.asarray(target.embedding, dtype=np.float32)
        s.add(concept)
        s.flush()
        n = K.map_concept_chunks(s, course, concept)
        s.commit()
        assert n >= 1
        links = s.query(ConceptChunk).filter(ConceptChunk.concept_id == concept.id).all()
        assert any(link.chunk_id == target.id for link in links)
        assert float(next(link.cos for link in links if link.chunk_id == target.id)) > 0.99
