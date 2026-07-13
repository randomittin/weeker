"""Authored-bank loader: valid inserts, gate rejection, idempotent re-run.

Builds a tiny in-memory course (one chapter, two embedded chunks) and loads an
inline one-chapter authored fixture: two concepts, a valid question, and a
crafted G1-violating question (a "none of the above" option that passes the
pydantic contract but trips G1's banned-phrase rule). Asserts the valid content
lands, the bad question is rejected under G1, and a second run adds nothing.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from tests.test_learn_fixtures import seed_chapter, seed_course
from weeker.core.config import EMBED_DIM
from weeker.core.models import Base, CaseGroup, Chunk, Concept, Objective, Question
from weeker.generate.pipeline import STATUS_ACTIVE
from weeker.seed.load_authored import load_authored

_WORD = re.compile(r"[a-z0-9]+")


class BowEmbed:
    """Deterministic bag-of-words embedder (same shape as the pipeline test's).

    ``dim`` is a real knob so a test can prove the loader honours the *model's*
    output width rather than a hand-set ``config.EMBED_DIM``.
    """

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim

    def embed(self, texts):
        out = []
        for t in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for w in _WORD.findall(t.lower()):
                v[int(hashlib.sha256(w.encode()).hexdigest()[:8], 16) % self.dim] += 1.0
            out.append(v)
        return np.vstack(out) if out else np.empty((0, self.dim), dtype=np.float32)


def _mem() -> Session:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(engine, expire_on_commit=False)


def _seed(db, dim: int = EMBED_DIM):
    course = seed_course(db)
    course.embedding_dim = dim
    ch = seed_chapter(db, course, 1, "Fundamentals")
    embed = BowEmbed(dim)
    for content in ("Mutual funds pool investor capital", "Net asset value per unit"):
        chunk = Chunk(
            course_id=course.id,
            source_id=course.id,
            chapter_id=ch.id,
            content=content,
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
            token_count=5,
            page_start=1,
            page_end=1,
        )
        chunk.embedding = embed.embed([content])[0]
        db.add(chunk)
    db.flush()
    return course, ch, embed


def _valid_question(stem: str, difficulty: int = 2) -> dict:
    return {
        "stem": stem,
        "options": [
            {"key": "A", "text": "the correct pricing statement here", "misconception": None},
            {"key": "B", "text": "a plausible but wrong claim", "misconception": "confuses timing"},
            {"key": "C", "text": "another tempting distractor idea", "misconception": "wrong basis"},
            {"key": "D", "text": "a fourth alternative reading", "misconception": "misreads rule"},
        ],
        "correct_key": "A",
        "explanation": "A holds because the stated facts support it.",
        "difficulty": difficulty,
    }


def _g1_violating_question() -> dict:
    q = _valid_question("Which option below is deliberately malformed for the gate test today")
    q["options"][3]["text"] = "None of the above"  # passes contract, trips G1 banned phrase
    return q


def _fixture() -> dict:
    return {
        "chapter_ordinal": 1,
        "chapter_title": "Fundamentals",
        "concepts": [
            {
                "title": "Net asset value",
                "description": "Net asset value per unit computed daily for a fund",
                "difficulty": 2,
                "prerequisites": [],
                "keywords": ["nav", "pricing"],
                "misconceptions": ["nav equals market price"],
                "source_pages": [12, 13],
                "objectives": [
                    {"text": "Compute NAV per unit", "bloom": "apply", "assessment_style": "calculation"}
                ],
                "questions": [
                    _valid_question("A fund reports figures at close; which pricing statement holds"),
                    _g1_violating_question(),
                ],
            },
            {
                "title": "Expense ratio",
                "description": "Annual recurring cost charged as a percentage of assets",
                "difficulty": 1,
                "prerequisites": ["Net asset value"],
                "keywords": ["ter", "cost"],
                "misconceptions": ["fees do not affect returns"],
                "source_pages": [20],
                "objectives": [
                    {"text": "Interpret an expense ratio", "bloom": "understand", "assessment_style": "definition"}
                ],
                "questions": [
                    _valid_question("How does an annual recurring charge influence net returns over time")
                ],
            },
        ],
        "caselets": [],
    }


def _write_fixture(tmp_path) -> str:
    path = tmp_path / "ch01.json"
    path.write_text(json.dumps(_fixture()), encoding="utf-8")
    return str(path)


def test_load_authored_inserts_valid_and_rejects_g1(tmp_path):
    db = _mem()
    course, _, _ = _seed(db)
    path = _write_fixture(tmp_path)

    report = load_authored([path], db=db, embed_transport=BowEmbed(), cache_dir=tmp_path)

    # Two concepts, three objectives-per-file worth (2), two valid questions in, one G1 reject.
    assert report.concepts_inserted == 2
    assert report.objectives_inserted == 2
    assert report.questions_inserted == 2
    assert report.rejected_by_gate.get("G1") == 1
    assert report.questions_rejected == 1

    concepts = db.scalars(select(Concept).where(Concept.course_id == course.id)).all()
    assert {c.title for c in concepts} == {"Net asset value", "Expense ratio"}
    for c in concepts:
        assert c.review_status == "accepted"
        assert c.reviewed_at is not None
        assert c.embedding is not None
    nav = next(c for c in concepts if c.title == "Net asset value")
    assert nav.keywords == ["nav", "pricing"]
    assert nav.source_pages == [12, 13]
    assert nav.misconceptions == ["nav equals market price"]

    questions = db.scalars(
        select(Question).where(Question.course_id == course.id)
    ).all()
    assert len(questions) == 2
    for q in questions:
        assert q.status == STATUS_ACTIVE
        assert q.generator_model == "claude-authored"
        assert q.gate_log["G1"]["passed"] is True
        assert q.gate_log["G2"]["skipped"] is True
        assert q.gate_log["G3"]["skipped"] is True
        assert "None of the above" not in " ".join(o["text"] for o in q.options)

    assert db.scalar(select(func.count()).select_from(Objective)) == 2


def test_load_authored_is_idempotent(tmp_path):
    db = _mem()
    course, _, _ = _seed(db)
    path = _write_fixture(tmp_path)

    load_authored([path], db=db, embed_transport=BowEmbed(), cache_dir=tmp_path)
    q_before = db.scalar(select(func.count()).select_from(Question))
    c_before = db.scalar(select(func.count()).select_from(Concept))
    o_before = db.scalar(select(func.count()).select_from(Objective))

    report = load_authored([path], db=db, embed_transport=BowEmbed(), cache_dir=tmp_path)

    assert report.concepts_inserted == 0
    assert sum(c.concepts_skipped for c in report.chapters) == 2
    assert report.questions_inserted == 0
    assert report.objectives_inserted == 0
    assert db.scalar(select(func.count()).select_from(Question)) == q_before
    assert db.scalar(select(func.count()).select_from(Concept)) == c_before
    assert db.scalar(select(func.count()).select_from(Objective)) == o_before
    assert db.scalar(select(func.count()).select_from(CaseGroup)) == 0


def _pages_fixture(source_pages) -> dict:
    """One-concept fixture whose sole variable is ``source_pages``."""
    return {
        "chapter_ordinal": 1,
        "chapter_title": "Fundamentals",
        "concepts": [
            {
                "title": "Net asset value",
                "description": "Net asset value per unit computed daily for a fund",
                "difficulty": 2,
                "keywords": ["nav"],
                "source_pages": source_pages,
                "objectives": [
                    {"text": "Compute NAV per unit", "bloom": "apply", "assessment_style": "calculation"}
                ],
                "questions": [
                    _valid_question("A fund reports figures at close; which pricing statement holds")
                ],
            }
        ],
        "caselets": [],
    }


def test_load_authored_tolerates_non_int_source_pages(tmp_path):
    """Bug: section labels ('11.1.1') in source_pages crashed the whole file.

    The loader must keep integer-parseable pages, drop the rest, and still load
    the concept instead of rolling the file back with an int() ValueError.
    """
    db = _mem()
    course, _, _ = _seed(db)
    path = tmp_path / "ch01.json"
    # Mixed: section strings ("11.1.1", "12.1"), a bare int, an int-string, junk.
    path.write_text(json.dumps(_pages_fixture(["11.1.1", 13, "12.1", "14", "x"])), encoding="utf-8")

    report = load_authored([str(path)], db=db, embed_transport=BowEmbed(), cache_dir=tmp_path)

    assert report.file_errors == {}  # file did NOT roll back
    assert report.concepts_inserted == 1
    concept = db.scalars(select(Concept).where(Concept.course_id == course.id)).one()
    assert concept.source_pages == [13, 14]  # ints kept, section labels dropped


def test_load_authored_derives_dim_from_model_not_config(tmp_path):
    """Bug: loader depended on a hand-set WEEKER_EMBED_DIM that can mismatch.

    Load with an embedder whose true dim differs from ``config.EMBED_DIM`` but
    matches the course's stored ``embedding_dim``. It must succeed and persist
    concept vectors at the *model's* width — proving the dim is derived, not
    hardcoded to ``config.EMBED_DIM``.
    """
    model_dim = 96
    assert model_dim != EMBED_DIM  # the whole point: derived, not the config default
    db = _mem()
    course, _, embed = _seed(db, dim=model_dim)
    path = _write_fixture(tmp_path)

    report = load_authored([str(path)], db=db, embed_transport=embed, cache_dir=tmp_path)

    assert report.file_errors == {}
    assert report.concepts_inserted == 2
    concept = db.scalars(select(Concept).where(Concept.course_id == course.id)).first()
    assert len(np.asarray(concept.embedding).ravel()) == model_dim


def test_load_authored_rejects_embedder_dim_mismatch(tmp_path):
    """A model whose dim ≠ the course's ingestion dim is a hard error (no pad/truncate)."""
    db = _mem()
    course, _, _ = _seed(db, dim=EMBED_DIM)  # course ingested at EMBED_DIM
    path = _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="dim"):
        # embedder emits 96-dim vectors — a different space than the course's chunks
        load_authored([str(path)], db=db, embed_transport=BowEmbed(96), cache_dir=tmp_path)

    assert db.scalar(select(func.count()).select_from(Concept)) == 0  # nothing persisted
