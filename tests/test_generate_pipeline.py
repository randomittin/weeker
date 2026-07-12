"""T-24 gate: the generation pipeline end-to-end against fake LLM + embed.

Overgenerate → gate → persist: a clean run fills the active quota and records a
gate_log per question; a disagreeing blind solver routes questions to ``disputed``.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from atlas.core.config import EMBED_DIM
from atlas.core.models import (
    Base,
    BlueprintWeight,
    Chunk,
    ConceptChunk,
    Question,
)
from atlas.generate.pipeline import run_generation
from tests.test_learn_fixtures import new_user, seed_chapter, seed_concept, seed_course

_WORD = re.compile(r"[a-z0-9]+")


class BowEmbed:
    def embed(self, texts):
        out = []
        for t in texts:
            v = np.zeros(EMBED_DIM, dtype=np.float32)
            for w in _WORD.findall(t.lower()):
                v[int(hashlib.sha256(w.encode()).hexdigest()[:8], 16) % EMBED_DIM] += 1.0
            out.append(v)
        return np.vstack(out) if out else np.empty((0, EMBED_DIM), dtype=np.float32)


class PipelineLLM:
    """Dispatches by system prompt: generation / G2 grounding / G3 blind solve."""

    def __init__(self, solver_answer="A"):
        self.solver_answer = solver_answer
        self.gen_calls = 0

    def _questions(self):
        self.gen_calls += 1
        base = self.gen_calls * 100
        qs = []
        for i in range(5):
            n = base + i
            qs.append(
                {
                    "stem": f"Synthetic probe number {n} evaluating reasoning depth clearly",
                    "options": [
                        {"key": "A", "text": f"Alpha choice wording {n}"},
                        {"key": "B", "text": f"Beta alternative {n}", "misconception": "beta error"},
                        {"key": "C", "text": f"Gamma option {n}", "misconception": "gamma error"},
                        {"key": "D", "text": f"Delta candidate {n}", "misconception": "delta error"},
                    ],
                    "correct_key": "A",
                    "explanation": f"Alpha is right for probe {n} because of the stated reasoning.",
                    "difficulty": 2,
                }
            )
        return json.dumps({"questions": qs})

    def request(self, model, system, user):
        if system.startswith("You write ORIGINAL certification"):
            return self._questions()
        if system.startswith("You are a strict verifier"):
            return json.dumps({"supported": True, "ambiguous_option": None, "reason": "ok"})
        # blind solver
        return json.dumps(
            {"answer": self.solver_answer, "confidence": "sure", "rationale": "blind reasoning"}
        )


def _seed(db, *, chunk_text="Mutual funds pool investor capital into diversified holdings."):
    course = seed_course(db)
    ch = seed_chapter(db, course, 1, "Funds")
    concept = seed_concept(db, course, "Net asset value", chapter=ch)
    db.add(BlueprintWeight(course_id=course.id, chapter="Funds", weight=1.0))
    chunk = Chunk(
        course_id=course.id,
        source_id=course.id,
        chapter_id=ch.id,
        content=chunk_text,
        content_hash="h",
        token_count=10,
        page_start=1,
        page_end=1,
    )
    db.add(chunk)
    db.flush()
    db.add(ConceptChunk(concept_id=concept.id, chunk_id=chunk.id, cos=0.8))
    db.flush()
    return course, concept


def _mem():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(engine, expire_on_commit=False)


def test_pipeline_fills_active_quota(tmp_path):
    db = _mem()
    course, concept = _seed(db)
    report = run_generation(
        db,
        course,
        new_user(),
        count=3,
        llm_transport=PipelineLLM(solver_answer="A"),
        embed_transport=BowEmbed(),
        cache_dir=tmp_path,
    )
    assert report.accepted == 3
    active = db.scalars(
        select(Question).where(Question.course_id == course.id, Question.status == "active")
    ).all()
    assert len(active) == 3
    for q in active:
        assert q.gate_log["G1"]["passed"] is True
        assert q.gate_log["G2"]["passed"] is True
        assert q.gate_log["G3"]["disputed"] is False
        assert q.concept_id == concept.id
        assert q.correct_key == "A"
        assert float(q.q_rating) == 0.0  # difficulty 2 → 0.0 logit


def test_pipeline_routes_disagreement_to_disputed(tmp_path):
    db = _mem()
    course, _ = _seed(db)
    report = run_generation(
        db,
        course,
        new_user(),
        count=3,
        llm_transport=PipelineLLM(solver_answer="B"),  # solver always disagrees with key A
        embed_transport=BowEmbed(),
        cache_dir=tmp_path,
    )
    assert report.accepted == 0
    assert report.disputed > 0
    disputed = db.scalars(
        select(Question).where(Question.course_id == course.id, Question.status == "disputed")
    ).all()
    assert disputed
    assert all(q.gate_log["G3"]["disputed"] is True for q in disputed)


def test_pipeline_rejects_verbatim_lift(tmp_path):
    db = _mem()
    # A chunk whose text the generator will echo verbatim → G4 8-gram lift.
    lifted = "the quick brown fox jumps over the lazy sleeping dog again"
    course, _ = _seed(db, chunk_text=lifted)

    class LiftingLLM(PipelineLLM):
        def _questions(self):
            self.gen_calls += 1
            qs = [
                {
                    "stem": f"Explain why {lifted} today please number {self.gen_calls}-{i}",
                    "options": [
                        {"key": "A", "text": f"correct alpha {i}"},
                        {"key": "B", "text": f"beta {i}", "misconception": "b"},
                        {"key": "C", "text": f"gamma {i}", "misconception": "c"},
                        {"key": "D", "text": f"delta {i}", "misconception": "d"},
                    ],
                    "correct_key": "A",
                    "explanation": "alpha reasoning",
                    "difficulty": 1,
                }
                for i in range(5)
            ]
            return json.dumps({"questions": qs})

    report = run_generation(
        db,
        course,
        new_user(),
        count=3,
        llm_transport=LiftingLLM(),
        embed_transport=BowEmbed(),
        cache_dir=tmp_path,
    )
    assert report.accepted == 0
    assert report.by_gate.get("G4", 0) > 0
    n_active = db.scalar(
        select(func.count()).select_from(Question).where(Question.status == "active")
    )
    assert n_active == 0
