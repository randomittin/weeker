"""G3 blind solver on its OWN endpoint/key, independent of the generator.

The ≠-family gate means the blind solver can (and usually should) target a
different provider than the generator. These tests lock in three things:

* ``config.SOLVER_BASE_URL`` / ``SOLVER_API_KEY`` exist and fall back to the main
  LLM endpoint/key when their env vars are unset;
* ``gates.get_solver_transport`` builds an :class:`HTTPTransport` from those;
* ``run_generation`` routes G3 solves to ``solver_transport`` while the generator
  and G2 stay on ``llm_transport`` — the two never cross.
"""

from __future__ import annotations

import json

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from tests.test_generate_pipeline import BowEmbed
from tests.test_learn_fixtures import new_user, seed_chapter, seed_concept, seed_course
from weeker.core import config
from weeker.core.models import Base, BlueprintWeight, Chunk, ConceptChunk, Question
from weeker.generate import gates
from weeker.generate.pipeline import run_generation


def test_config_solver_falls_back_to_llm_endpoint():
    # With WEEKER_SOLVER_* unset in the test env, the solver config mirrors the LLM one.
    assert config.SOLVER_BASE_URL == config.LLM_BASE_URL
    assert config.SOLVER_API_KEY == config.LLM_API_KEY


def test_get_solver_transport_uses_solver_config(monkeypatch):
    monkeypatch.setattr(gates, "_default_solver_transport", None)
    monkeypatch.setattr(config, "SOLVER_BASE_URL", "https://groq.example/openai/v1")
    monkeypatch.setattr(config, "SOLVER_API_KEY", "gsk_test")
    t = gates.get_solver_transport()
    assert t.base_url == "https://groq.example/openai/v1"
    assert t.api_key == "gsk_test"


class RecordingGen:
    """Generator + G2 transport (never asked to blind-solve)."""

    def __init__(self):
        self.gen_calls = 0
        self.saw_blind = False

    def _questions(self):
        self.gen_calls += 1
        base = self.gen_calls * 100
        qs = [
            {
                "stem": f"Independent probe {base + i} exploring applied judgement carefully",
                "options": [
                    {"key": "A", "text": f"Alpha wording {base + i}"},
                    {"key": "B", "text": f"Beta wording {base + i}", "misconception": "b"},
                    {"key": "C", "text": f"Gamma wording {base + i}", "misconception": "c"},
                    {"key": "D", "text": f"Delta wording {base + i}", "misconception": "d"},
                ],
                "correct_key": "A",
                "explanation": f"Alpha is correct for probe {base + i}.",
                "difficulty": 2,
            }
            for i in range(5)
        ]
        return json.dumps({"questions": qs})

    def request(self, model, system, user):
        if system.startswith("You write ORIGINAL certification"):
            return self._questions()
        if system.startswith("You are a strict verifier"):
            return json.dumps({"supported": True, "ambiguous_option": None, "reason": "ok"})
        # A blind-solve prompt should NEVER reach the generator transport.
        self.saw_blind = True
        return json.dumps({"answer": "A", "confidence": "sure", "rationale": "x"})


class RecordingSolver:
    """Blind-solver transport: only ever asked to solve (system = expert candidate)."""

    def __init__(self, answer="A"):
        self.answer = answer
        self.blind_calls = 0
        self.saw_other = False

    def request(self, model, system, user):
        if system.startswith("You are an expert exam candidate"):
            self.blind_calls += 1
            return json.dumps(
                {"answer": self.answer, "confidence": "sure", "rationale": "blind"}
            )
        self.saw_other = True
        return json.dumps({"supported": True, "ambiguous_option": None, "reason": "x"})


def _mem():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(engine, expire_on_commit=False)


def _seed(db):
    course = seed_course(db)
    ch = seed_chapter(db, course, 1, "Funds")
    concept = seed_concept(db, course, "Net asset value", chapter=ch)
    db.add(BlueprintWeight(course_id=course.id, chapter="Funds", weight=1.0))
    chunk = Chunk(
        course_id=course.id,
        source_id=course.id,
        chapter_id=ch.id,
        content="Mutual funds pool investor capital into diversified holdings.",
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


def test_run_generation_routes_g3_to_distinct_solver_transport(tmp_path):
    db = _mem()
    course, _ = _seed(db)
    gen = RecordingGen()
    solver = RecordingSolver(answer="A")

    report = run_generation(
        db,
        course,
        new_user(),
        count=3,
        llm_transport=gen,
        solver_transport=solver,
        embed_transport=BowEmbed(),
        cache_dir=tmp_path,
    )

    assert report.accepted == 3
    # G3 went to the solver transport, and ONLY to it.
    assert solver.blind_calls >= 3
    assert solver.saw_other is False
    assert gen.saw_blind is False
    # G3 verdict recorded from the solver's answer (agrees with key A → not disputed).
    active = db.scalars(
        select(Question).where(Question.course_id == course.id, Question.status == "active")
    ).all()
    assert all(q.gate_log["G3"]["disputed"] is False for q in active)


def test_solver_disagreement_still_disputes_across_endpoints(tmp_path):
    db = _mem()
    course, _ = _seed(db)
    report = run_generation(
        db,
        course,
        new_user(),
        count=3,
        llm_transport=RecordingGen(),
        solver_transport=RecordingSolver(answer="B"),  # different endpoint disagrees
        embed_transport=BowEmbed(),
        cache_dir=tmp_path,
    )
    assert report.accepted == 0
    assert report.disputed > 0
