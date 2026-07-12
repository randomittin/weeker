"""T-50 gate: `weeker refill` runs green end-to-end (generate --refill + bank
verify + status build) against fake transports on a fixture course.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
from sqlalchemy import func, select

from tests.test_learn_fixtures import (
    mem_session,
    new_user,
    seed_chapter,
    seed_concept,
    seed_course,
)
from weeker.core.config import EMBED_DIM
from weeker.core.models import BlueprintWeight, Chunk, ConceptChunk, Question
from weeker.learn import refill
from weeker.learn.status import render_status

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
    """Generation / G2 grounding / G3 blind-solve, dispatched by system prompt."""

    def __init__(self):
        self.gen_calls = 0

    def _questions(self):
        self.gen_calls += 1
        base = self.gen_calls * 100
        qs = [
            {
                "stem": f"Synthetic probe number {base + i} evaluating reasoning depth clearly",
                "options": [
                    {"key": "A", "text": f"Alpha choice wording {base + i}"},
                    {"key": "B", "text": f"Beta alternative {base + i}", "misconception": "beta"},
                    {"key": "C", "text": f"Gamma option {base + i}", "misconception": "gamma"},
                    {"key": "D", "text": f"Delta candidate {base + i}", "misconception": "delta"},
                ],
                "correct_key": "A",
                "explanation": f"Alpha is right for probe {base + i}.",
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
        return json.dumps({"answer": "A", "confidence": "sure", "rationale": "reasoning"})


def _seed(db):
    course = seed_course(db)
    ch = seed_chapter(db, course, 1, "Funds")
    concept = seed_concept(db, course, "Net asset value", chapter=ch)
    db.add(BlueprintWeight(course_id=course.id, chapter="Funds", weight=1.0))
    chunk = Chunk(
        course_id=course.id,
        source_id=course.id,
        chapter_id=ch.id,
        content="Mutual funds pool investor capital into diversified holdings across assets.",
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


def test_refill_runs_green_end_to_end(tmp_path):
    db = mem_session()
    course, concept = _seed(db)
    user = new_user()

    result = refill.run_refill(
        db,
        course,
        user,
        count=4,
        llm_transport=PipelineLLM(),
        embed_transport=BowEmbed(),
        cache_dir=tmp_path,
    )

    # 1. generation refilled the weak concept.
    assert result.generation.accepted > 0
    n_active = db.scalar(
        select(func.count()).select_from(Question).where(Question.status == "active")
    )
    assert n_active == result.generation.accepted

    # 2. bank gate ran and passed on the freshly-authored active bank.
    assert result.bank.total_active == n_active
    assert result.bank.passed is True

    # 3. status snapshot built and renders.
    assert result.status.course_title == course.title
    text = render_status(result.status)
    assert "Tonight's refill plan" in text
    assert "Prediction" in text
