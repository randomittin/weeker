"""T-26 gate: caselet engine — contract, G6 scenario-dependence, e2e, drill, verify.

A caselet parses to a 120-220-word scenario + exactly 5 questions (positions 1-5,
marks 1|2). G6 blind-solves each question WITHOUT the scenario and flags any that
is still answered confidently correct (scenario-independent) for regeneration.
``verify_bank --caselets`` applies the ≥25-active / 100%-G6 / per-question bar.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from atlas.core.config import EMBED_DIM
from atlas.core.models import (
    Base,
    BlueprintWeight,
    CaseGroup,
    Chunk,
    ConceptChunk,
    Question,
)
from atlas.generate import caselets
from atlas.verify.bank_checks import verify_bank
from tests.test_learn_fixtures import new_user, seed_chapter, seed_concept, seed_course

_WORD = re.compile(r"[a-z0-9]+")


def _scenario(n_words=150):
    words = ("Priya age thirty five earns twelve lakh holds equity funds and bonds "
             "planning retirement goals with moderate risk tolerance across accounts ").split()
    out = []
    while len(out) < n_words:
        out.extend(words)
    return " ".join(out[:n_words])


def _cq(pos, marks=1, correct="A", salt=0):
    # Mostly-unique tokens per (salt,pos) so distinct caselet stems stay well below
    # the G5 dedup cosine while sharing only a little boilerplate.
    uniq = " ".join(f"w{salt}q{pos}t{k}" for k in range(6))
    return {
        "stem": f"Using the scenario compute {uniq} for this investor",
        "options": [
            {"key": "A", "text": f"correct alpha {salt}-{pos}"},
            {"key": "B", "text": f"beta {salt}-{pos}", "misconception": "b"},
            {"key": "C", "text": f"gamma {salt}-{pos}", "misconception": "c"},
            {"key": "D", "text": f"delta {salt}-{pos}", "misconception": "d"},
        ],
        "correct_key": correct,
        "explanation": f"alpha is right at {pos} given the scenario numbers",
        "difficulty": 2,
        "case_position": pos,
        "marks": marks,
    }


def _caselet_payload(marks=1, salt=0):
    return {"scenario": _scenario(), "questions": [_cq(p, marks, salt=salt) for p in range(1, 6)]}


class BowEmbed:
    def embed(self, texts):
        out = []
        for t in texts:
            v = np.zeros(EMBED_DIM, dtype=np.float32)
            for w in _WORD.findall(t.lower()):
                v[int(hashlib.sha256(w.encode()).hexdigest()[:8], 16) % EMBED_DIM] += 1.0
            out.append(v)
        return np.vstack(out) if out else np.empty((0, EMBED_DIM), dtype=np.float32)


class CaseletLLM:
    """Dispatch: caselet gen / G2 / caselet combined solve (G3) / per-Q blind solve (G6)."""

    def __init__(self, *, g6_independent=False, g3_answer="A", marks=1):
        self.g6_independent = g6_independent
        self.g3_answer = g3_answer
        self.marks = marks
        self.gen_calls = 0

    def request(self, model, system, user):
        if system.startswith("You write ORIGINAL case-based"):
            self.gen_calls += 1
            marks = 2 if '"marks": 2' in user else 1
            return json.dumps(_caselet_payload(marks, salt=self.gen_calls))
        if system.startswith("You are a strict verifier"):
            return json.dumps({"supported": True, "ambiguous_option": None, "reason": "ok"})
        if "CASE STUDY" in system:  # combined caselet blind solve (G3)
            answers = [
                {"position": p, "answer": self.g3_answer, "confidence": "sure", "rationale": "r"}
                for p in range(1, 6)
            ]
            return json.dumps({"answers": answers})
        # per-question blind solve without scenario (G6)
        if self.g6_independent:
            return json.dumps({"answer": "A", "confidence": "sure", "rationale": "no scenario needed"})
        return json.dumps({"answer": "B", "confidence": "unsure", "rationale": "need the scenario"})


# ── contract ─────────────────────────────────────────────────────────────────
def test_caselet_contract_parses():
    c = caselets.GeneratedCaselet.model_validate(_caselet_payload())
    assert len(c.questions) == 5
    assert [q.case_position for q in c.questions] == [1, 2, 3, 4, 5]
    assert all(q.marks == 1 for q in c.questions)


def test_caselet_scenario_length_enforced():
    bad = _caselet_payload()
    bad["scenario"] = "too short scenario"
    with pytest.raises(ValidationError):
        caselets.GeneratedCaselet.model_validate(bad)


def test_caselet_requires_five_questions():
    bad = _caselet_payload()
    bad["questions"] = bad["questions"][:4]
    with pytest.raises(ValidationError):
        caselets.GeneratedCaselet.model_validate(bad)


def test_caselet_marks_must_be_one_or_two():
    bad = _caselet_payload()
    bad["questions"][0]["marks"] = 3
    with pytest.raises(ValidationError):
        caselets.GeneratedCaselet.model_validate(bad)


# ── G6 ───────────────────────────────────────────────────────────────────────
def test_g6_flags_scenario_independent_question():
    qs = caselets.GeneratedCaselet.model_validate(_caselet_payload()).questions
    log: dict = {}
    # solver answers correctly WITHOUT the scenario → scenario-independent → fail
    res = gates_g6 = caselets.check_g6(
        qs,
        transport=CaseletLLM(g6_independent=True),
        model="solver",
        gate_log=log,
    )
    assert not res.passed
    assert log["G6"]["independent_positions"]
    assert log["G6"]["passed"] is False
    del gates_g6


def test_g6_passes_when_scenario_required():
    qs = caselets.GeneratedCaselet.model_validate(_caselet_payload()).questions
    log: dict = {}
    res = caselets.check_g6(
        qs, transport=CaseletLLM(g6_independent=False), model="solver", gate_log=log
    )
    assert res.passed
    assert log["G6"]["independent_positions"] == []


# ── e2e + drill ────────────────────────────────────────────────────────────────
def _mem():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(engine, expire_on_commit=False)


def _seed(db, n_concepts=5):
    course = seed_course(db)
    ch = seed_chapter(db, course, 1, "Funds")
    db.add(BlueprintWeight(course_id=course.id, chapter="Funds", weight=1.0))
    for i in range(n_concepts):
        concept = seed_concept(db, course, f"concept {i}", chapter=ch)
        chunk = Chunk(
            course_id=course.id,
            source_id=course.id,
            chapter_id=ch.id,
            content=f"Grounding text about topic {i} for funds investors and taxation.",
            content_hash=f"h{i}",
            token_count=10,
            page_start=1,
            page_end=1,
        )
        db.add(chunk)
        db.flush()
        db.add(ConceptChunk(concept_id=concept.id, chunk_id=chunk.id, cos=0.8))
    db.flush()
    return course


def test_run_caselets_persists_and_passes_g6(tmp_path):
    db = _mem()
    course = _seed(db)
    report = caselets.run_caselets(
        db,
        course,
        new_user(),
        count=2,
        two_mark_count=1,
        llm_transport=CaseletLLM(g6_independent=False),
        embed_transport=BowEmbed(),
        cache_dir=tmp_path,
    )
    assert report.accepted == 2
    groups = db.scalars(select(CaseGroup).where(CaseGroup.course_id == course.id)).all()
    assert len(groups) == 2
    for g in groups:
        assert g.gate_log["G6"]["passed"] is True
        qs = db.scalars(
            select(Question).where(Question.case_group_id == g.id).order_by(Question.case_position)
        ).all()
        assert [q.case_position for q in qs] == [1, 2, 3, 4, 5]
        assert all(q.status == "active" for q in qs)
        for q in qs:
            assert q.gate_log["G1"]["passed"] and q.gate_log["G2"]["passed"]
    # one caselet is 2-mark (all five questions worth 2)
    two_mark_groups = [
        g
        for g in groups
        if all(
            float(q.marks) == 2
            for q in db.scalars(select(Question).where(Question.case_group_id == g.id))
        )
    ]
    assert len(two_mark_groups) == 1


def test_run_caselets_regenerates_on_g6_failure(tmp_path):
    db = _mem()
    course = _seed(db)

    class FlipLLM(CaseletLLM):
        """G6-independent on the first caselet attempt, scenario-dependent after."""

        def __init__(self):
            super().__init__(g6_independent=True)
            self._probe = 0

        def request(self, model, system, user):
            if system.startswith("You write ORIGINAL case-based"):
                return super().request(model, system, user)
            if system.startswith("You are a strict verifier") or "CASE STUDY" in system:
                return super().request(model, system, user)
            # per-question G6 probe: first caselet's probes answer confidently correct,
            # then flip to scenario-dependent so a regeneration passes.
            self._probe += 1
            if self._probe <= 5:
                return json.dumps({"answer": "A", "confidence": "sure", "rationale": "x"})
            return json.dumps({"answer": "B", "confidence": "unsure", "rationale": "y"})

    report = caselets.run_caselets(
        db,
        course,
        new_user(),
        count=1,
        two_mark_count=0,
        llm_transport=FlipLLM(),
        embed_transport=BowEmbed(),
        cache_dir=tmp_path,
    )
    assert report.accepted == 1
    assert report.g6_regenerated >= 1


def test_caselet_drill_returns_grouped_active(tmp_path):
    db = _mem()
    course = _seed(db)
    caselets.run_caselets(
        db,
        course,
        new_user(),
        count=2,
        two_mark_count=0,
        llm_transport=CaseletLLM(),
        embed_transport=BowEmbed(),
        cache_dir=tmp_path,
    )
    drill = caselets.caselet_drill(db, course.id)
    assert len(drill) == 2
    for group, qs in drill:
        assert group.status == "active"
        assert len(qs) == 5


# ── verify_bank --caselets ───────────────────────────────────────────────────
def test_verify_bank_caselets_target_and_gates(tmp_path):
    db = _mem()
    course = _seed(db)
    # Build 25 active caselets directly (10 at 2-mark) with clean gate logs.
    concept = db.scalars(select(Question.concept_id)).first()
    from atlas.core.models import Concept

    cid = db.scalars(select(Concept.id).where(Concept.course_id == course.id)).first()
    for i in range(25):
        marks = 2 if i < 10 else 1
        cg = CaseGroup(
            course_id=course.id,
            scenario=_scenario(),
            concept_ids=[str(cid)],
            status="active",
            gate_log={"G6": {"passed": True, "independent_positions": []}},
            generator_model="m",
        )
        db.add(cg)
        db.flush()
        for p in range(1, 6):
            db.add(
                Question(
                    course_id=course.id,
                    concept_id=cid,
                    case_group_id=cg.id,
                    stem=f"c{i} q{p}",
                    options=[
                        {"key": "A", "text": "a"},
                        {"key": "B", "text": "b", "misconception": "b"},
                        {"key": "C", "text": "c", "misconception": "c"},
                        {"key": "D", "text": "d", "misconception": "d"},
                    ],
                    correct_key="A",
                    explanation="e",
                    difficulty=2,
                    marks=marks,
                    case_position=p,
                    status="active",
                    gate_log={
                        "G1": {"passed": True},
                        "G2": {"passed": True},
                        "G4": {"passed": True},
                    },
                )
            )
    db.flush()
    _ = concept
    report = verify_bank(db, course.id, caselets=True)
    assert report.caselets_active == 25
    # standalone bank is empty here → that failure is expected; caselet bar itself is green
    assert not any("caselet" in f.lower() for f in report.failures)
