"""T-26 — the caselet engine (017 §3).

A caselet is one client scenario (120-220 words) plus exactly five questions that
can be answered *only* by applying the scenario's facts. It is generated with
``prompts/caselet_gen.txt`` and put through the same gate chain as a standalone
question, per question, with two additions:

* **G4 also runs on the scenario text** (no verbatim lift of the narrative);
* **G6 — scenario dependence**: every question is blind-solved a second time
  *without* the scenario; any question the solver still answers *confidently
  correct* is scenario-independent — the caselet is regenerated, because a
  question answerable without the scenario is just an MCQ in a costume.

G3 blind-solves the scenario + all five questions together (how the exam presents
them). Accepted caselets are served only in caselet-drill and mock sessions
(:func:`caselet_drill`), never mixed into adaptive singles.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from pydantic import BaseModel, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from atlas.core import config
from atlas.core.embed import EmbedTransport
from atlas.core.llm import Transport, complete
from atlas.core.models import CaseGroup, Concept, Course, Question
from atlas.generate import gates
from atlas.generate.calibrate import initial_q_rating
from atlas.generate.pipeline import (
    STATUS_ACTIVE,
    STATUS_DISPUTED,
    GenerationReport,
    _embedder,
    _existing_stems,
    course_ngram_index,
)
from atlas.generate.prompts import (
    QUESTION_CONTRACT_SCHEMA,
    GeneratedQuestion,
    top_chunks,
)
from atlas.ingest.textutil import load_prompt, render_prompt

SCENARIO_MIN_WORDS = 120
SCENARIO_MAX_WORDS = 220
CASELET_QUESTIONS = 5
MAX_CASELET_ATTEMPTS = 3
DEFAULT_GROUP_SIZE = 4


class CaseletQuestion(GeneratedQuestion):
    """A standard question plus its position (1-5) and marks (1 or 2) in a caselet."""

    case_position: int
    marks: float

    @model_validator(mode="after")
    def _enforce_caselet_fields(self) -> CaseletQuestion:
        if self.case_position not in (1, 2, 3, 4, 5):
            raise ValueError("case_position must be 1-5")
        if float(self.marks) not in (1.0, 2.0):
            raise ValueError("marks must be 1 or 2")
        return self


class GeneratedCaselet(BaseModel):
    """A caselet contract: scenario + exactly five positioned questions."""

    scenario: str
    questions: list[CaseletQuestion]

    @model_validator(mode="after")
    def _enforce_caselet(self) -> GeneratedCaselet:
        words = len(self.scenario.split())
        if not SCENARIO_MIN_WORDS <= words <= SCENARIO_MAX_WORDS:
            raise ValueError(
                f"scenario must be {SCENARIO_MIN_WORDS}-{SCENARIO_MAX_WORDS} words, got {words}"
            )
        if len(self.questions) != CASELET_QUESTIONS:
            raise ValueError(f"exactly {CASELET_QUESTIONS} questions required")
        positions = sorted(q.case_position for q in self.questions)
        if positions != [1, 2, 3, 4, 5]:
            raise ValueError(f"case positions must be 1-5 distinct, got {positions}")
        return self


# ── contract schema for the prompt / strict-JSON retry ──────────────────────────
_CASELET_Q_SCHEMA = {
    "type": "object",
    "required": [*QUESTION_CONTRACT_SCHEMA["required"], "case_position", "marks"],
    "properties": {
        **QUESTION_CONTRACT_SCHEMA["properties"],
        "case_position": {"type": "integer"},
        "marks": {"type": "number"},
    },
}
CASELET_SCHEMA: dict = {
    "type": "object",
    "required": ["scenario", "questions"],
    "properties": {
        "scenario": {"type": "string"},
        "questions": {"type": "array", "items": _CASELET_Q_SCHEMA},
    },
}


# ── G6 (scenario dependence) + caselet G3 (combined blind solve) ────────────────
_CASELET_SOLVE_SYSTEM = (
    "You are an exam candidate answering a CASE STUDY: a scenario followed by five "
    "questions. Answer every question using only the scenario and your knowledge. "
    'Output STRICT JSON: {"answers":[{"position":1,"answer":"A|B|C|D",'
    '"confidence":"sure|unsure|guessing","rationale":"..."}]} with all five positions.'
)

_CASELET_SOLVE_SCHEMA: dict = {
    "type": "object",
    "required": ["answers"],
    "properties": {
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["position", "answer", "confidence"],
                "properties": {
                    "position": {"type": "integer"},
                    "answer": {"type": "string", "enum": ["A", "B", "C", "D"]},
                    "confidence": {"type": "string", "enum": ["sure", "unsure", "guessing"]},
                    "rationale": {"type": "string"},
                },
            },
        }
    },
}


def check_g6(
    questions: list[CaseletQuestion],
    *,
    model: str | None = None,
    transport: Transport | None = None,
    gate_log: dict,
) -> gates.GateResult:
    """Blind-solve each question WITHOUT the scenario; flag any answered confidently correct."""
    independent: list[int] = []
    for q in questions:
        solve = gates.blind_solve(
            stem=q.stem,
            options=q.option_dicts(),
            scenario=None,
            model=model or config.MODEL_SOLVER_BLIND,
            transport=transport,
        )
        if solve.answer == q.correct_key and solve.confidence == "sure":
            independent.append(q.case_position)
    passed = not independent
    reason = (
        "every question depends on the scenario"
        if passed
        else f"scenario-independent at positions {independent}"
    )
    return gates.GateResult(
        "G6", passed, reason, tags={"independent_positions": independent}
    ).record(gate_log)


def check_g3_caselet(
    scenario: str,
    questions: list[CaseletQuestion],
    *,
    consensus: int = 1,
    model: str | None = None,
    transport: Transport | None = None,
    gate_log: dict,
) -> gates.GateResult:
    """Blind-solve scenario + all five questions together; dispute on any disagreement."""
    body = [f"Scenario:\n{scenario}\n", "Questions:"]
    for q in questions:
        opts = "\n".join(f"  {o['key']}) {o['text']}" for o in q.option_dicts())
        body.append(f"{q.case_position}. {q.stem}\n{opts}")
    user = "\n".join(body)

    runs: list[dict[int, str]] = []
    for _ in range(max(1, consensus)):
        data = complete(
            model or config.MODEL_SOLVER_BLIND,
            _CASELET_SOLVE_SYSTEM,
            user,
            json_schema=_CASELET_SOLVE_SCHEMA,
            transport=transport,
        )
        assert isinstance(data, dict)
        runs.append(
            {int(a["position"]): str(a["answer"]).strip().upper() for a in data["answers"]}
        )

    key = {q.case_position: q.correct_key for q in questions}
    unanimous = all(run.get(pos) == key[pos] for run in runs for pos in key)
    votes = [[run.get(pos) for pos in sorted(key)] for run in runs]
    if unanimous:
        return gates.GateResult(
            "G3", True, "blind solver agrees on all five", tags={"disputed": False, "votes": votes}
        ).record(gate_log)
    return gates.GateResult(
        "G3",
        False,
        f"blind solver disagreement across the caselet, votes={votes} → disputed",
        tags={"disputed": True, "votes": votes},
    ).record(gate_log)


# ── generation ───────────────────────────────────────────────────────────────
def _concepts_block(concepts: list[Concept]) -> str:
    lines = []
    for c in concepts:
        misc = ", ".join(c.misconceptions or []) or "none recorded"
        lines.append(f"- {c.title}: {c.description or ''} (misconceptions: {misc})")
    return "\n".join(lines)


def _recent_scenarios(session: Session, course_id: uuid.UUID, limit: int = 15) -> str:
    rows = session.scalars(
        select(CaseGroup.scenario)
        .where(CaseGroup.course_id == course_id)
        .order_by(CaseGroup.created_at.desc())
        .limit(limit)
    ).all()
    return "\n".join(f"- {' '.join(s.split()[:20])}…" for s in rows) or "(none yet)"


def generate_caselet(
    session: Session,
    course: Course,
    concepts: list[Concept],
    marks_each: int,
    *,
    model: str | None = None,
    transport: Transport | None = None,
) -> GeneratedCaselet:
    """Render the caselet prompt and parse the response into the contract."""
    system, user_t = load_prompt("caselet_gen")
    chunk_texts: list[str] = []
    for c in concepts:
        chunk_texts += [ch.content for ch in top_chunks(session, c.id)]
    user = render_prompt(
        user_t,
        concepts_with_descriptions_and_misconceptions=_concepts_block(concepts),
        chunks="\n\n".join(chunk_texts) or "(no grounding available)",
        recent_scenario_summaries=_recent_scenarios(session, course.id),
        marks_each=marks_each,
    )
    data = complete(
        model or config.MODEL_GENERATOR,
        system,
        user,
        json_schema=CASELET_SCHEMA,
        transport=transport,
    )
    assert isinstance(data, dict)
    return GeneratedCaselet.model_validate(data)


def _select_concept_groups(
    session: Session,
    course_id: uuid.UUID,
    n_groups: int,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> list[list[Concept]]:
    """Group concepts into same/adjacent-chapter sets of 3-5 (017 §3 target selection)."""
    concepts = list(
        session.scalars(
            select(Concept)
            .where(Concept.course_id == course_id)
            .order_by(Concept.chapter_id, Concept.title)
        )
    )
    if not concepts:
        return []
    groups: list[list[Concept]] = []
    step = max(1, group_size)
    for i in range(n_groups):
        start = (i * step) % len(concepts)
        window = [concepts[(start + j) % len(concepts)] for j in range(group_size)]
        # de-dup while preserving order; ensure at least 3 distinct concepts
        seen: dict = {}
        for c in window:
            seen[c.id] = c
        group = list(seen.values())
        while len(group) < min(3, len(concepts)):
            extra = concepts[(start + len(group)) % len(concepts)]
            if extra.id not in seen:
                seen[extra.id] = extra
                group.append(extra)
            else:
                break
        groups.append(group)
    return groups


@dataclass
class CaseletReport(GenerationReport):
    """Generation report extended with the G6 regeneration count."""

    g6_regenerated: int = 0
    caselets: list = field(default_factory=list)


def _gate_caselet(
    caselet: GeneratedCaselet,
    *,
    ngram_index: set[str],
    chunk_matrix: np.ndarray,
    chunk_texts: list[str],
    existing: list[tuple[object, np.ndarray]],
    embed_fn,
    llm_transport: Transport | None,
    verifier_model: str | None,
    solver_model: str | None,
    consensus: int,
) -> tuple[dict, dict[int, dict], str, bool]:
    """Run scenario + per-question gates + G6 + G3.

    Returns ``(case_log, per_question_logs, status, g6_regen)`` where status is
    ``active``/``disputed``/``rejected`` and ``g6_regen`` is True when G6 failed
    (the caller should regenerate).
    """
    case_log: dict = {}
    scenario_vec = np.asarray(embed_fn([caselet.scenario])[0], dtype=np.float32)
    gates.check_g4(
        caselet.scenario, ngram_index, stem_vec=scenario_vec, chunk_matrix=chunk_matrix,
        gate_log=case_log,
    )
    if not case_log["G4"]["passed"]:
        return case_log, {}, "rejected", False

    per_q: dict[int, dict] = {}
    for q in caselet.questions:
        log: dict = {}
        options = q.option_dicts()
        if not gates.check_g1(
            stem=q.stem, options=options, correct_key=q.correct_key, embed_fn=embed_fn, gate_log=log
        ).passed:
            return case_log, {q.case_position: log}, "rejected", False
        stem_vec = np.asarray(embed_fn([q.stem])[0], dtype=np.float32)
        if not gates.check_g4(
            q.stem, ngram_index, stem_vec=stem_vec, chunk_matrix=chunk_matrix, gate_log=log
        ).passed:
            return case_log, {q.case_position: log}, "rejected", False
        if not gates.check_g5(stem_vec, existing, gate_log=log).passed:
            return case_log, {q.case_position: log}, "rejected", False
        if not gates.check_g2(
            stem=q.stem,
            options=options,
            correct_key=q.correct_key,
            chunk_texts=chunk_texts,
            model=verifier_model,
            transport=llm_transport,
            gate_log=log,
        ).passed:
            return case_log, {q.case_position: log}, "rejected", False
        per_q[q.case_position] = log

    # G6 — scenario dependence (drives regeneration on failure).
    if not check_g6(
        caselet.questions, model=solver_model, transport=llm_transport, gate_log=case_log
    ).passed:
        return case_log, per_q, "rejected", True

    # G3 — combined blind solve → active or disputed.
    g3 = check_g3_caselet(
        caselet.scenario,
        caselet.questions,
        consensus=consensus,
        model=solver_model,
        transport=llm_transport,
        gate_log=case_log,
    )
    status = STATUS_ACTIVE if g3.passed else STATUS_DISPUTED
    for log in per_q.values():
        log["G3"] = case_log["G3"]
    return case_log, per_q, status, False


def _persist_caselet(
    session: Session,
    course: Course,
    concepts: list[Concept],
    caselet: GeneratedCaselet,
    case_log: dict,
    per_q: dict[int, dict],
    status: str,
    generator_model: str | None,
) -> CaseGroup:
    cg = CaseGroup(
        course_id=course.id,
        scenario=caselet.scenario,
        # stringified ids: ARRAY(Uuid()) accepts them on Postgres and the SQLite
        # JSON fallback (UuidArray) can only carry JSON-serializable values.
        concept_ids=[str(c.id) for c in concepts],
        status=status,
        gate_log=case_log,
        generator_model=generator_model or config.MODEL_GENERATOR,
    )
    session.add(cg)
    session.flush()
    for i, q in enumerate(sorted(caselet.questions, key=lambda x: x.case_position)):
        concept = concepts[i % len(concepts)]
        session.add(
            Question(
                course_id=course.id,
                concept_id=concept.id,
                case_group_id=cg.id,
                stem=q.stem,
                options=q.option_dicts(),
                correct_key=q.correct_key,
                explanation=q.explanation,
                difficulty=q.difficulty,
                q_rating=initial_q_rating(q.difficulty),
                status=status,
                gate_log=per_q.get(q.case_position, {}),
                generator_model=generator_model,
                marks=q.marks,
                case_position=q.case_position,
            )
        )
    session.flush()
    return cg


def run_caselets(
    session: Session,
    course: Course,
    user_id: str,
    *,
    count: int,
    two_mark_count: int = 0,
    refill: bool = False,
    consensus: int = 1,
    generator_model: str | None = None,
    verifier_model: str | None = None,
    solver_model: str | None = None,
    llm_transport: Transport | None = None,
    embed_transport: EmbedTransport | None = None,
    cache_dir: str | Path | None = None,
) -> CaseletReport:
    """Generate, gate (incl. G6) and persist ``count`` caselets for the course."""
    report = CaseletReport()
    embed_fn = _embedder(embed_transport, cache_dir)
    ngram_index = course_ngram_index(session, course.id)

    existing_pairs = _existing_stems(session, course.id)
    existing: list[tuple[object, np.ndarray]] = []
    if existing_pairs:
        vecs = embed_fn([s for _, s in existing_pairs])
        existing = [(qid, np.asarray(vecs[i], dtype=np.float32)) for i, (qid, _) in enumerate(existing_pairs)]

    groups = _select_concept_groups(session, course.id, count)
    for idx in range(count):
        concepts = groups[idx] if idx < len(groups) else []
        if not concepts:
            continue
        marks_each = 2 if idx < two_mark_count else 1

        chunk_texts: list[str] = []
        for c in concepts:
            chunk_texts += [ch.content for ch in top_chunks(session, c.id)]
        chunk_matrix = (
            np.asarray(embed_fn(chunk_texts), dtype=np.float32)
            if chunk_texts
            else np.empty((0, 0), dtype=np.float32)
        )

        accepted = False
        rejected_hard = False
        for _ in range(MAX_CASELET_ATTEMPTS):
            report.generated += 1
            caselet = generate_caselet(
                session,
                course,
                concepts,
                marks_each,
                model=generator_model,
                transport=llm_transport,
            )
            case_log, per_q, status, g6_regen = _gate_caselet(
                caselet,
                ngram_index=ngram_index,
                chunk_matrix=chunk_matrix,
                chunk_texts=chunk_texts,
                existing=existing,
                embed_fn=embed_fn,
                llm_transport=llm_transport,
                verifier_model=verifier_model,
                solver_model=solver_model,
                consensus=consensus,
            )
            if g6_regen:
                report.g6_regenerated += 1
                continue
            if status == "rejected":
                failed = next(
                    (g for g in ("G4", "G1", "G5", "G2") if not _passed(case_log, per_q, g)), "G?"
                )
                report._reject(failed)
                rejected_hard = True
                break
            cg = _persist_caselet(
                session, course, concepts, caselet, case_log, per_q, status, generator_model
            )
            for q in caselet.questions:
                existing.append((q.stem, np.asarray(embed_fn([q.stem])[0], dtype=np.float32)))
            report.caselets.append(cg.id)
            if status == STATUS_ACTIVE:
                report.accepted += 1
            else:
                report.disputed += 1
            accepted = True
            break
        if not accepted and not rejected_hard:
            # exhausted regeneration attempts without a G6-clean caselet
            report.rejected += 1

    session.flush()
    return report


def _passed(case_log: dict, per_q: dict[int, dict], gate: str) -> bool:
    """True if ``gate`` passed anywhere it was recorded (scenario log or any question)."""
    logs = [case_log, *per_q.values()]
    return all(log.get(gate, {}).get("passed", True) for log in logs)


def caselet_drill(
    session: Session, course_id: uuid.UUID, limit: int | None = None
) -> list[tuple[CaseGroup, list[Question]]]:
    """Active caselets with their five questions (position-ordered) for drill/mock use."""
    stmt = select(CaseGroup).where(
        CaseGroup.course_id == course_id, CaseGroup.status == "active"
    ).order_by(CaseGroup.created_at)
    if limit is not None:
        stmt = stmt.limit(limit)
    out: list[tuple[CaseGroup, list[Question]]] = []
    for cg in session.scalars(stmt):
        qs = list(
            session.scalars(
                select(Question)
                .where(Question.case_group_id == cg.id)
                .order_by(Question.case_position)
            )
        )
        out.append((cg, qs))
    return out
