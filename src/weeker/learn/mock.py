"""Timed mock exam engine — scoring, pause/resume, review order (016 §T-40, 017 §4/§6).

A mock is a fixed ordered list of :class:`MockItem` (question id + marks + keyed
answer + section). The learner answers each with an optional confidence; nothing
is graded until submit. This module is the **pure engine** the ``weeker mock``
TUIs wrap:

* :func:`score_item` / :func:`grade` — marks-weighted scoring with a per-question
  negative deduction of ``neg_fraction × marks`` (0.25 × marks by default, 017 §1);
* :func:`compose_standalone_items` — a blueprint-stratified 90-question standalone
  section, reusing the committed :mod:`weeker.learn.scheduler` mock composer;
* :func:`start_mock` / :func:`save_progress` / :func:`resume_mock` — a
  :class:`~weeker.core.models.MockExam` row whose ``config`` jsonb persists the item
  plan + answers so a paused exam resumes exactly where it stopped;
* :func:`submit_mock` — grades, records one graded :func:`record_attempt` per
  answered item (so mastery + realized calibration update from the mock), writes
  the score/pass onto the row, and returns the post-mock review ordered
  confident-wrong first with caselet questions grouped.

The prediction snapshot (:func:`snapshot_prediction`) is stored on
``MockExam.predicted_before`` at start, per 016 §T-40.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from weeker.core.models import (
    BlueprintWeight,
    Chapter,
    Course,
    MockExam,
    Question,
    SessionRun,
)
from weeker.learn import predict, scheduler
from weeker.learn.attempt_service import record_attempt

DEFAULT_NEG_FRACTION = 0.25
STANDALONE = "standalone"
CASELET = "caselet"


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(UTC)


# ── exam parameters (from the manifest exam block, mirrored to exam_config) ─────
def exam_params(course: Course) -> dict:
    """Read passing/total/duration/neg from ``course.exam_config`` with 017 §1 defaults."""
    cfg = course.exam_config or {}
    exam = cfg.get("exam", cfg) if isinstance(cfg, dict) else {}
    return {
        "passing_marks": float(exam.get("passing_marks", 90)),
        "total_marks": float(exam.get("total_marks", 150)),
        "duration_minutes": float(exam.get("duration_minutes", 180)),
        "neg_fraction": float(exam.get("negative_marking_fraction", DEFAULT_NEG_FRACTION)),
    }


# ── item / answer value objects ────────────────────────────────────────────────
@dataclass
class MockItem:
    """One servable exam question: its keyed answer, marks and section."""

    question_id: object
    concept_id: object
    marks: float
    correct_key: str
    section: str = STANDALONE
    case_group_id: object | None = None
    case_position: int | None = None
    difficulty: int = 2


@dataclass
class MockAnswer:
    """A learner's response to one item (``chosen_key=None`` → skipped)."""

    chosen_key: str | None = None
    confidence: str | None = None  # "sure" | "unsure" | "guessing"


def item_from_question(q: Question, section: str = STANDALONE) -> MockItem:
    """Build a :class:`MockItem` from an ORM :class:`~weeker.core.models.Question`."""
    return MockItem(
        question_id=q.id,
        concept_id=q.concept_id,
        marks=float(q.marks),
        correct_key=q.correct_key,
        section=(CASELET if q.case_group_id is not None else section),
        case_group_id=q.case_group_id,
        case_position=q.case_position,
        difficulty=int(q.difficulty),
    )


# ── scoring ────────────────────────────────────────────────────────────────────
def score_item(
    item: MockItem, answer: MockAnswer | None, neg_fraction: float = DEFAULT_NEG_FRACTION
) -> float:
    """Marks for one item: ``+marks`` if right, ``-neg_fraction×marks`` if wrong, ``0`` if skipped."""
    if answer is None or answer.chosen_key is None:
        return 0.0
    if answer.chosen_key == item.correct_key:
        return float(item.marks)
    return -neg_fraction * float(item.marks)


@dataclass
class GradeResult:
    """Marks-weighted grade of a whole mock."""

    raw_score: float
    total_marks: float
    per_item: dict


def grade(
    items: list[MockItem],
    answers: dict,
    neg_fraction: float = DEFAULT_NEG_FRACTION,
) -> GradeResult:
    """Score every item; ``answers`` maps ``question_id → MockAnswer``."""
    raw = 0.0
    total = 0.0
    per_item: dict = {}
    for it in items:
        total += float(it.marks)
        s = score_item(it, answers.get(it.question_id), neg_fraction)
        per_item[it.question_id] = s
        raw += s
    return GradeResult(raw_score=raw, total_marks=total, per_item=per_item)


# ── post-mock review ordering (confident-wrong first, caselets grouped) ─────────
def _item_rank(item: MockItem, answer: MockAnswer | None) -> int:
    """0 confident-wrong · 1 other wrong · 2 skipped · 3 correct."""
    if answer is not None and answer.chosen_key is not None:
        correct = answer.chosen_key == item.correct_key
        if not correct and answer.confidence == "sure":
            return 0
        if not correct:
            return 1
        return 3
    return 2


def review_order(items: list[MockItem], answers: dict) -> list[MockItem]:
    """Order for review: confident-wrong first, then wrong, skipped, correct.

    Caselet questions stay grouped (the scenario context is the point) and a group
    floats to the position of its worst-reviewed question; within a group,
    questions keep exam order (``case_position``).
    """
    groups: dict = {}
    first_seen: dict = {}
    for idx, it in enumerate(items):
        gkey = it.case_group_id if it.case_group_id is not None else it.question_id
        groups.setdefault(gkey, []).append(it)
        first_seen.setdefault(gkey, idx)
    group_rank = {
        g: min(_item_rank(it, answers.get(it.question_id)) for it in members)
        for g, members in groups.items()
    }
    ordered = sorted(groups, key=lambda g: (group_rank[g], first_seen[g]))
    out: list[MockItem] = []
    for g in ordered:
        out.extend(sorted(groups[g], key=lambda it: (it.case_position or 0)))
    return out


# ── standalone composition (blueprint-stratified, reuses scheduler) ─────────────
def blueprint_by_chapter(db: Session, course_id: object) -> dict:
    """Map ``chapter_id → weight`` from the blueprint (chapter titles), default 1.0."""
    title_by_id = dict(
        db.execute(select(Chapter.id, Chapter.title).where(Chapter.course_id == course_id)).all()
    )
    weight_by_title = dict(
        db.execute(
            select(BlueprintWeight.chapter, BlueprintWeight.weight).where(
                BlueprintWeight.course_id == course_id
            )
        ).all()
    )
    return {cid: float(weight_by_title.get(title, 0.0)) or 1.0 for cid, title in title_by_id.items()}


def compose_standalone_items(
    db: Session,
    user_id: str,
    course: Course,
    now: datetime,
    rng: random.Random,
    total: int = 90,
) -> list[MockItem]:
    """Compose the ``total``-question standalone section as ordered :class:`MockItem`."""
    states = scheduler.load_concept_states(db, user_id, course.id, now)
    qstates = scheduler.load_question_states(
        db, user_id, course.id, [s.concept_id for s in states], now
    )
    blueprint = blueprint_by_chapter(db, course.id)
    composed = scheduler.compose_mock(states, qstates, blueprint, rng, total=total)
    ids = [q.question_id for q in composed]
    by_id = {q.id: q for q in db.scalars(select(Question).where(Question.id.in_(ids))).all()}
    return [item_from_question(by_id[qid]) for qid in ids if qid in by_id]


# ── prediction snapshot ─────────────────────────────────────────────────────────
def snapshot_prediction(
    db: Session,
    user_id: str,
    course: Course,
    *,
    passing_marks: float,
    total_marks: float,
    neg_fraction: float = DEFAULT_NEG_FRACTION,
    rng: random.Random | None = None,
) -> dict:
    """Whole-bank score prediction as a jsonb-safe dict for ``predicted_before``."""
    p = predict.predict_from_db(
        db,
        user_id=user_id,
        course_id=course.id,
        passing_marks=passing_marks,
        total_marks=total_marks,
        neg_fraction=neg_fraction,
        rng=rng,
    )
    return {
        "expected_marks": p.expected_marks,
        "expected_fraction": p.expected_fraction,
        "ci_low": p.ci_low,
        "ci_high": p.ci_high,
        "p_pass": p.p_pass,
        "coverage": p.coverage,
        "suppressed": p.suppressed,
        "total_marks": p.total_marks,
        "passing_marks": p.passing_marks,
    }


# ── pause/resume persistence ─────────────────────────────────────────────────────
def _items_to_json(items: list[MockItem]) -> list[dict]:
    return [
        {
            "q": str(it.question_id),
            "c": str(it.concept_id),
            "m": float(it.marks),
            "k": it.correct_key,
            "s": it.section,
            "cg": str(it.case_group_id) if it.case_group_id is not None else None,
            "cp": it.case_position,
            "d": it.difficulty,
        }
        for it in items
    ]


def _items_from_json(rows: list[dict]) -> list[MockItem]:
    out = []
    for r in rows:
        out.append(
            MockItem(
                question_id=uuid.UUID(r["q"]),
                concept_id=uuid.UUID(r["c"]),
                marks=float(r["m"]),
                correct_key=r["k"],
                section=r.get("s", STANDALONE),
                case_group_id=uuid.UUID(r["cg"]) if r.get("cg") else None,
                case_position=r.get("cp"),
                difficulty=int(r.get("d", 2)),
            )
        )
    return out


def _answers_to_json(answers: dict) -> dict:
    return {
        str(qid): {"chosen_key": a.chosen_key, "confidence": a.confidence}
        for qid, a in answers.items()
    }


def _answers_from_json(raw: dict) -> dict:
    return {
        uuid.UUID(qid): MockAnswer(chosen_key=v.get("chosen_key"), confidence=v.get("confidence"))
        for qid, v in (raw or {}).items()
    }


def start_mock(
    db: Session,
    *,
    user_id: str,
    course: Course,
    items: list[MockItem],
    predicted_before: dict,
    duration_minutes: float,
    kind: str = "mock",
    now: datetime | None = None,
) -> MockExam:
    """Open a mock: create its :class:`SessionRun` + :class:`MockExam` with the item plan persisted."""
    now = _now(now)
    session = SessionRun(
        user_id=user_id, course_id=course.id, kind="mock", started_at=now, config={"pattern": kind}
    )
    db.add(session)
    db.flush()
    mock = MockExam(
        user_id=user_id,
        course_id=course.id,
        session_id=session.id,
        predicted_before=predicted_before,
        started_at=now,
        config={
            "pattern": kind,
            "duration_minutes": duration_minutes,
            "items": _items_to_json(items),
            "answers": {},
            "index": 0,
            "elapsed_seconds": 0.0,
            "submitted": False,
        },
    )
    db.add(mock)
    db.flush()
    return mock


def save_progress(
    db: Session,
    mock: MockExam,
    *,
    answers: dict,
    index: int,
    elapsed_seconds: float,
) -> None:
    """Persist in-progress answers/position/elapsed onto the mock's ``config`` jsonb."""
    cfg = dict(mock.config or {})
    cfg["answers"] = _answers_to_json(answers)
    cfg["index"] = int(index)
    cfg["elapsed_seconds"] = float(elapsed_seconds)
    mock.config = cfg
    db.add(mock)
    db.flush()


@dataclass
class ResumedMock:
    """A paused mock reloaded from its persisted plan."""

    mock: MockExam
    items: list[MockItem]
    answers: dict
    index: int
    elapsed_seconds: float


def resume_mock(db: Session, mock_id: object) -> ResumedMock:
    """Reload a paused mock's item plan, answers and position from ``config``."""
    mock = db.get(MockExam, mock_id)
    if mock is None:
        raise ValueError(f"mock {mock_id} not found")
    cfg = mock.config or {}
    return ResumedMock(
        mock=mock,
        items=_items_from_json(cfg.get("items", [])),
        answers=_answers_from_json(cfg.get("answers", {})),
        index=int(cfg.get("index", 0)),
        elapsed_seconds=float(cfg.get("elapsed_seconds", 0.0)),
    )


def latest_open_mock(db: Session, user_id: str, course_id: object) -> MockExam | None:
    """The most recent un-submitted mock for this learner/course, if any."""
    return db.scalars(
        select(MockExam)
        .where(
            MockExam.user_id == user_id,
            MockExam.course_id == course_id,
            MockExam.submitted_at.is_(None),
        )
        .order_by(MockExam.started_at.desc())
    ).first()


# ── submission ───────────────────────────────────────────────────────────────────
@dataclass
class MockResult:
    """Outcome of a submitted mock: score, pass, and the review order."""

    mock_id: object
    raw_score: float
    total_marks: float
    passed: bool
    passing_marks: float
    answered: int
    skipped: int
    review: list[MockItem] = field(default_factory=list)


def submit_mock(
    db: Session,
    *,
    mock: MockExam,
    items: list[MockItem],
    answers: dict,
    passing_marks: float,
    neg_fraction: float = DEFAULT_NEG_FRACTION,
    record_attempts: bool = True,
    now: datetime | None = None,
) -> MockResult:
    """Grade, record graded attempts, finalize the row, and return the review order."""
    now = _now(now)
    g = grade(items, answers, neg_fraction)
    answered = 0
    if record_attempts:
        for it in items:
            a = answers.get(it.question_id)
            if a is None or a.chosen_key is None:
                continue
            answered += 1
            record_attempt(
                user_id=mock.user_id,
                question_id=it.question_id,
                chosen_key=a.chosen_key,
                confidence=a.confidence or "guessing",
                session_id=mock.session_id,
                now=now,
                db=db,
            )
    else:
        answered = sum(
            1 for it in items if (a := answers.get(it.question_id)) and a.chosen_key is not None
        )

    passed = g.raw_score >= passing_marks
    mock.raw_score = g.raw_score
    mock.total_marks = g.total_marks
    mock.passed = passed
    mock.submitted_at = now
    cfg = dict(mock.config or {})
    cfg["answers"] = _answers_to_json(answers)
    cfg["submitted"] = True
    mock.config = cfg
    db.add(mock)
    db.flush()

    return MockResult(
        mock_id=mock.id,
        raw_score=g.raw_score,
        total_marks=g.total_marks,
        passed=passed,
        passing_marks=passing_marks,
        answered=answered,
        skipped=len(items) - answered,
        review=review_order(items, answers),
    )
