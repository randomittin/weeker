"""``weeker status`` — the learner dashboard (016 §T-41, 017 §6).

:func:`build_status` assembles everything the evening review needs from the live
tables; :func:`render_status` turns it into deterministic text (the golden-output
gate):

* **heatmap** — per-chapter mean mastery with each concept's mastery, attempts and
  days-until-due;
* **due counts** — concepts past their retention due date + flashcards due;
* **flags** — concepts carrying an open ``misconception_pending``;
* **prediction band** — whole-bank expected score / CI / ``P(pass)`` (suppressed
  when coverage is too thin), from :func:`weeker.learn.predict.predict_from_db`;
* **tonight's refill plan** — the weak-concept refill quotas from
  :func:`weeker.generate.targets.compute_targets`;
* **realized calibration** — empirical accuracy per confidence keystroke, shown
  only once the learner has ≥ 3 days of attempts (017 §6, "from day 3").
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from weeker.core.models import Chapter, Concept, Course, Flashcard, Mastery
from weeker.generate.targets import compute_targets
from weeker.learn import elo, predict, retention

# Days of attempt history before the realized-calibration table is shown (017 §6).
CALIBRATION_MIN_DAYS = 3
DEFAULT_REFILL_COUNT = 40


def _aware(dt: datetime | None) -> datetime | None:
    """Treat a persisted timestamp as UTC when the backend dropped the tzinfo (SQLite)."""
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


@dataclass
class ConceptCell:
    """One heatmap cell: a concept's mastery for this learner."""

    concept_id: object
    title: str
    theta: float
    mastery: float
    attempts: int
    due_in_days: float | None
    misconception: bool


@dataclass
class ChapterRow:
    """A heatmap row: a chapter's concepts and their mean mastery."""

    chapter_id: object
    title: str
    mean_mastery: float
    cells: list[ConceptCell] = field(default_factory=list)


@dataclass
class RefillEntry:
    """One line of tonight's refill plan."""

    concept_id: object
    title: str
    quota: int


@dataclass
class CalibrationRow:
    """Empirical accuracy for one confidence keystroke."""

    confidence: str
    n: int
    correct: int
    rate: float
    recommendation: str


@dataclass
class StatusReport:
    """Everything ``weeker status`` prints for one learner/course."""

    course_title: str
    chapters: list[ChapterRow]
    due_concepts: int
    due_flashcards: int
    misconception_flags: int
    prediction: predict.Prediction
    refill_plan: list[RefillEntry]
    calibration_ready: bool
    calibration: list[CalibrationRow]


def _distinct_attempt_days(db: Session, user_id: str) -> int:
    """Number of distinct calendar days this learner has attempts on."""
    from weeker.core.models import Attempt

    days = {
        (created.date() if hasattr(created, "date") else created)
        for (created,) in db.execute(
            select(Attempt.created_at).where(Attempt.user_id == user_id)
        ).all()
    }
    return len(days)


def realized_calibration(
    db: Session, user_id: str, *, neg_fraction: float = predict.DEFAULT_NEG_FRACTION
) -> list[CalibrationRow]:
    """Accuracy per confidence level with an attempt/skip recommendation vs break-even."""
    from weeker.core.models import Attempt

    breakeven = predict.breakeven_p(neg_fraction)
    agg: dict = {}
    for conf, correct in db.execute(
        select(Attempt.confidence, Attempt.correct).where(Attempt.user_id == user_id)
    ).all():
        d = agg.setdefault(conf, [0, 0])
        d[0] += 1
        d[1] += 1 if correct else 0
    rows: list[CalibrationRow] = []
    for conf in ("sure", "unsure", "guessing"):
        if conf not in agg:
            continue
        n, c = agg[conf]
        rate = c / n if n else 0.0
        rec = (
            "always attempt"
            if rate > breakeven
            else "attempt only if you can eliminate an option"
        )
        rows.append(CalibrationRow(conf, n, c, rate, rec))
    return rows


def build_status(
    db: Session,
    *,
    user_id: str,
    course: Course,
    now: datetime,
    refill_count: int = DEFAULT_REFILL_COUNT,
    rng: random.Random | None = None,
) -> StatusReport:
    """Assemble the full status report from the live tables."""
    concepts = list(
        db.scalars(
            select(Concept).where(
                Concept.course_id == course.id, Concept.review_status != "dropped"
            )
        ).all()
    )
    chapters = {
        c.id: c.title
        for c in db.scalars(select(Chapter).where(Chapter.course_id == course.id)).all()
    }
    mastery_by_concept = {
        m.concept_id: m
        for m in db.scalars(select(Mastery).where(Mastery.user_id == user_id)).all()
    }
    from weeker.core.models import Attempt

    attempt_counts = dict(
        db.execute(
            select(Attempt.concept_id, func.count())
            .where(Attempt.user_id == user_id)
            .group_by(Attempt.concept_id)
        ).all()
    )

    rows_by_chapter: dict = {}
    for c in concepts:
        m = mastery_by_concept.get(c.id)
        theta = float(m.theta) if m is not None else 0.0
        due_in = None
        if m is not None and m.last_seen is not None:
            days_since = max(0.0, (now - _aware(m.last_seen)).total_seconds() / 86400.0)
            due_in = retention.days_until_due(float(m.stability)) - days_since
        cell = ConceptCell(
            concept_id=c.id,
            title=c.title,
            theta=theta,
            mastery=elo.mastery(theta),
            attempts=int(attempt_counts.get(c.id, 0)),
            due_in_days=due_in,
            misconception=bool(m.misconception_pending) if m is not None else False,
        )
        rows_by_chapter.setdefault(c.chapter_id, []).append(cell)

    chapter_rows: list[ChapterRow] = []
    for chapter_id, cells in rows_by_chapter.items():
        cells.sort(key=lambda x: x.title)
        mean = sum(x.mastery for x in cells) / len(cells) if cells else 0.0
        chapter_rows.append(
            ChapterRow(
                chapter_id=chapter_id,
                title=chapters.get(chapter_id, "(unassigned)"),
                mean_mastery=mean,
                cells=cells,
            )
        )
    chapter_rows.sort(key=lambda r: r.title)

    due_concepts = sum(
        1
        for m in mastery_by_concept.values()
        if m.due_at is not None and _aware(m.due_at) <= now
    )
    due_flashcards = (
        db.scalar(
            select(func.count())
            .select_from(Flashcard)
            .where(
                Flashcard.user_id == user_id,
                Flashcard.due_at.is_not(None),
                Flashcard.due_at <= now,
            )
        )
        or 0
    )
    misconception_flags = sum(
        1 for m in mastery_by_concept.values() if m.misconception_pending
    )

    params = _exam_params(course)
    prediction = predict.predict_from_db(
        db,
        user_id=user_id,
        course_id=course.id,
        passing_marks=params["passing_marks"],
        total_marks=params["total_marks"],
        neg_fraction=params["neg_fraction"],
        rng=rng,
    )

    title_by_concept = {c.id: c.title for c in concepts}
    refill_plan = [
        RefillEntry(
            concept_id=t.concept_id,
            title=title_by_concept.get(t.concept_id, str(t.concept_id)),
            quota=t.quota,
        )
        for t in compute_targets(db, course.id, user_id, refill_count, refill=True)
        if t.quota > 0
    ]
    refill_plan.sort(key=lambda e: (-e.quota, e.title))

    calibration_ready = _distinct_attempt_days(db, user_id) >= CALIBRATION_MIN_DAYS
    calibration = (
        realized_calibration(db, user_id, neg_fraction=params["neg_fraction"])
        if calibration_ready
        else []
    )

    return StatusReport(
        course_title=course.title,
        chapters=chapter_rows,
        due_concepts=due_concepts,
        due_flashcards=int(due_flashcards),
        misconception_flags=misconception_flags,
        prediction=prediction,
        refill_plan=refill_plan,
        calibration_ready=calibration_ready,
        calibration=calibration,
    )


def _exam_params(course: Course) -> dict:
    cfg = course.exam_config or {}
    exam = cfg.get("exam", cfg) if isinstance(cfg, dict) else {}
    return {
        "passing_marks": float(exam.get("passing_marks", 90)),
        "total_marks": float(exam.get("total_marks", 150)),
        "neg_fraction": float(exam.get("negative_marking_fraction", 0.25)),
    }


def render_status(report: StatusReport) -> str:
    """Deterministic text rendering of a :class:`StatusReport` (the golden-output gate)."""
    lines: list[str] = [f"Status — {report.course_title}", ""]
    lines.append("Heatmap (mastery by chapter/concept):")
    for row in report.chapters:
        lines.append(f"  {row.title}  [mean {row.mean_mastery:.2f}]")
        for cell in row.cells:
            due = "never" if cell.due_in_days is None else f"{cell.due_in_days:+.1f}d"
            flag = " !misconception" if cell.misconception else ""
            lines.append(
                f"    - {cell.title}: mastery {cell.mastery:.2f} "
                f"(theta {cell.theta:+.2f}, {cell.attempts} attempts, due {due}){flag}"
            )
    lines.append("")
    lines.append(
        f"Due: {report.due_concepts} concepts, {report.due_flashcards} flashcards | "
        f"Misconception flags: {report.misconception_flags}"
    )
    lines.append("")

    p = report.prediction
    if p.suppressed:
        lines.append(
            f"Prediction: suppressed (coverage {p.coverage:.0%} < threshold — practise more)"
        )
    else:
        lines.append(
            f"Prediction: {p.expected_marks:.1f}/{p.total_marks:.0f} marks "
            f"(95% CI {p.ci_low:.1f}-{p.ci_high:.1f}), "
            f"P(pass at {p.passing_marks:.0f}) = {p.p_pass:.0%}"
        )
    lines.append("")

    lines.append("Tonight's refill plan (weak-concept quotas):")
    if report.refill_plan:
        for e in report.refill_plan:
            lines.append(f"  - {e.title}: {e.quota}")
    else:
        lines.append("  (nothing to refill)")
    lines.append("")

    lines.append("Realized calibration:")
    if not report.calibration_ready:
        lines.append(f"  (available from day {CALIBRATION_MIN_DAYS})")
    elif not report.calibration:
        lines.append("  (no attempts recorded yet)")
    else:
        for c in report.calibration:
            lines.append(
                f"  - {c.confidence}: {c.correct}/{c.n} = {c.rate:.0%} -> {c.recommendation}"
            )
    return "\n".join(lines)
