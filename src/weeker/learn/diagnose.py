"""Day-0 diagnostic — theta seeding from a silent measurement (016 §T-36, 017 §5).

``weeker diagnose`` runs once before the first study session: 36 standalone
questions (3 per chapter × 12 chapters, one per authored difficulty tier),
chosen for maximum concept spread by a **greedy max-coverage** over concept ids.
No feedback during the run — it is measurement, not practice — then a full review.

Seeding (017 §5), applied here rather than through the normal attempt path so the
cold-start learning rate is a fixed ``K = 0.5`` (not the attempt-decayed K):

* **tested concept** — run the Elo update from ``theta = 0`` with ``K`` pinned at
  :data:`DIAG_K` over that concept's diagnostic answers;
* **untested chapter-mate** — seed ``theta = DIAG_SHRINK × mean(theta of tested
  chapter-mates)`` (shrunk toward 0);
* **confident-wrong** sets ``misconception_pending`` on the concept, exactly as the
  normal flow does.

Reseeding is idempotent: a second run refuses unless ``force=True`` (``--force``),
so the day-0 measurement is never silently overwritten. Output is the first real
``weeker status`` heatmap + tonight's refill plan.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from weeker.core import config
from weeker.core.models import (
    Attempt,
    Chapter,
    Concept,
    Course,
    Mastery,
    Question,
    SessionRun,
)
from weeker.learn import elo, status
from weeker.learn.elo import _clamp

# Diagnostic-specific structural constants (017 §5). Pinned here, not in config:
# they define this measurement procedure, not an adaptive tunable the gates read
# (same rationale as elo.THETA_ABS_MAX).
DIAG_K = 0.5  # fixed cold-start Elo learning rate for the diagnostic seeding
DIAG_SHRINK = 0.6  # shrink factor for untested chapter-mates toward theta=0
PER_CHAPTER = 3  # questions per chapter (one per difficulty tier)
DIFFICULTY_TIERS = (1, 2, 3)
THETA_ABS_MAX = 6.0


class DiagnosticAlreadyRun(RuntimeError):
    """Raised when a diagnostic already exists for this learner/course and ``force`` is off."""


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(UTC)


def diagnostic_exists(db: Session, user_id: str, course_id: object) -> bool:
    """True when a diagnostic :class:`SessionRun` already exists for this learner/course."""
    n = db.scalar(
        select(func.count())
        .select_from(SessionRun)
        .where(
            SessionRun.user_id == user_id,
            SessionRun.course_id == course_id,
            SessionRun.kind == "diagnostic",
        )
    )
    return bool(n)


# ── selection (greedy max-coverage over concept ids) ────────────────────────────
def select_diagnostic_questions(
    db: Session,
    course_id: object,
    rng: random.Random,
    *,
    per_chapter: int = PER_CHAPTER,
) -> list[Question]:
    """Pick ``per_chapter`` questions per chapter (one per difficulty tier), spread over concepts.

    Greedy: within each chapter/tier prefer a question whose concept is not yet
    covered, so the 36-question set touches as many distinct concepts as possible.
    """
    chapters = list(
        db.scalars(
            select(Chapter).where(Chapter.course_id == course_id).order_by(Chapter.ordinal)
        ).all()
    )
    questions = list(
        db.scalars(
            select(Question).where(
                Question.course_id == course_id,
                Question.status == "active",
                Question.case_group_id.is_(None),
            )
        ).all()
    )
    concept_chapter = dict(
        db.execute(select(Concept.id, Concept.chapter_id).where(Concept.course_id == course_id)).all()
    )

    by_chapter: dict = {}
    for q in questions:
        ch = concept_chapter.get(q.concept_id)
        by_chapter.setdefault(ch, []).append(q)

    covered: set = set()
    selected: list[Question] = []
    for ch in chapters:
        pool = list(by_chapter.get(ch.id, []))
        if not pool:
            continue
        chapter_used: set = set()
        by_diff: dict = {}
        for q in pool:
            by_diff.setdefault(int(q.difficulty), []).append(q)
        for tier in DIFFICULTY_TIERS[:per_chapter]:
            candidates = [q for q in by_diff.get(tier, []) if q.id not in chapter_used]
            if not candidates:  # tier empty → fall back to any unused question in the chapter
                candidates = [q for q in pool if q.id not in chapter_used]
            if not candidates:
                continue
            uncovered = [q for q in candidates if q.concept_id not in covered]
            choice = rng.choice(uncovered) if uncovered else rng.choice(candidates)
            selected.append(choice)
            covered.add(choice.concept_id)
            chapter_used.add(choice.id)
    return selected


# ── seeding ──────────────────────────────────────────────────────────────────────
@dataclass
class Graded:
    """One diagnostic answer: the question plus the learner's chosen key and confidence."""

    question: Question
    chosen_key: str | None
    confidence: str


@dataclass
class DiagnosticResult:
    """What the diagnostic seeded, plus the first status snapshot."""

    session_id: object
    n_questions: int
    tested: dict = field(default_factory=dict)  # concept_id → seeded theta
    seeded_untested: dict = field(default_factory=dict)  # concept_id → seeded theta
    misconceptions: set = field(default_factory=set)  # concept_ids flagged confident-wrong
    status: status.StatusReport | None = None


def _seed_theta_for_concept(attempts: list[tuple[float, bool, str]]) -> float:
    """Fixed-K (=DIAG_K) Elo from theta=0 over one concept's (q_rating, correct, confidence)."""
    theta = 0.0
    for q_rating, correct, confidence in attempts:
        exp = elo.expected(theta, q_rating)
        actual = 1.0 if correct else 0.0
        outcome = "correct" if correct else "wrong"
        conf_mult = config.CONF_MULT[(outcome, confidence)]
        theta = _clamp(theta + DIAG_K * conf_mult * (actual - exp), -THETA_ABS_MAX, THETA_ABS_MAX)
    return theta


def _upsert_mastery(
    db: Session,
    user_id: str,
    concept_id: object,
    theta: float,
    *,
    now: datetime | None,
    misconception: bool,
) -> None:
    m = db.scalars(
        select(Mastery).where(Mastery.user_id == user_id, Mastery.concept_id == concept_id)
    ).first()
    if m is None:
        m = Mastery(user_id=user_id, concept_id=concept_id, theta=theta, stability=1.0)
        db.add(m)
    else:
        m.theta = theta
    if now is not None:
        m.last_seen = now
    if misconception:
        m.misconception_pending = True
    db.flush()


def seed_diagnostic(
    db: Session,
    *,
    user_id: str,
    course: Course,
    graded: list[Graded],
    now: datetime | None = None,
    force: bool = False,
    build_status: bool = True,
) -> DiagnosticResult:
    """Seed mastery from the silent measurement; idempotent unless ``force``.

    Tested concepts get a fixed-K Elo seed; untested chapter-mates get the shrunk
    mean; confident-wrong answers flag ``misconception_pending``. Records the
    diagnostic session and one :class:`Attempt` per answered question.
    """
    now = _now(now)
    if diagnostic_exists(db, user_id, course.id) and not force:
        raise DiagnosticAlreadyRun(
            "diagnostic already run for this learner/course; pass force=True (--force) to reseed"
        )

    session = SessionRun(
        user_id=user_id,
        course_id=course.id,
        kind="diagnostic",
        started_at=now,
        finished_at=now,
        config={"n_questions": len(graded)},
    )
    db.add(session)
    db.flush()

    # Group answers by concept, preserving order (for the running-theta attempts).
    by_concept: dict = {}
    confident_wrong: set = set()
    for g in graded:
        q = g.question
        correct = g.chosen_key == q.correct_key
        by_concept.setdefault(q.concept_id, []).append((g, correct))
        if not correct and g.confidence == "sure":
            confident_wrong.add(q.concept_id)

    tested: dict = {}
    for cid, entries in by_concept.items():
        theta = 0.0
        for g, correct in entries:
            q = g.question
            exp = elo.expected(theta, float(q.q_rating))
            actual = 1.0 if correct else 0.0
            outcome = "correct" if correct else "wrong"
            conf_mult = config.CONF_MULT[(outcome, g.confidence)]
            before = theta
            theta = _clamp(
                before + DIAG_K * conf_mult * (actual - exp), -THETA_ABS_MAX, THETA_ABS_MAX
            )
            db.add(
                Attempt(
                    user_id=user_id,
                    question_id=q.id,
                    concept_id=cid,
                    session_id=session.id,
                    correct=correct,
                    confidence=g.confidence,
                    chosen_key=g.chosen_key,
                    theta_before=before,
                    theta_after=theta,
                    created_at=now,
                )
            )
        tested[cid] = theta
        _upsert_mastery(
            db, user_id, cid, theta, now=now, misconception=(cid in confident_wrong)
        )

    # Untested chapter-mates: theta = DIAG_SHRINK × mean(tested chapter-mates).
    concepts = list(db.scalars(select(Concept).where(Concept.course_id == course.id)).all())
    chapter_of = {c.id: c.chapter_id for c in concepts}
    tested_by_chapter: dict = {}
    for cid, th in tested.items():
        tested_by_chapter.setdefault(chapter_of.get(cid), []).append(th)

    seeded_untested: dict = {}
    for c in concepts:
        if c.id in tested:
            continue
        mates = tested_by_chapter.get(c.chapter_id)
        if not mates:
            continue
        seed = DIAG_SHRINK * (sum(mates) / len(mates))
        seeded_untested[c.id] = seed
        _upsert_mastery(db, user_id, c.id, seed, now=None, misconception=False)

    db.flush()
    report = (
        status.build_status(db, user_id=user_id, course=course, now=now)
        if build_status
        else None
    )
    return DiagnosticResult(
        session_id=session.id,
        n_questions=len(graded),
        tested=tested,
        seeded_untested=seeded_untested,
        misconceptions=confident_wrong,
        status=report,
    )
