"""Adaptive scheduler and session/mock composers (013 §4, T-32).

Two layers. A **pure engine** works on lightweight :class:`ConceptState` /
:class:`QuestionState` value objects — priority scoring, temperature-softmax
sampling, desirable-difficulty question pick, variant exclusion, the N-day
recency reserve, session-mix composition, and a blueprint-stratified mock
composer with a difficulty ramp and a no-same-chapter-adjacency rule. A thin
**DB adapter** loads those value objects from the tables so the study TUI can
call :func:`next_session`. Every tunable comes from :mod:`atlas.core.config`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from atlas.core import config
from atlas.core.models import Attempt, Concept, Mastery, Question
from atlas.learn import elo, retention


# ── value objects ────────────────────────────────────────────────────────────
@dataclass
class ConceptState:
    """Everything the priority formula needs about one concept for one learner."""

    concept_id: object
    theta: float = 0.0
    stability: float = 1.0
    days_since: float | None = None  # since last_seen; None → never practised
    attempts: int = 0
    misconception_pending: bool = False
    chapter: object | None = None


@dataclass
class QuestionState:
    """A servable question, with the per-learner recency the reserve rule needs."""

    question_id: object
    concept_id: object
    q_rating: float = 0.0
    difficulty: int = 2
    variant_group: str | None = None
    days_since_attempt: float | None = None  # since this learner last saw it
    chapter: object | None = None


# ── priority + sampling ──────────────────────────────────────────────────────
def priority(state: ConceptState) -> float:
    """Blend weakness, overdue-ness, misconception and novelty per ``PRIORITY_W``."""
    weak = 1.0 - elo.mastery(state.theta)
    if state.days_since is None:
        overdue = 1.0  # never seen → maximally due
    else:
        overdue = 1.0 - retention.retrievability(state.stability, state.days_since)
    misconception = 1.0 if state.misconception_pending else 0.0
    novelty = 1.0 / (1.0 + max(0, state.attempts))
    w = config.PRIORITY_W
    return (
        w["weak"] * weak
        + w["overdue"] * overdue
        + w["misconception"] * misconception
        + w["novelty"] * novelty
    )


def softmax(scores: list[float], temperature: float = config.SOFTMAX_T) -> list[float]:
    """Numerically stable softmax over ``scores`` at the given temperature."""
    if not scores:
        return []
    hi = max(scores)
    exps = [math.exp((s - hi) / temperature) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def sample_concepts(
    states: list[ConceptState],
    k: int,
    rng: random.Random,
    temperature: float = config.SOFTMAX_T,
) -> list[ConceptState]:
    """Priority-proportional sample of ``k`` distinct concepts (softmax, no replacement)."""
    pool = list(states)
    chosen: list[ConceptState] = []
    for _ in range(min(k, len(pool))):
        weights = softmax([priority(s) for s in pool], temperature)
        idx = _weighted_index(weights, rng)
        chosen.append(pool.pop(idx))
    return chosen


def _weighted_index(weights: list[float], rng: random.Random) -> int:
    r = rng.random()
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(weights) - 1


# ── question pick (desirable difficulty + variant/recency exclusion) ──────────
def eligible_questions(
    questions: list[QuestionState],
    served_variant_groups: set[str],
    reserve_days: int = config.RESERVE_DAYS,
) -> list[QuestionState]:
    """Drop variant-collisions and anything seen inside the reserve window."""
    out = []
    for q in questions:
        group = q.variant_group or str(q.question_id)
        if group in served_variant_groups:
            continue
        if q.days_since_attempt is not None and q.days_since_attempt < reserve_days:
            continue
        out.append(q)
    return out


def pick_question(
    theta: float,
    questions: list[QuestionState],
    served_variant_groups: set[str],
    rng: random.Random,
    reserve_days: int = config.RESERVE_DAYS,
) -> QuestionState | None:
    """Pick the question nearest the desirable-difficulty target ``theta + offset``."""
    candidates = eligible_questions(questions, served_variant_groups, reserve_days)
    if not candidates:
        return None
    target = theta + config.DESIRABLE_DIFFICULTY_OFFSET
    best = min(candidates, key=lambda q: abs(q.q_rating - target))
    tied = [q for q in candidates if abs(abs(q.q_rating - target) - abs(best.q_rating - target)) < 1e-9]
    return rng.choice(tied)


# ── session composition ──────────────────────────────────────────────────────
@dataclass
class ComposedSession:
    """A composed adaptive session: ordered questions plus their bucket labels."""

    questions: list[QuestionState] = field(default_factory=list)
    buckets: list[str] = field(default_factory=list)


def compose_session(
    states: list[ConceptState],
    questions_by_concept: dict,
    rng: random.Random,
    mix: dict | None = None,
) -> ComposedSession:
    """Compose warmup/core/mixed/review buckets into one variant-free session.

    * review  — most overdue previously-seen concepts.
    * warmup  — highest-mastery seen concepts (a confidence ease-in).
    * core    — priority-softmax sample (the adaptive heart).
    * mixed   — random spread across chapters.

    Each chosen concept contributes one desirable-difficulty question; a served
    variant group is never used twice in the session.
    """
    mix = mix or config.SESSION_MIX
    by_id = {s.concept_id: s for s in states}
    seen = [s for s in states if s.days_since is not None]

    used: set[object] = set()
    plan: list[tuple[str, ConceptState]] = []

    def take(bucket: str, ordered: list[ConceptState], n: int) -> None:
        for s in ordered:
            if n <= 0:
                break
            if s.concept_id in used:
                continue
            used.add(s.concept_id)
            plan.append((bucket, s))
            n -= 1

    review_sorted = sorted(
        seen,
        key=lambda s: 1.0 - retention.retrievability(s.stability, s.days_since or 0.0),
        reverse=True,
    )
    take("review", review_sorted, mix.get("review", 0))

    warmup_sorted = sorted(seen, key=lambda s: elo.mastery(s.theta), reverse=True)
    take("warmup", warmup_sorted, mix.get("warmup", 0))

    core_pool = [s for s in states if s.concept_id not in used]
    take("core", sample_concepts(core_pool, mix.get("core", 0), rng), mix.get("core", 0))

    mixed_pool = [s for s in states if s.concept_id not in used]
    rng.shuffle(mixed_pool)
    take("mixed", mixed_pool, mix.get("mixed", 0))

    out = ComposedSession()
    served_groups: set[str] = set()
    for bucket, state in plan:
        q = pick_question(
            state.theta, questions_by_concept.get(state.concept_id, []), served_groups, rng
        )
        if q is None:
            continue
        served_groups.add(q.variant_group or str(q.question_id))
        out.questions.append(q)
        out.buckets.append(bucket)
    _ = by_id  # states already carry chapter/theta; kept for adapter symmetry
    return out


# ── mock composer (blueprint stratified, ramped, adjacency-free) ─────────────
def blueprint_counts(blueprint: dict, total: int) -> dict:
    """Largest-remainder apportionment of ``total`` questions across chapters."""
    weights = {k: max(0.0, float(v)) for k, v in blueprint.items()}
    wsum = sum(weights.values()) or 1.0
    exact = {k: total * v / wsum for k, v in weights.items()}
    floor = {k: int(math.floor(x)) for k, x in exact.items()}
    remainder = total - sum(floor.values())
    order = sorted(exact, key=lambda k: exact[k] - floor[k], reverse=True)
    for k in order[:remainder]:
        floor[k] += 1
    return floor


def compose_mock(
    states: list[ConceptState],
    questions_by_concept: dict,
    blueprint: dict,
    rng: random.Random,
    total: int = 90,
) -> list[QuestionState]:
    """Build the 90-question standalone section, blueprint-stratified.

    Per-chapter counts follow ``blueprint`` (largest-remainder, so ±1 of exact),
    concepts within a chapter are drawn priority-first, each contributes one
    desirable-difficulty question, no variant group repeats, the final order
    ramps easy→hard, and no two consecutive questions share a chapter.
    """
    counts = blueprint_counts(blueprint, total)
    by_chapter_states: dict = {}
    for s in states:
        by_chapter_states.setdefault(s.chapter, []).append(s)

    served_groups: set[str] = set()
    picked_by_chapter: dict = {}
    for chapter, n in counts.items():
        chosen = sample_concepts(by_chapter_states.get(chapter, []), n, rng)
        bucket: list[QuestionState] = []
        # If a chapter has fewer concepts than its quota, revisit concepts for
        # additional (still variant-distinct) questions.
        ci = 0
        while len(bucket) < n and (chosen or by_chapter_states.get(chapter)):
            source = chosen if chosen else by_chapter_states.get(chapter, [])
            if not source:
                break
            state = source[ci % len(source)]
            ci += 1
            q = pick_question(
                state.theta, questions_by_concept.get(state.concept_id, []), served_groups, rng
            )
            if q is None:
                # No eligible question left for any concept in this chapter.
                if ci >= 4 * max(1, len(source)):
                    break
                continue
            served_groups.add(q.variant_group or str(q.question_id))
            if q.chapter is None:
                q = QuestionState(**{**q.__dict__, "chapter": chapter})
            bucket.append(q)
        if bucket:
            picked_by_chapter[chapter] = bucket

    return _ramped_no_adjacency(picked_by_chapter)


def _ramped_no_adjacency(by_chapter: dict) -> list[QuestionState]:
    """Interleave per-chapter (easy→hard) queues, never repeating a chapter back-to-back."""
    queues = {
        ch: sorted(qs, key=lambda q: (q.difficulty, q.q_rating))
        for ch, qs in by_chapter.items()
        if qs
    }
    order: list[QuestionState] = []
    last_chapter = object()
    while queues:
        # Prefer the chapter whose next question is easiest, but not the one just used.
        candidates = [ch for ch in queues if ch != last_chapter] or list(queues)
        chapter = min(candidates, key=lambda ch: (queues[ch][0].difficulty, queues[ch][0].q_rating))
        q = queues[chapter].pop(0)
        if not queues[chapter]:
            del queues[chapter]
        order.append(q)
        last_chapter = chapter
    return order


# ── DB adapters ──────────────────────────────────────────────────────────────
def load_concept_states(
    db: Session, user_id: str, course_id: object, now
) -> list[ConceptState]:
    """Build one :class:`ConceptState` per reviewed concept for this learner."""
    rows = db.scalars(
        select(Concept).where(
            Concept.course_id == course_id,
            Concept.review_status != "dropped",
        )
    ).all()
    mastery_rows = {
        m.concept_id: m
        for m in db.scalars(select(Mastery).where(Mastery.user_id == user_id)).all()
    }
    attempt_counts = dict(
        db.execute(
            select(Attempt.concept_id, func.count())
            .where(Attempt.user_id == user_id)
            .group_by(Attempt.concept_id)
        ).all()
    )
    states = []
    for c in rows:
        m = mastery_rows.get(c.id)
        days_since = None
        if m is not None and m.last_seen is not None:
            days_since = max(0.0, (now - m.last_seen).total_seconds() / 86400.0)
        states.append(
            ConceptState(
                concept_id=c.id,
                theta=float(m.theta) if m else 0.0,
                stability=float(m.stability) if m else 1.0,
                days_since=days_since,
                attempts=int(attempt_counts.get(c.id, 0)),
                misconception_pending=bool(m.misconception_pending) if m else False,
                chapter=c.chapter_id,
            )
        )
    return states


def load_question_states(
    db: Session, user_id: str, course_id: object, concept_ids: list, now
) -> dict:
    """Map each concept id → its servable standalone :class:`QuestionState` list."""
    if not concept_ids:
        return {}
    questions = db.scalars(
        select(Question).where(
            Question.concept_id.in_(concept_ids),
            Question.status == "active",
            Question.case_group_id.is_(None),  # caselets never enter adaptive singles
        )
    ).all()
    q_ids = [q.id for q in questions]
    last_seen: dict = {}
    if q_ids:
        for qid, ts in db.execute(
            select(Attempt.question_id, func.max(Attempt.created_at))
            .where(Attempt.user_id == user_id, Attempt.question_id.in_(q_ids))
            .group_by(Attempt.question_id)
        ).all():
            last_seen[qid] = ts
    out: dict = {}
    for q in questions:
        ts = last_seen.get(q.id)
        days = None if ts is None else max(0.0, (now - ts).total_seconds() / 86400.0)
        out.setdefault(q.concept_id, []).append(
            QuestionState(
                question_id=q.id,
                concept_id=q.concept_id,
                q_rating=float(q.q_rating),
                difficulty=int(q.difficulty),
                variant_group=(q.gate_log or {}).get("variant_group"),
                days_since_attempt=days,
                chapter=q.concept_id,
            )
        )
    return out


def next_session(
    db: Session,
    user_id: str,
    course_id: object,
    now,
    rng: random.Random | None = None,
    mix: dict | None = None,
) -> list[Question]:
    """Compose the next adaptive session and return the ordered ORM questions."""
    rng = rng or random.Random()
    states = load_concept_states(db, user_id, course_id, now)
    qstates = load_question_states(
        db, user_id, course_id, [s.concept_id for s in states], now
    )
    composed = compose_session(states, qstates, rng, mix)
    ordered_ids = [q.question_id for q in composed.questions]
    by_id = {
        q.id: q for q in db.scalars(select(Question).where(Question.id.in_(ordered_ids))).all()
    }
    return [by_id[qid] for qid in ordered_ids if qid in by_id]
