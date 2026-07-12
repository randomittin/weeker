"""Full-pattern mock + exam-strategy report (016 §T-42, 017 §4/§6/§7).

The full NISM XA pattern is 90 standalone + 6 one-mark caselets + 3 two-mark
caselets = 135 questions / 150 marks. This module composes that pattern and
produces the strategy report printed on the mock summary:

* :func:`compose_caselet_items` — pick caselets to **maximise weak-concept
  coverage**, never reusing a caselet a learner saw inside the last 5 days
  (017 §4). One-mark and two-mark tiers are selected separately to hit the
  6 + 3 shape;
* :func:`compose_full_items` — the 90 standalone (via :mod:`weeker.learn.mock`)
  followed by the 9 caselets, each as five positioned :class:`~weeker.learn.mock.MockItem`;
* :func:`predict_sections` — per-section expected score with a bootstrap band
  plus the overall ``P(pass)`` (013 §7 / 017 §4);
* :func:`realized_calibration` — empirical accuracy per confidence keystroke
  ("your 'unsure' is actually 61% correct"), the calibration table of 017 §6;
* :func:`skip_policy_lines` / :func:`strategy_report` — the attempt/skip coaching
  (break-even ``p = neg/(1+neg)`` = 0.2 at 25% negative marking).

Scoring is the marks-weighted engine in :mod:`weeker.learn.mock` (per-question
negative = ``0.25 × marks``, so a wrong 2-mark caselet question costs 0.5).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from weeker.core.models import Attempt, Course, Mastery, Question
from weeker.generate.caselets import caselet_drill
from weeker.learn import elo, mock, predict

CASELET_REUSE_DAYS = 5
SECTION_STANDALONE = "standalone"
SECTION_CASELET_1 = "caselet_1mark"
SECTION_CASELET_2 = "caselet_2mark"


# ── caselet selection (weak-concept coverage, 5-day reuse exclusion) ────────────
def _theta_by_concept(db: Session, user_id: str) -> dict:
    return {
        m.concept_id: float(m.theta)
        for m in db.scalars(select(Mastery).where(Mastery.user_id == user_id)).all()
    }


def _recent_caselet_groups(db: Session, user_id: str, now: datetime, days: int) -> set:
    """Case-group ids this learner attempted within ``days`` (the reuse guard)."""
    cutoff = now - timedelta(days=days)
    rows = db.execute(
        select(Question.case_group_id)
        .join(Attempt, Attempt.question_id == Question.id)
        .where(
            Attempt.user_id == user_id,
            Attempt.created_at >= cutoff,
            Question.case_group_id.is_not(None),
        )
        .distinct()
    ).all()
    return {r[0] for r in rows if r[0] is not None}


def _weak_coverage(theta_by: dict, questions: list[Question]) -> float:
    """Sum of ``1 − mastery`` over the caselet's concepts (higher → weaker → prefer)."""
    concept_ids = {q.concept_id for q in questions}
    return sum(1.0 - elo.mastery(theta_by.get(cid, 0.0)) for cid in concept_ids)


def _tier(questions: list[Question]) -> int:
    """A caselet is 2-mark if any question is worth ≥ 2 marks, else 1-mark."""
    return 2 if any(float(q.marks) >= 2 for q in questions) else 1


def compose_caselet_items(
    db: Session,
    user_id: str,
    course: Course,
    now: datetime,
    *,
    one_mark: int = 6,
    two_mark: int = 3,
    reuse_days: int = CASELET_REUSE_DAYS,
) -> list[mock.MockItem]:
    """Pick ``one_mark`` + ``two_mark`` caselets maximising weak coverage, then flatten to items."""
    theta_by = _theta_by_concept(db, user_id)
    recent = _recent_caselet_groups(db, user_id, now, reuse_days)

    ones: list[tuple[float, object, list[Question]]] = []
    twos: list[tuple[float, object, list[Question]]] = []
    for cg, qs in caselet_drill(db, course.id):
        if not qs or cg.id in recent:
            continue
        cov = _weak_coverage(theta_by, qs)
        (twos if _tier(qs) == 2 else ones).append((cov, cg.id, qs))

    ones.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
    twos.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
    chosen = ones[:one_mark] + twos[:two_mark]

    items: list[mock.MockItem] = []
    for _cov, _cid, qs in chosen:
        for q in sorted(qs, key=lambda x: (x.case_position or 0)):
            items.append(mock.item_from_question(q, mock.CASELET))
    return items


def compose_full_items(
    db: Session,
    user_id: str,
    course: Course,
    now: datetime,
    rng: random.Random,
    *,
    standalone: int = 90,
    one_mark: int = 6,
    two_mark: int = 3,
) -> list[mock.MockItem]:
    """The full pattern: standalone section then the selected caselets, in exam order."""
    section = mock.compose_standalone_items(db, user_id, course, now, rng, total=standalone)
    caselets = compose_caselet_items(
        db, user_id, course, now, one_mark=one_mark, two_mark=two_mark
    )
    return section + caselets


# ── per-section prediction ───────────────────────────────────────────────────────
def _section_of(item: mock.MockItem) -> str:
    if item.section != mock.CASELET:
        return SECTION_STANDALONE
    return SECTION_CASELET_2 if float(item.marks) >= 2 else SECTION_CASELET_1


def _predict_items(db: Session, user_id: str, items: list[mock.MockItem]) -> list[tuple]:
    """Pair each :class:`MockItem` with a :class:`predict.PredictItem` from live state."""
    theta_by = _theta_by_concept(db, user_id)
    counts = dict(
        db.execute(
            select(Attempt.concept_id, func.count())
            .where(Attempt.user_id == user_id)
            .group_by(Attempt.concept_id)
        ).all()
    )
    qmap = {
        q.id: q
        for q in db.scalars(
            select(Question).where(Question.id.in_([it.question_id for it in items]))
        ).all()
    }
    paired = []
    for it in items:
        q = qmap.get(it.question_id)
        paired.append(
            (
                it,
                predict.PredictItem(
                    marks=float(it.marks),
                    theta=theta_by.get(it.concept_id, 0.0),
                    q_rating=float(q.q_rating) if q is not None else 0.0,
                    attempts=int(counts.get(it.concept_id, 0)),
                    covered=counts.get(it.concept_id, 0) > 0,
                ),
            )
        )
    return paired


@dataclass
class SectionPrediction:
    """Expected marks + band for one exam section."""

    section: str
    expected_marks: float
    ci_low: float
    ci_high: float
    total_marks: float


def predict_sections(
    db: Session,
    user_id: str,
    items: list[mock.MockItem],
    *,
    passing_marks: float,
    neg_fraction: float = mock.DEFAULT_NEG_FRACTION,
    rng: random.Random | None = None,
) -> tuple[list[SectionPrediction], predict.Prediction]:
    """Per-section expected score (013 §7) plus the whole-paper ``P(pass)`` prediction."""
    rng = rng or random.Random()
    paired = _predict_items(db, user_id, items)

    by_section: dict = {}
    for it, pi in paired:
        by_section.setdefault(_section_of(it), []).append(pi)

    sections: list[SectionPrediction] = []
    for name in (SECTION_STANDALONE, SECTION_CASELET_1, SECTION_CASELET_2):
        pis = by_section.get(name)
        if not pis:
            continue
        section_total = sum(pi.marks for pi in pis)
        p = predict.predict_score(
            pis,
            passing_marks=section_total,  # section p_pass is not used; band/expected are
            total_marks=section_total,
            neg_fraction=neg_fraction,
            rng=rng,
        )
        sections.append(
            SectionPrediction(
                section=name,
                expected_marks=p.expected_marks,
                ci_low=p.ci_low,
                ci_high=p.ci_high,
                total_marks=section_total,
            )
        )

    overall = predict.predict_score(
        [pi for _, pi in paired],
        passing_marks=passing_marks,
        total_marks=sum(pi.marks for _, pi in paired),
        neg_fraction=neg_fraction,
        rng=rng,
    )
    return sections, overall


# ── realized calibration + skip policy (017 §6) ─────────────────────────────────
@dataclass
class CalibrationRow:
    """Empirical accuracy for one confidence keystroke."""

    confidence: str
    n: int
    correct: int
    rate: float
    recommendation: str


def realized_calibration(
    db: Session,
    user_id: str,
    *,
    neg_fraction: float = mock.DEFAULT_NEG_FRACTION,
) -> list[CalibrationRow]:
    """Accuracy per confidence level, with an attempt/skip recommendation vs break-even."""
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


def skip_policy_lines(neg_fraction: float = mock.DEFAULT_NEG_FRACTION) -> list[str]:
    """The static attempt/skip coaching printed on every mock summary (017 §6)."""
    be = predict.breakeven_p(neg_fraction)
    return [
        f"Negative marking: -{neg_fraction:g} x marks per wrong answer. Break-even p = {be:.2f}.",
        "Can eliminate >= 1 option -> attempt (p >= 1/3 > break-even).",
        "Blind guess (p=0.25) is marginally +EV but noisy: skip 2-mark caselet "
        "questions when time-pressured, attempt 1-markers.",
    ]


@dataclass
class StrategyReport:
    """The full-mock strategy summary: per-section prediction + calibration + skip policy."""

    sections: list[SectionPrediction]
    overall: predict.Prediction
    calibration: list[CalibrationRow]
    skip_policy: list[str]
    breakeven_p: float = field(default=0.0)


def strategy_report(
    db: Session,
    user_id: str,
    items: list[mock.MockItem],
    *,
    passing_marks: float,
    neg_fraction: float = mock.DEFAULT_NEG_FRACTION,
    rng: random.Random | None = None,
) -> StrategyReport:
    """Assemble the strategy report for a composed full-pattern mock."""
    sections, overall = predict_sections(
        db, user_id, items, passing_marks=passing_marks, neg_fraction=neg_fraction, rng=rng
    )
    return StrategyReport(
        sections=sections,
        overall=overall,
        calibration=realized_calibration(db, user_id, neg_fraction=neg_fraction),
        skip_policy=skip_policy_lines(neg_fraction),
        breakeven_p=predict.breakeven_p(neg_fraction),
    )
