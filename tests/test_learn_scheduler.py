"""T-32 statistical + structural tests for the scheduler and composers."""

from __future__ import annotations

import math
import random

from weeker.core import config
from weeker.learn import scheduler
from weeker.learn.scheduler import ConceptState, QuestionState


def _pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / math.sqrt(vx * vy)


def _synthetic_concepts(n, rng):
    states = []
    for i in range(n):
        states.append(
            ConceptState(
                concept_id=i,
                theta=rng.uniform(-2.5, 2.5),
                stability=rng.uniform(0.5, 20.0),
                days_since=rng.choice([None, rng.uniform(0, 30)]),
                attempts=rng.randint(0, 25),
                misconception_pending=rng.random() < 0.15,
                chapter=i % 12,
            )
        )
    return states


def test_serve_share_correlates_with_priority():
    rng = random.Random(7)
    states = _synthetic_concepts(50, rng)
    counts = {s.concept_id: 0 for s in states}
    for _ in range(500):
        for s in scheduler.sample_concepts(states, 10, rng):
            counts[s.concept_id] += 1
    prios = [scheduler.priority(s) for s in states]
    serve = [counts[s.concept_id] for s in states]
    rho = _pearson(prios, serve)
    assert rho > 0.8, f"serve/priority correlation too low: {rho:.3f}"


def test_priority_weights_move_scores_the_right_way():
    base = ConceptState(concept_id="b", theta=0.0, stability=5.0, days_since=1.0, attempts=5)
    weaker = ConceptState(concept_id="w", theta=-2.0, stability=5.0, days_since=1.0, attempts=5)
    overdue = ConceptState(concept_id="o", theta=0.0, stability=5.0, days_since=60.0, attempts=5)
    misc = ConceptState(
        concept_id="m", theta=0.0, stability=5.0, days_since=1.0, attempts=5,
        misconception_pending=True,
    )
    novel = ConceptState(concept_id="n", theta=0.0, stability=5.0, days_since=1.0, attempts=0)
    p = scheduler.priority
    assert p(weaker) > p(base)
    assert p(overdue) > p(base)
    assert p(misc) > p(base)
    assert p(novel) > p(base)


def _questions_for(states, per_concept=4, rng=None, variant_every=0):
    rng = rng or random.Random(0)
    out = {}
    qid = 0
    for s in states:
        lst = []
        for j in range(per_concept):
            group = None
            if variant_every and j < variant_every:
                group = f"vg-{s.concept_id}"  # these share a variant group
            lst.append(
                QuestionState(
                    question_id=f"q{qid}",
                    concept_id=s.concept_id,
                    q_rating=rng.uniform(-2, 2),
                    difficulty=rng.randint(1, 3),
                    variant_group=group,
                    days_since_attempt=None,
                    chapter=s.chapter,
                )
            )
            qid += 1
        out[s.concept_id] = lst
    return out


def test_session_has_no_variant_pairs():
    rng = random.Random(3)
    states = _synthetic_concepts(60, rng)
    qbc = _questions_for(states, per_concept=4, rng=rng, variant_every=2)
    for _ in range(50):
        composed = scheduler.compose_session(states, qbc, rng)
        groups = [q.variant_group for q in composed.questions if q.variant_group]
        assert len(groups) == len(set(groups)), "variant collision inside a session"


def test_reserve_excludes_recently_seen_questions():
    q_recent = QuestionState("r", "c", 0.0, 2, None, days_since_attempt=1.0)
    q_old = QuestionState("o", "c", 0.0, 2, None, days_since_attempt=10.0)
    elig = scheduler.eligible_questions([q_recent, q_old], set())
    ids = {q.question_id for q in elig}
    assert "r" not in ids and "o" in ids  # 1 day < RESERVE_DAYS (3), 10 days ok


def test_desirable_difficulty_pick_targets_theta_plus_offset():
    rng = random.Random(1)
    theta = 0.5
    target = theta + config.DESIRABLE_DIFFICULTY_OFFSET
    qs = [
        QuestionState("a", "c", q_rating=-1.0, difficulty=2),
        QuestionState("b", "c", q_rating=target, difficulty=2),
        QuestionState("c", "c", q_rating=2.0, difficulty=2),
    ]
    chosen = scheduler.pick_question(theta, qs, set(), rng)
    assert chosen.question_id == "b"


def _mock_setup(rng):
    # 12 chapters, 5 concepts each, 5 questions each.
    states = []
    cid = 0
    for ch in range(12):
        for _ in range(5):
            states.append(
                ConceptState(
                    concept_id=cid, theta=rng.uniform(-2, 2), stability=5.0,
                    days_since=rng.uniform(0, 20), attempts=rng.randint(0, 10), chapter=ch,
                )
            )
            cid += 1
    qbc = _questions_for(states, per_concept=5, rng=rng)
    blueprint = {ch: rng.uniform(0.5, 2.0) for ch in range(12)}
    return states, qbc, blueprint


def test_mock_matches_blueprint_within_one():
    rng = random.Random(11)
    states, qbc, blueprint = _mock_setup(rng)
    total = 90
    mock = scheduler.compose_mock(states, qbc, blueprint, rng, total=total)
    assert len(mock) == total
    expected = scheduler.blueprint_counts(blueprint, total)
    got: dict = {}
    for q in mock:
        got[q.chapter] = got.get(q.chapter, 0) + 1
    for ch, exp in expected.items():
        assert abs(got.get(ch, 0) - exp) <= 1, f"chapter {ch}: got {got.get(ch,0)} vs {exp}"


def test_mock_has_no_variant_pairs_and_no_chapter_adjacency():
    rng = random.Random(13)
    states, qbc, blueprint = _mock_setup(rng)
    mock = scheduler.compose_mock(states, qbc, blueprint, rng, total=90)
    groups = [q.variant_group for q in mock if q.variant_group]
    assert len(groups) == len(set(groups))
    adjacent = sum(1 for a, b in zip(mock, mock[1:], strict=False) if a.chapter == b.chapter)
    assert adjacent == 0, f"{adjacent} same-chapter adjacencies"


def test_mock_difficulty_ramps_upward():
    rng = random.Random(17)
    states, qbc, blueprint = _mock_setup(rng)
    mock = scheduler.compose_mock(states, qbc, blueprint, rng, total=90)
    third = len(mock) // 3
    first_mean = sum(q.difficulty for q in mock[:third]) / third
    last_mean = sum(q.difficulty for q in mock[-third:]) / third
    assert last_mean > first_mean, f"no ramp: {first_mean:.2f} -> {last_mean:.2f}"
