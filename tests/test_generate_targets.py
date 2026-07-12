"""T-20 gate: initial-fill + refill quota logic (012 §2).

Quotas sum to N; a min-3 floor per included concept; concepts at mastery ≥ 0.9 are
excluded on refill; refill effective weight follows ``weight × (1 − mastery)``.
"""

from __future__ import annotations

import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from atlas.core.models import Base, BlueprintWeight
from atlas.generate.targets import (
    MASTERY_EXCLUDE_CAP,
    MIN_FLOOR,
    ConceptWeight,
    allocate_targets,
    compute_targets,
)
from tests.test_learn_fixtures import (
    new_user,
    seed_chapter,
    seed_concept,
    seed_course,
    seed_mastery,
)


def _items(n: int) -> list[ConceptWeight]:
    return [
        ConceptWeight(concept_id=uuid.uuid4(), weight=1.0 / n, mastery=0.0) for _ in range(n)
    ]


def test_initial_quotas_sum_to_total():
    items = _items(4)
    targets = allocate_targets(items, total=40, refill=False)
    assert sum(t.quota for t in targets) == 40
    assert {t.concept_id for t in targets} == {i.concept_id for i in items}


def test_min_floor_per_concept():
    # 3 concepts, one carrying almost all the weight — the light ones still floor at 3.
    items = [
        ConceptWeight(concept_id=uuid.uuid4(), weight=0.90),
        ConceptWeight(concept_id=uuid.uuid4(), weight=0.05),
        ConceptWeight(concept_id=uuid.uuid4(), weight=0.05),
    ]
    targets = allocate_targets(items, total=30, refill=False)
    assert sum(t.quota for t in targets) == 30
    assert all(t.quota >= MIN_FLOOR for t in targets)


def test_mastery_exclusion_on_refill():
    mastered = ConceptWeight(concept_id=uuid.uuid4(), weight=0.5, mastery=MASTERY_EXCLUDE_CAP)
    weak = ConceptWeight(concept_id=uuid.uuid4(), weight=0.5, mastery=0.1)
    targets = allocate_targets([mastered, weak], total=20, refill=True)
    ids = {t.concept_id for t in targets}
    assert mastered.concept_id not in ids
    assert weak.concept_id in ids
    assert sum(t.quota for t in targets) == 20


def test_refill_weight_is_weight_times_one_minus_mastery():
    # Equal base weight, different mastery → the weaker concept gets the larger share.
    a = ConceptWeight(concept_id=uuid.uuid4(), weight=0.5, mastery=0.0)  # eff 0.5
    b = ConceptWeight(concept_id=uuid.uuid4(), weight=0.5, mastery=0.6)  # eff 0.2
    targets = {t.concept_id: t.quota for t in allocate_targets([a, b], total=70, refill=True)}
    # both floored at 3, remainder 64 split 0.5:0.2 → ~46 / ~18 extra
    assert targets[a.concept_id] > targets[b.concept_id]
    assert sum(targets.values()) == 70


def test_total_below_floor_sum_still_sums_exactly():
    items = _items(5)  # 5 concepts, floor would want 15
    targets = allocate_targets(items, total=8, refill=False)
    assert sum(t.quota for t in targets) == 8
    assert all(t.quota >= 0 for t in targets)


def test_compute_targets_from_blueprint_and_mastery():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as db:
        course = seed_course(db)
        ch_a = seed_chapter(db, course, 1, "Alpha")
        ch_b = seed_chapter(db, course, 2, "Beta")
        c1 = seed_concept(db, course, "c1", chapter=ch_a)
        c2 = seed_concept(db, course, "c2", chapter=ch_a)
        c3 = seed_concept(db, course, "c3", chapter=ch_b)
        db.add_all(
            [
                BlueprintWeight(course_id=course.id, chapter="Alpha", weight=0.6),
                BlueprintWeight(course_id=course.id, chapter="Beta", weight=0.4),
            ]
        )
        db.flush()
        user = new_user()
        # c1 fully mastered → excluded on refill.
        seed_mastery(db, user, c1, theta=5.0)
        seed_mastery(db, user, c2, theta=0.0)
        seed_mastery(db, user, c3, theta=0.0)
        db.flush()

        init = compute_targets(db, course.id, user, total=60, refill=False)
        assert sum(t.quota for t in init) == 60
        assert len(init) == 3  # every concept fills initially

        refill = compute_targets(db, course.id, user, total=40, refill=True)
        assert sum(t.quota for t in refill) == 40
        assert c1.id not in {t.concept_id for t in refill}  # mastered dropped
