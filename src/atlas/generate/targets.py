"""T-20 — generation target quotas (012 §2).

How many questions to author per concept, for both the initial bank fill and the
nightly refill:

* **initial fill** — spread ``total`` questions across every concept in proportion
  to its blueprint weight, with a floor of :data:`MIN_FLOOR` per concept so no
  concept is left thin;
* **refill** — drop concepts already at mastery ≥ :data:`MASTERY_EXCLUDE_CAP`, then
  weight each remaining concept by ``blueprint_weight × (1 − mastery)`` so effort
  concentrates on weak spots.

A concept's blueprint weight is its chapter's weight shared equally among the
chapter's concepts, so the per-concept weights sum to the (normalized) blueprint.
Apportionment uses the largest-remainder (Hamilton) method, so quotas always sum
to exactly ``total``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from atlas.core.models import BlueprintWeight, Concept, Mastery
from atlas.learn import elo

# Floor of authored questions per included concept (structural, not a scoring gate).
MIN_FLOOR = 3
# Concepts at or above this mastery are excluded from refill generation (012 §2).
MASTERY_EXCLUDE_CAP = 0.9


@dataclass(frozen=True)
class ConceptWeight:
    """A concept's generation weight and current mastery (0–1)."""

    concept_id: uuid.UUID
    weight: float
    mastery: float = 0.0


@dataclass(frozen=True)
class Target:
    """How many questions to author for one concept."""

    concept_id: uuid.UUID
    quota: int


def _largest_remainder(weights: list[float], total: int) -> list[int]:
    """Apportion ``total`` integer seats across ``weights`` (Hamilton method).

    The returned counts sum to exactly ``total``. Zero/negative total → all zeros;
    zero total weight → an even split by largest remainder on equal shares.
    """
    n = len(weights)
    if n == 0 or total <= 0:
        return [0] * n
    wsum = sum(w for w in weights if w > 0)
    shares = [(w / wsum if w > 0 else 0.0) * total for w in weights] if wsum > 0 else [
        total / n
    ] * n
    floors = [int(s) for s in shares]
    remainder = total - sum(floors)
    order = sorted(range(n), key=lambda i: shares[i] - floors[i], reverse=True)
    for i in order[:remainder]:
        floors[i] += 1
    return floors


def allocate_targets(
    items: list[ConceptWeight],
    total: int,
    *,
    refill: bool,
    min_floor: int = MIN_FLOOR,
    mastery_cap: float = MASTERY_EXCLUDE_CAP,
) -> list[Target]:
    """Allocate ``total`` questions across ``items``; quotas sum to exactly ``total``.

    On ``refill`` concepts at mastery ≥ ``mastery_cap`` are excluded and effective
    weight becomes ``weight × (1 − mastery)``. Every included concept floors at
    ``min_floor`` when the total allows; when ``total`` is below the floor sum, the
    total is apportioned by largest remainder instead (still summing exactly).
    """
    pool = [it for it in items if not (refill and it.mastery >= mastery_cap)]
    if not pool:
        return []

    eff = [it.weight * (1.0 - it.mastery) if refill else it.weight for it in pool]

    floor_sum = min_floor * len(pool)
    if total <= floor_sum:
        counts = _largest_remainder(eff, total)
    else:
        extra = _largest_remainder(eff, total - floor_sum)
        counts = [min_floor + e for e in extra]
    return [Target(concept_id=it.concept_id, quota=c) for it, c in zip(pool, counts, strict=True)]


def _concept_weights(
    session: Session, course_id: uuid.UUID, user_id: str
) -> list[ConceptWeight]:
    """Build per-concept generation weights from the blueprint + mastery table."""
    concepts = list(
        session.scalars(select(Concept).where(Concept.course_id == course_id))
    )
    if not concepts:
        return []

    # Blueprint weights are keyed by chapter *title*; map each chapter id to it.
    from atlas.core.models import Chapter

    title_by_chapter = dict(
        session.execute(
            select(Chapter.id, Chapter.title).where(Chapter.course_id == course_id)
        ).all()
    )
    weight_by_title = dict(
        session.execute(
            select(BlueprintWeight.chapter, BlueprintWeight.weight).where(
                BlueprintWeight.course_id == course_id
            )
        ).all()
    )
    # Count concepts per chapter to share a chapter's weight equally.
    per_chapter: dict[uuid.UUID | None, int] = {}
    for c in concepts:
        per_chapter[c.chapter_id] = per_chapter.get(c.chapter_id, 0) + 1

    mastery_by_concept = dict(
        session.execute(
            select(Mastery.concept_id, Mastery.theta).where(
                Mastery.user_id == user_id,
                Mastery.concept_id.in_([c.id for c in concepts]),
            )
        ).all()
    )

    out: list[ConceptWeight] = []
    for c in concepts:
        title = title_by_chapter.get(c.chapter_id)
        chapter_w = float(weight_by_title.get(title, 0.0)) if title else 0.0
        n_in_chapter = per_chapter.get(c.chapter_id, 1) or 1
        weight = chapter_w / n_in_chapter
        theta = mastery_by_concept.get(c.id)
        mastery = elo.mastery(float(theta)) if theta is not None else 0.0
        out.append(ConceptWeight(concept_id=c.id, weight=weight, mastery=mastery))
    return out


def compute_targets(
    session: Session,
    course_id: uuid.UUID,
    user_id: str,
    total: int,
    *,
    refill: bool = False,
) -> list[Target]:
    """Per-concept authoring quotas for a course, read from blueprint + mastery."""
    return allocate_targets(_concept_weights(session, course_id, user_id), total, refill=refill)
