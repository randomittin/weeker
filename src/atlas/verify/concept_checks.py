"""Concept gates as code — the M2 gate (016 §T-17).

``verify_concepts`` asserts the S5/S6 output is fit to generate against:

1. every reviewed (accepted/edited) concept has ≥ 1 learning objective;
2. blueprint weights exist for the course and normalize to 1.0 across chapters;
3. no orphan concepts — every reviewed concept is anchored to a chapter and
   grounded in ≥ 1 source chunk (a ``concept_chunks`` link).

Returns a list of human-readable failure strings; ``[]`` means the gate passed.
Reads persisted rows via an injected session so it runs identically on SQLite
(tests) and Postgres (production).
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from atlas.core.models import BlueprintWeight, Concept, ConceptChunk, Objective

REVIEWED_STATUSES = ("accepted", "edited")
BLUEPRINT_SUM_TOL = 1e-4


def verify_concepts(session: Session, course_id: uuid.UUID) -> list[str]:
    """Full M2 concept gate. Returns [] on pass, else failure strings."""
    failures: list[str] = []

    reviewed = list(
        session.scalars(
            select(Concept)
            .where(Concept.course_id == course_id)
            .where(Concept.review_status.in_(REVIEWED_STATUSES))
        )
    )
    if not reviewed:
        return [f"course {course_id}: no reviewed concepts (run the review pass first)"]

    obj_counts = dict(
        session.execute(
            select(Objective.concept_id, func.count(Objective.id))
            .where(Objective.concept_id.in_([c.id for c in reviewed]))
            .group_by(Objective.concept_id)
        ).all()
    )
    link_counts = dict(
        session.execute(
            select(ConceptChunk.concept_id, func.count(ConceptChunk.id))
            .where(ConceptChunk.concept_id.in_([c.id for c in reviewed]))
            .group_by(ConceptChunk.concept_id)
        ).all()
    )

    no_objective = [c for c in reviewed if obj_counts.get(c.id, 0) < 1]
    if no_objective:
        failures.append(
            f"{len(no_objective)} reviewed concept(s) without an objective: "
            f"e.g. {[c.title for c in no_objective[:5]]}"
        )

    orphan_chapter = [c for c in reviewed if c.chapter_id is None]
    if orphan_chapter:
        failures.append(
            f"{len(orphan_chapter)} orphan concept(s) with no chapter: "
            f"e.g. {[c.title for c in orphan_chapter[:5]]}"
        )

    orphan_source = [c for c in reviewed if link_counts.get(c.id, 0) < 1]
    if orphan_source:
        failures.append(
            f"{len(orphan_source)} orphan concept(s) with no source chunk: "
            f"e.g. {[c.title for c in orphan_source[:5]]}"
        )

    weights = list(
        session.scalars(
            select(BlueprintWeight).where(BlueprintWeight.course_id == course_id)
        )
    )
    if not weights:
        failures.append("no blueprint weights (run S6)")
    else:
        total = sum(float(w.weight) for w in weights)
        if abs(total - 1.0) > BLUEPRINT_SUM_TOL:
            failures.append(f"blueprint weights sum to {total:.5f}, expected 1.0")

    return failures
