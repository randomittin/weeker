"""S6 — learning objectives + blueprint weights (016 §T-17).

Two halves of the S6 stage:

* :func:`generate_objectives` — batches reviewed concepts (10 per call) through
  ``prompts/objectives.txt`` and persists 1-3 :class:`Objective` rows per concept.
  The prompt returns an object keyed by concept id, so the response is validated
  per concept rather than against a fixed-key schema.
* :func:`compute_blueprint_weights` / :func:`persist_blueprint_weights` — a
  per-chapter weight derived from reviewed-concept counts, with manifest
  overrides taken as authoritative for the chapters they name and the remaining
  mass split proportionally across the rest. The result always sums to 1.0.

:func:`run_objectives` wires both against one course for the pipeline.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from weeker.core.llm import Transport, complete
from weeker.core.models import BlueprintWeight, Chapter, Concept, Objective
from weeker.ingest.textutil import load_prompt, render_prompt

OBJECTIVES_BATCH = 10
REVIEWED_STATUSES = ("accepted", "edited")

BLOOM_VALUES = {"remember", "understand", "apply", "analyze", "evaluate"}
ASSESSMENT_VALUES = {
    "definition",
    "scenario",
    "calculation",
    "regulator-id",
    "suitability",
    "ethics",
    "trap",
}

# Per-concept shape validated in code; the top-level object has dynamic (concept
# id) keys, so the schema handed to ``complete`` is just "an object".
_OBJ_SCHEMA = {"type": "object"}


def _reviewed_concepts(session: Session, course_id) -> list[Concept]:
    return list(
        session.scalars(
            select(Concept)
            .where(Concept.course_id == course_id)
            .where(Concept.review_status.in_(REVIEWED_STATUSES))
            .order_by(Concept.title)
        )
    )


def _batch_block(concepts: list[Concept]) -> str:
    lines = []
    for c in concepts:
        lines.append(f"id: {c.id}\ntitle: {c.title}\ndescription: {c.description}")
    return "\n\n".join(lines)


def _clean_objective(raw: dict) -> tuple[str, str, str] | None:
    """Coerce one objective dict into ``(text, bloom, assessment_style)`` or None."""
    text = str(raw.get("text", "")).strip()
    if not text:
        return None
    bloom = str(raw.get("bloom", "")).strip().lower()
    if bloom not in BLOOM_VALUES:
        bloom = "understand"
    style = str(raw.get("assessment_style", "")).strip().lower()
    if style not in ASSESSMENT_VALUES:
        style = "definition"
    return text, bloom, style


def generate_objectives(
    session: Session,
    course,
    transport: Transport | None = None,
    model: str = "objectives",
    batch: int = OBJECTIVES_BATCH,
) -> int:
    """Generate + persist objectives for every reviewed concept. Returns count.

    Existing objectives for the course's reviewed concepts are replaced so the
    stage is idempotent.
    """
    concepts = _reviewed_concepts(session, course.id)
    if not concepts:
        return 0

    ids = [c.id for c in concepts]
    session.query(Objective).filter(Objective.concept_id.in_(ids)).delete(
        synchronize_session=False
    )
    session.flush()

    system, user_t = load_prompt("objectives")
    by_id = {str(c.id): c for c in concepts}
    made = 0
    for start in range(0, len(concepts), batch):
        chunk = concepts[start : start + batch]
        user = render_prompt(
            user_t, batch_of_10_concepts_with_descriptions=_batch_block(chunk)
        )
        data = complete(model, system, user, json_schema=_OBJ_SCHEMA, transport=transport)
        assert isinstance(data, dict)
        for cid, payload in data.items():
            concept = by_id.get(str(cid))
            if concept is None or not isinstance(payload, dict):
                continue
            for raw in payload.get("objectives", []) or []:
                if not isinstance(raw, dict):
                    continue
                cleaned = _clean_objective(raw)
                if cleaned is None:
                    continue
                text, bloom, style = cleaned
                session.add(
                    Objective(
                        concept_id=concept.id,
                        text=text,
                        bloom=bloom,
                        assessment_style=style,
                    )
                )
                made += 1
    session.flush()
    return made


def compute_blueprint_weights(
    session: Session, course, overrides: dict[str, float] | None = None
) -> dict[str, float]:
    """Per-chapter blueprint weights summing to 1.0.

    Base weight of a chapter is proportional to the number of reviewed concepts
    it holds (floored at 1 so no chapter is weightless). Chapters named in
    ``overrides`` take their given weight verbatim; the remaining mass
    ``1 - Σ overrides`` is split across the other chapters by base share. With no
    overrides this reduces to a normalized concept-count distribution.
    """
    chapters = list(
        session.scalars(
            select(Chapter).where(Chapter.course_id == course.id).order_by(Chapter.ordinal)
        )
    )
    if not chapters:
        return {}

    overrides = {k: float(v) for k, v in (overrides or {}).items()}
    base: dict[str, float] = {}
    for ch in chapters:
        count = (
            session.query(Concept)
            .filter(
                Concept.chapter_id == ch.id,
                Concept.review_status.in_(REVIEWED_STATUSES),
            )
            .count()
        )
        # Floor at 1 so a chapter with no kept concepts still gets base mass.
        base[ch.title] = float(max(1, count))

    over = {t: w for t, w in overrides.items() if t in base}
    non_over = [t for t in base if t not in over]
    sum_over = sum(over.values())
    remaining = max(0.0, 1.0 - sum_over)

    weights: dict[str, float] = {}
    if not non_over:
        # Every chapter overridden — normalize the overrides to be safe.
        total = sum_over or 1.0
        return {t: over[t] / total for t in base}

    base_non_total = sum(base[t] for t in non_over) or float(len(non_over))
    for t in base:
        if t in over:
            weights[t] = over[t]
        else:
            weights[t] = remaining * (base[t] / base_non_total)
    return weights


def persist_blueprint_weights(
    session: Session, course, weights: dict[str, float]
) -> int:
    """Replace the course's blueprint rows with ``weights``. Returns row count."""
    session.query(BlueprintWeight).filter(
        BlueprintWeight.course_id == course.id
    ).delete(synchronize_session=False)
    session.flush()
    for chapter, weight in weights.items():
        session.add(
            BlueprintWeight(
                course_id=course.id, chapter=chapter, weight=round(float(weight), 5)
            )
        )
    session.flush()
    return len(weights)


def run_objectives(
    session: Session,
    course,
    transport: Transport | None = None,
    overrides: dict[str, float] | None = None,
) -> tuple[int, dict[str, float]]:
    """Full S6: generate objectives, compute + persist blueprint weights."""
    made = generate_objectives(session, course, transport=transport)
    weights = compute_blueprint_weights(session, course, overrides)
    persist_blueprint_weights(session, course, weights)
    return made, weights
