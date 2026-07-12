"""T-21 — the question contract, prompt assembly, and the generation call.

:class:`GeneratedQuestion` is the pydantic contract every authored question must
satisfy before it can enter the gate chain: exactly four options keyed ``A``–``D``,
exactly one keyed-correct answer, an explanation, and a *named misconception* on
each of the three distractors (distractor → misconception mapping). The prompt is
``prompts/question_gen.txt`` (012 §3 verbatim); grounding is the top-4 supporting
chunks per concept (the S5 concept→chunk mapping). ``generate_batch`` asks for a
batch of :data:`~weeker.core.config.GEN_BATCH` questions and parses them through
the contract.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from weeker.core import config
from weeker.core.llm import Transport, complete
from weeker.core.models import Chunk, Concept, ConceptChunk, Objective, Question
from weeker.ingest.textutil import load_prompt, render_prompt

OPTION_KEYS = ("A", "B", "C", "D")
Key = Literal["A", "B", "C", "D"]

# Top supporting chunks handed to the model as grounding (012 §3: top-4).
GROUNDING_TOPK = 4
# How many existing stems to show so the model avoids repeating an angle.
RECENT_STEM_LIMIT = 10


class Option(BaseModel):
    """One answer option. Distractors carry the misconception they embody."""

    model_config = ConfigDict(extra="forbid")

    key: Key
    text: str
    misconception: str | None = None


class GeneratedQuestion(BaseModel):
    """The authored-question contract enforced before any gate runs."""

    model_config = ConfigDict(extra="forbid")

    stem: str
    options: list[Option]
    correct_key: Key
    explanation: str
    difficulty: int = 2

    @model_validator(mode="after")
    def _enforce_contract(self) -> GeneratedQuestion:
        if not self.stem.strip():
            raise ValueError("stem must be non-empty")
        if len(self.options) != 4:
            raise ValueError(f"exactly 4 options required, got {len(self.options)}")
        keys = [o.key for o in self.options]
        if sorted(keys) != list(OPTION_KEYS):
            raise ValueError(f"option keys must be exactly A,B,C,D distinct, got {keys}")
        if self.correct_key not in keys:
            raise ValueError("correct_key must name one of the options")
        distractors = [o for o in self.options if o.key != self.correct_key]
        missing = [o.key for o in distractors if not (o.misconception and o.misconception.strip())]
        if missing:
            raise ValueError(f"distractor(s) {missing} must each name a misconception")
        if not self.explanation.strip():
            raise ValueError("explanation must be non-empty")
        if self.difficulty not in (1, 2, 3):
            raise ValueError("difficulty must be 1, 2 or 3")
        return self

    def option_dicts(self) -> list[dict]:
        """Options in the ``Question.options`` jsonb shape (key/text/misconception)."""
        return [o.model_dump(exclude_none=True) for o in self.options]


# JSON-Schema shape of one question, embedded in the prompt as {question_contract_json}.
QUESTION_CONTRACT_SCHEMA: dict = {
    "type": "object",
    "required": ["stem", "options", "correct_key", "explanation", "difficulty"],
    "properties": {
        "stem": {"type": "string"},
        "options": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["key", "text"],
                "properties": {
                    "key": {"type": "string", "enum": list(OPTION_KEYS)},
                    "text": {"type": "string"},
                    "misconception": {"type": "string"},
                },
            },
        },
        "correct_key": {"type": "string", "enum": list(OPTION_KEYS)},
        "explanation": {"type": "string"},
        "difficulty": {"type": "integer", "enum": [1, 2, 3]},
    },
}

BATCH_SCHEMA: dict = {
    "type": "object",
    "required": ["questions"],
    "properties": {"questions": {"type": "array", "items": QUESTION_CONTRACT_SCHEMA}},
}


def question_contract_json() -> str:
    """Pretty-printed single-question JSON schema for the prompt body."""
    return json.dumps(QUESTION_CONTRACT_SCHEMA, indent=2)


def parse_batch(data: dict) -> list[GeneratedQuestion]:
    """Validate a ``{"questions": [...]}`` payload into contract objects."""
    return [GeneratedQuestion.model_validate(q) for q in data.get("questions", [])]


def top_chunks(session: Session, concept_id: object, k: int = GROUNDING_TOPK) -> list[Chunk]:
    """Top-``k`` supporting chunks for a concept via the S5 concept→chunk mapping."""
    return list(
        session.scalars(
            select(Chunk)
            .join(ConceptChunk, ConceptChunk.chunk_id == Chunk.id)
            .where(ConceptChunk.concept_id == concept_id)
            .order_by(ConceptChunk.cos.desc())
            .limit(k)
        )
    )


def recent_stems(
    session: Session, concept_id: object, limit: int = RECENT_STEM_LIMIT
) -> list[str]:
    """Existing question stems for a concept (most recent first)."""
    return list(
        session.scalars(
            select(Question.stem)
            .where(Question.concept_id == concept_id)
            .order_by(Question.created_at.desc())
            .limit(limit)
        )
    )


def _first_objective(session: Session, concept_id: object) -> Objective | None:
    return session.scalars(
        select(Objective).where(Objective.concept_id == concept_id).limit(1)
    ).first()


def render_question_prompt(
    *,
    concept: Concept,
    objective: Objective | None,
    chunk_texts: list[str],
    recent_stem_texts: list[str],
    batch: int = config.GEN_BATCH,
) -> tuple[str, str]:
    """Render (system, user) from ``prompts/question_gen.txt`` for one concept."""
    system, user_t = load_prompt("question_gen")
    misconceptions = ", ".join(concept.misconceptions or []) or "none recorded"
    user = render_prompt(
        user_t,
        title=concept.title,
        description=concept.description or "",
        objective_text=objective.text if objective else "apply this concept correctly",
        bloom=objective.bloom if objective else "apply",
        style=objective.assessment_style if objective else "scenario",
        difficulty=concept.difficulty,
        misconceptions=misconceptions,
        chunks="\n\n".join(chunk_texts) or "(no grounding chunks available)",
        recent_stems="\n".join(f"- {s}" for s in recent_stem_texts) or "(none yet)",
        question_contract_json=question_contract_json(),
        batch=batch,
    )
    return system, user


def generate_batch(
    concept: Concept,
    *,
    chunk_texts: list[str],
    objective: Objective | None = None,
    recent_stem_texts: list[str] | None = None,
    batch: int = config.GEN_BATCH,
    model: str | None = None,
    transport: Transport | None = None,
) -> list[GeneratedQuestion]:
    """Generate and contract-validate a batch of questions for one concept."""
    system, user = render_question_prompt(
        concept=concept,
        objective=objective,
        chunk_texts=chunk_texts,
        recent_stem_texts=recent_stem_texts or [],
        batch=batch,
    )
    data = complete(
        model or config.MODEL_GENERATOR,
        system,
        user,
        json_schema=BATCH_SCHEMA,
        transport=transport,
    )
    assert isinstance(data, dict)
    return parse_batch(data)


def generate_for_concept(
    session: Session,
    concept: Concept,
    *,
    batch: int = config.GEN_BATCH,
    model: str | None = None,
    transport: Transport | None = None,
) -> list[GeneratedQuestion]:
    """DB-facing batch generation: pulls grounding chunks + recent stems, then generates."""
    chunk_texts = [c.content for c in top_chunks(session, concept.id)]
    return generate_batch(
        concept,
        chunk_texts=chunk_texts,
        objective=_first_objective(session, concept.id),
        recent_stem_texts=recent_stems(session, concept.id),
        batch=batch,
        model=model,
        transport=transport,
    )
