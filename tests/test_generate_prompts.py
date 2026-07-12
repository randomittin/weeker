"""T-21 gate: question contract round-trip + recorded-fixture batch parsing.

The contract enforces exactly 4 options, exactly 1 keyed-correct, an explanation,
and a named misconception on every distractor. A recorded generation of five
questions parses into five valid contract objects.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from weeker.core.config import GEN_BATCH
from weeker.core.models import Concept, Course
from weeker.generate.prompts import (
    GeneratedQuestion,
    generate_batch,
    parse_batch,
    render_question_prompt,
)


def _valid_payload(stem="What is the NAV of a fund?", correct="A"):
    opts = [
        {"key": "A", "text": "Assets minus liabilities per unit"},
        {"key": "B", "text": "The market price", "misconception": "price==NAV"},
        {"key": "C", "text": "Total assets", "misconception": "ignores liabilities"},
        {"key": "D", "text": "Units outstanding", "misconception": "confuses units"},
    ]
    # give the correct option no misconception; each distractor one
    for o in opts:
        if o["key"] == correct:
            o.pop("misconception", None)
        elif "misconception" not in o:
            o["misconception"] = "generic error"
    return {
        "stem": stem,
        "options": opts,
        "correct_key": correct,
        "explanation": "NAV = (assets − liabilities) / units.",
        "difficulty": 2,
    }


def test_contract_round_trip():
    q = GeneratedQuestion.model_validate(_valid_payload())
    dumped = q.model_dump()
    again = GeneratedQuestion.model_validate(dumped)
    assert again.correct_key == "A"
    assert len(again.options) == 4
    # option_dicts feeds the Question.options jsonb (key/text/misconception).
    ods = q.option_dicts()
    assert {o["key"] for o in ods} == {"A", "B", "C", "D"}
    assert all("misconception" in o for o in ods if o["key"] != "A")


def test_three_options_rejected():
    payload = _valid_payload()
    payload["options"] = payload["options"][:3]
    with pytest.raises(ValidationError):
        GeneratedQuestion.model_validate(payload)


def test_correct_key_absent_from_options_rejected():
    payload = _valid_payload()
    payload["correct_key"] = "A"
    payload["options"][0]["key"] = "B"  # now two B's, no A
    with pytest.raises(ValidationError):
        GeneratedQuestion.model_validate(payload)


def test_distractor_without_misconception_rejected():
    payload = _valid_payload()
    for o in payload["options"]:
        if o["key"] == "B":
            o.pop("misconception", None)
    with pytest.raises(ValidationError):
        GeneratedQuestion.model_validate(payload)


class FakeGen:
    def __init__(self, payloads):
        self._json = json.dumps({"questions": payloads})
        self.calls = 0

    def request(self, model, system, user):
        self.calls += 1
        return self._json


def test_recorded_fixture_parses_to_five_valid_objects():
    payloads = [_valid_payload(stem=f"Q{i}", correct="ABCD"[i % 4]) for i in range(5)]
    concept = Concept(course_id=Course().id, title="Net asset value", description="per-unit value")
    concept.misconceptions = ["price==NAV"]
    qs = generate_batch(
        concept,
        chunk_texts=["NAV is assets minus liabilities divided by units."],
        transport=FakeGen(payloads),
        model="gen-test",
        batch=GEN_BATCH,
    )
    assert len(qs) == 5
    assert all(isinstance(q, GeneratedQuestion) for q in qs)
    assert all(len(q.options) == 4 for q in qs)


def test_parse_batch_rejects_bad_member():
    good = _valid_payload()
    bad = _valid_payload()
    bad["options"] = bad["options"][:3]
    with pytest.raises(ValidationError):
        parse_batch({"questions": [good, bad]})


def test_render_prompt_substitutes_and_keeps_json_schema():
    concept = Concept(course_id=Course().id, title="NAV", description="per-unit value")
    concept.misconceptions = ["price==NAV"]
    system, user = render_question_prompt(
        concept=concept,
        objective=None,
        chunk_texts=["grounding text here"],
        recent_stem_texts=["an old stem"],
        batch=GEN_BATCH,
    )
    assert "NAV" in user and "grounding text here" in user and "an old stem" in user
    assert "5" in user  # batch count substituted
    assert "correct_key" in user  # contract schema embedded, braces intact
    assert system.strip().startswith("You write ORIGINAL")
