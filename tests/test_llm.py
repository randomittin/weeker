"""llm.complete: strict-JSON retry against a fake transport."""

from __future__ import annotations

import pytest

from atlas.core.llm import LLMError, complete

SCHEMA = {
    "type": "object",
    "required": ["title", "count"],
    "properties": {
        "title": {"type": "string"},
        "count": {"type": "integer"},
    },
}


class FakeTransport:
    """Returns a scripted response per call; records the user prompts it saw."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, model, system, user):
        self.calls.append(user)
        return self.responses.pop(0)


def test_malformed_json_then_valid_retry():
    t = FakeTransport(['{"title": "x", not json', '{"title": "ok", "count": 3}'])
    out = complete("m", "sys", "do it", json_schema=SCHEMA, transport=t)
    assert out == {"title": "ok", "count": 3}
    assert len(t.calls) == 2
    # the retry prompt must carry the validation error back to the model
    assert "JSON" in t.calls[1] and t.calls[1] != t.calls[0]


def test_schema_violation_then_valid_retry():
    t = FakeTransport(['{"title": "x"}', '{"title": "x", "count": 1}'])
    out = complete("m", "sys", "u", json_schema=SCHEMA, transport=t)
    assert out == {"title": "x", "count": 1}
    assert len(t.calls) == 2
    assert "count" in t.calls[1]


def test_exhausted_retries_raises():
    t = FakeTransport(["nope", "still nope", "again nope"])
    with pytest.raises(LLMError):
        complete("m", "s", "u", json_schema=SCHEMA, max_retries=2, transport=t)
    assert len(t.calls) == 3  # initial + 2 retries


def test_plain_text_no_schema():
    t = FakeTransport(["just some text"])
    out = complete("m", "s", "u", transport=t)
    assert out == "just some text"
    assert len(t.calls) == 1
