"""llm.complete: strict-JSON retry against a fake transport."""

from __future__ import annotations

import httpx
import pytest

from weeker.core.llm import (
    HTTPTransport,
    LLMError,
    _content_from_envelope,
    complete,
    extract_json,
)

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


# ── real-provider response shapes (Gemini / Groq via OpenAI-compatible) ─────────
def test_extract_json_fenced_block():
    """Gemini/Groq routinely wrap strict-JSON output in a ```json fence."""
    raw = 'Here is the JSON:\n```json\n{"title": "x", "count": 2}\n```\nHope that helps!'
    assert extract_json(raw) == {"title": "x", "count": 2}


def test_extract_json_bare_fence_no_lang():
    raw = "```\n{\"title\": \"y\", \"count\": 5}\n```"
    assert extract_json(raw) == {"title": "y", "count": 5}


def test_extract_json_prose_wrapped_no_fence():
    raw = 'Sure. The answer object is {"title": "z", "count": 9} — let me know if wrong.'
    assert extract_json(raw) == {"title": "z", "count": 9}


def test_extract_json_ignores_braces_inside_strings():
    raw = 'prefix {"note": "a } inside string", "count": 1} suffix'
    assert extract_json(raw) == {"note": "a } inside string", "count": 1}


def test_extract_json_array_top_level():
    raw = "```json\n[{\"a\": 1}, {\"a\": 2}]\n```"
    assert extract_json(raw) == [{"a": 1}, {"a": 2}]


def test_complete_parses_fenced_response():
    """The retry loop must parse a fenced reply on the FIRST attempt (no re-prompt)."""
    fenced = '```json\n{"title": "ok", "count": 4}\n```'
    t = FakeTransport([fenced])
    out = complete("m", "s", "u", json_schema=SCHEMA, transport=t)
    assert out == {"title": "ok", "count": 4}
    assert len(t.calls) == 1  # parsed cleanly, no retry needed


def test_complete_parses_prose_wrapped_response():
    prose = 'Certainly! Here you go:\n{"title": "p", "count": 7}\nThanks.'
    t = FakeTransport([prose])
    out = complete("m", "s", "u", json_schema=SCHEMA, transport=t)
    assert out == {"title": "p", "count": 7}
    assert len(t.calls) == 1


# ── OpenAI-compatible envelope extraction ───────────────────────────────────────
def test_envelope_standard_shape():
    payload = {"choices": [{"message": {"content": "hello world"}}]}
    assert _content_from_envelope(payload) == "hello world"


def test_envelope_multipart_content():
    """Some proxies deliver content as a list of text parts."""
    payload = {"choices": [{"message": {"content": [{"text": "a"}, {"text": "b"}]}}]}
    assert _content_from_envelope(payload) == "ab"


def test_envelope_error_body_raises():
    payload = {"error": {"message": "rate limit exceeded", "code": 429}}
    with pytest.raises(LLMError, match="rate limit exceeded"):
        _content_from_envelope(payload)


def test_envelope_unexpected_shape_raises():
    with pytest.raises(LLMError, match="unexpected completion envelope"):
        _content_from_envelope({"weird": True})


# ── HTTPTransport end-to-end through a mocked OpenAI-compatible endpoint ─────────
def _mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_http_transport_extracts_fenced_content_from_envelope():
    """Full real path: chat/completions → choices[0].message.content (fenced) → dict."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '```json\n{"title": "t", "count": 1}\n```'}}
                ]
            },
        )

    t = HTTPTransport(base_url="https://api.example/v1", api_key="k", client=_mock_client(handler))
    out = complete("m", "sys", "user", json_schema=SCHEMA, transport=t)
    assert out == {"title": "t", "count": 1}


def test_http_transport_surfaces_error_status_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "too many requests"}})

    t = HTTPTransport(base_url="https://api.example/v1", api_key="k", client=_mock_client(handler))
    with pytest.raises(LLMError, match="429"):
        complete("m", "sys", "user", transport=t)
