"""Real OpenAI-compatible response shapes (Gemini/Groq) through llm.complete.

Covers the envelope/JSON extraction the providers actually emit: standard
``choices[0].message.content``, ```json fenced content, prose-wrapped JSON,
multi-part list content, and a 200-status ``{"error": {...}}`` envelope. Also
asserts a >=400 HTTP body surfaces as a clear ``LLMError`` (not a bare status).
"""

from __future__ import annotations

import json

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
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
}


class FakeTransport:
    """Returns a scripted reply per call (mirrors tests/test_llm.py)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, model, system, user):
        self.calls.append(user)
        return self.responses.pop(0)


def _envelope(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


# ── _content_from_envelope: the shapes providers return ────────────────────────
def test_standard_openai_envelope():
    assert _content_from_envelope(_envelope("hello")) == "hello"


def test_multi_part_list_content():
    # Some OpenAI-compatible proxies deliver content as a list of text parts.
    payload = _envelope([{"type": "text", "text": "foo"}, {"type": "text", "text": "bar"}])
    assert _content_from_envelope(payload) == "foobar"


def test_200_status_error_envelope_raises():
    # Gemini/Groq can return HTTP 200 with a JSON {"error": {...}} body.
    payload = {"error": {"code": 429, "message": "rate limit exceeded"}}
    with pytest.raises(LLMError, match="rate limit exceeded"):
        _content_from_envelope(payload)


def test_malformed_envelope_surfaces_body():
    with pytest.raises(LLMError, match="unexpected completion envelope"):
        _content_from_envelope({"nonsense": True})


def test_null_content_raises():
    with pytest.raises(LLMError, match="no content"):
        _content_from_envelope(_envelope(None))


# ── extract_json: fences and prose the models wrap strict JSON in ──────────────
def test_extract_plain_json():
    assert extract_json('{"answer": "A"}') == {"answer": "A"}


def test_extract_fenced_json():
    raw = 'Here is the JSON:\n```json\n{"answer": "B"}\n```\nHope that helps!'
    assert extract_json(raw) == {"answer": "B"}


def test_extract_prose_wrapped_json():
    raw = 'Sure — the answer object is {"answer": "C"} based on the scenario.'
    assert extract_json(raw) == {"answer": "C"}


def test_extract_ignores_braces_inside_strings():
    raw = 'prefix {"answer": "use } and { carefully"} suffix'
    assert extract_json(raw) == {"answer": "use } and { carefully"}


def test_extract_array_span():
    raw = "results below:\n[1, 2, 3]\ndone"
    assert extract_json(raw) == [1, 2, 3]


def test_extract_nothing_raises():
    with pytest.raises(json.JSONDecodeError):
        extract_json("no json here at all")


# ── complete(): end-to-end parse through the fake transport ────────────────────
def test_complete_parses_fenced_content_first_try():
    t = FakeTransport(['```json\n{"answer": "A"}\n```'])
    out = complete("m", "s", "u", json_schema=SCHEMA, transport=t)
    assert out == {"answer": "A"}
    assert len(t.calls) == 1  # no retry needed — fence was tolerated


def test_complete_parses_prose_wrapped_content():
    t = FakeTransport(['The correct choice is {"answer": "D"}.'])
    out = complete("m", "s", "u", json_schema=SCHEMA, transport=t)
    assert out == {"answer": "D"}


# ── HTTP path: provider error body raises a clear error ────────────────────────
def _mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_http_transport_ok_envelope():
    def handler(_req):
        return httpx.Response(200, json=_envelope('{"answer": "A"}'))

    t = HTTPTransport(base_url="https://x/v1", api_key="k", client=_mock_client(handler))
    out = complete("m", "s", "u", json_schema=SCHEMA, transport=t)
    assert out == {"answer": "A"}


def test_http_transport_error_status_surfaces_body():
    def handler(_req):
        return httpx.Response(
            401, json={"error": {"message": "invalid api key", "type": "auth"}}
        )

    t = HTTPTransport(base_url="https://x/v1", api_key="bad", client=_mock_client(handler))
    with pytest.raises(LLMError, match="401.*invalid api key"):
        complete("m", "s", "u", json_schema=SCHEMA, transport=t)
