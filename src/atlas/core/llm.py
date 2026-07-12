"""Thin LLM client with strict-JSON retry.

``complete`` returns a parsed ``dict`` when ``json_schema`` is given (re-prompting
with the validation error appended on malformed/invalid output), or the raw text
otherwise. The transport is injectable: tests pass a fake implementing the
:class:`Transport` protocol; the default :class:`HTTPTransport` talks to an
OpenAI-compatible chat-completions endpoint.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

import httpx

from atlas.core import config


class LLMError(RuntimeError):
    """Raised when completion fails after exhausting retries."""


@runtime_checkable
class Transport(Protocol):
    """Minimal completion transport: given a model + prompts, return text."""

    def request(self, model: str, system: str, user: str) -> str: ...


class HTTPTransport:
    """OpenAI-compatible chat-completions transport (real HTTP path)."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ):
        self.base_url = (base_url or config.LLM_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else config.LLM_API_KEY
        self._client = client
        self._timeout = timeout

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def request(self, model: str, system: str, user: str) -> str:
        resp = self._http().post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


_default_transport: Transport | None = None


def _get_default_transport() -> Transport:
    global _default_transport
    if _default_transport is None:
        _default_transport = HTTPTransport()
    return _default_transport


def validate_json(data: Any, schema: dict) -> list[str]:
    """Validate ``data`` against a small subset of JSON Schema.

    Supports: ``type`` (object/array/string/integer/number/boolean/null),
    ``required``, ``properties``, ``items``, ``enum``. Returns a list of error
    strings ([] means valid).
    """
    return _validate(data, schema, "$")


_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _validate(data: Any, schema: dict, path: str) -> list[str]:
    errs: list[str] = []
    expected = schema.get("type")
    if expected:
        types = expected if isinstance(expected, list) else [expected]
        if not any(_TYPE_CHECKS.get(t, lambda _v: True)(data) for t in types):
            errs.append(f"{path}: expected type {expected}, got {type(data).__name__}")
            return errs

    if "enum" in schema and data not in schema["enum"]:
        errs.append(f"{path}: {data!r} not in enum {schema['enum']}")

    if expected == "object" or (expected is None and isinstance(data, dict)):
        for key in schema.get("required", []):
            if key not in data:
                errs.append(f"{path}: missing required property '{key}'")
        for key, subschema in schema.get("properties", {}).items():
            if key in data:
                errs.extend(_validate(data[key], subschema, f"{path}.{key}"))

    if (expected == "array" or (expected is None and isinstance(data, list))) and "items" in schema:
        for i, item in enumerate(data):
            errs.extend(_validate(item, schema["items"], f"{path}[{i}]"))

    return errs


def complete(
    model: str,
    system: str,
    user: str,
    json_schema: dict | None = None,
    max_retries: int = 2,
    transport: Transport | None = None,
) -> dict | str:
    """Complete a prompt.

    With ``json_schema``: parse + validate the response; on a JSON decode or
    schema error, re-prompt (up to ``max_retries`` extra times) with the error
    appended, and return the parsed ``dict``. Without a schema: return raw text.
    """
    transport = transport or _get_default_transport()
    last_error = "no attempts made"

    for attempt in range(max_retries + 1):
        prompt = user if attempt == 0 else (
            f"{user}\n\nYour previous response was rejected: {last_error}\n"
            f"Return ONLY valid JSON that satisfies the schema."
        )
        raw = transport.request(model, system, prompt)

        if json_schema is None:
            return raw

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            last_error = f"invalid JSON ({e})"
            continue

        errors = validate_json(data, json_schema)
        if errors:
            last_error = "schema validation failed: " + "; ".join(errors)
            continue

        return data

    raise LLMError(f"completion failed after {max_retries + 1} attempts: {last_error}")
