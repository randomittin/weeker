"""T-23 gate: G2 grounding verifier + G3 blind-solve consensus / dispute routing.

Fixture transports drive the LLM gates deterministically: G2 supported vs
ambiguous; G3 unanimous-agree → pass, disagree → disputed, 2/3 split → disputed,
3/3 → pass.
"""

from __future__ import annotations

import json

from atlas.generate import gates

_OPTS = [
    {"key": "A", "text": "Assets minus liabilities per unit"},
    {"key": "B", "text": "The market price"},
    {"key": "C", "text": "Total assets only"},
    {"key": "D", "text": "Units outstanding"},
]


class ScriptedLLM:
    """Returns a queued list of raw responses, one per request() call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def request(self, model, system, user):
        self.calls += 1
        return self._responses.pop(0)


def _g2(supported, ambiguous=None, reason="r"):
    return json.dumps(
        {"supported": supported, "ambiguous_option": ambiguous, "reason": reason}
    )


def _solve(answer, confidence="sure", rationale="because"):
    return json.dumps({"answer": answer, "confidence": confidence, "rationale": rationale})


def test_g2_supported_passes():
    log: dict = {}
    res = gates.check_g2(
        stem="What is NAV?",
        options=_OPTS,
        correct_key="A",
        chunk_texts=["NAV = assets minus liabilities / units."],
        transport=ScriptedLLM([_g2(True, None)]),
        model="verify-test",
        gate_log=log,
    )
    assert res.passed
    assert log["G2"]["supported"] is True
    assert log["G2"]["ambiguous_option"] is None


def test_g2_ambiguous_option_fails():
    log: dict = {}
    res = gates.check_g2(
        stem="What is NAV?",
        options=_OPTS,
        correct_key="A",
        chunk_texts=["ambiguous source"],
        transport=ScriptedLLM([_g2(True, "C", "C also defensible")]),
        model="verify-test",
        gate_log=log,
    )
    assert not res.passed
    assert log["G2"]["ambiguous_option"] == "C"


def test_g2_unsupported_fails():
    log: dict = {}
    res = gates.check_g2(
        stem="What is NAV?",
        options=_OPTS,
        correct_key="A",
        chunk_texts=["no support for the key"],
        transport=ScriptedLLM([_g2(False, None, "not in source")]),
        model="verify-test",
        gate_log=log,
    )
    assert not res.passed
    assert log["G2"]["supported"] is False


def test_g3_unanimous_agreement_passes():
    log: dict = {}
    res = gates.check_g3(
        stem="What is NAV?",
        options=_OPTS,
        correct_key="A",
        consensus=1,
        transport=ScriptedLLM([_solve("A")]),
        model="solver-test",
        gate_log=log,
    )
    assert res.passed
    assert log["G3"]["disputed"] is False


def test_g3_disagreement_disputes():
    log: dict = {}
    res = gates.check_g3(
        stem="What is NAV?",
        options=_OPTS,
        correct_key="A",
        consensus=1,
        transport=ScriptedLLM([_solve("B")]),
        model="solver-test",
        gate_log=log,
    )
    assert not res.passed
    assert log["G3"]["disputed"] is True
    assert log["G3"]["votes"] == ["B"]
    assert log["G3"]["solver_rationales"]


def test_g3_two_of_three_split_disputes():
    log: dict = {}
    res = gates.check_g3(
        stem="What is NAV?",
        options=_OPTS,
        correct_key="A",
        consensus=3,
        transport=ScriptedLLM([_solve("A"), _solve("A"), _solve("B")]),
        model="solver-test",
        gate_log=log,
    )
    assert not res.passed
    assert log["G3"]["disputed"] is True
    assert log["G3"]["votes"] == ["A", "A", "B"]


def test_g3_three_of_three_passes():
    log: dict = {}
    res = gates.check_g3(
        stem="What is NAV?",
        options=_OPTS,
        correct_key="A",
        consensus=3,
        transport=ScriptedLLM([_solve("A"), _solve("A"), _solve("A")]),
        model="solver-test",
        gate_log=log,
    )
    assert res.passed
    assert log["G3"]["disputed"] is False


def test_g3_scenario_is_passed_to_solver():
    seen: dict = {}

    class Capture:
        def request(self, model, system, user):
            seen["user"] = user
            return _solve("A")

    gates.check_g3(
        stem="Given the case, what is the tax?",
        options=_OPTS,
        correct_key="A",
        consensus=1,
        scenario="Investor Priya, age 35, holds equity funds for 18 months.",
        transport=Capture(),
        model="solver-test",
        gate_log={},
    )
    assert "Priya" in seen["user"]
