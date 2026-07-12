"""G3 blind solver can target a DIFFERENT provider than the generator.

The ≠-family gate only means something if the blind solve runs on a distinct
model/endpoint. We inject two fake transports tagged by ``base_url`` and assert
the generator + G2 grounding hit ONE endpoint while the G3 blind solve hits the
OTHER — end-to-end through ``run_generation``. Also confirms the default solver
transport is built from ``WEEKER_SOLVER_*`` independently of the LLM endpoint.
"""

from __future__ import annotations

import json

from tests.test_generate_pipeline import BowEmbed, _mem, _seed
from tests.test_learn_fixtures import new_user
from weeker.generate.gates import _BLIND_SOLVE_SYSTEM
from weeker.generate.pipeline import run_generation


class GeneratorLLM:
    """Serves generation + G2 grounding. Records every system prompt it saw."""

    base_url = "https://generator.example/v1"

    def __init__(self):
        self.systems = []
        self.gen_calls = 0

    def _questions(self):
        self.gen_calls += 1
        base = self.gen_calls * 100
        qs = [
            {
                "stem": f"Synthetic probe number {base + i} evaluating reasoning depth clearly",
                "options": [
                    {"key": "A", "text": f"Alpha choice wording {base + i}"},
                    {"key": "B", "text": f"Beta alt {base + i}", "misconception": "beta error"},
                    {"key": "C", "text": f"Gamma opt {base + i}", "misconception": "gamma error"},
                    {"key": "D", "text": f"Delta cand {base + i}", "misconception": "delta error"},
                ],
                "correct_key": "A",
                "explanation": f"Alpha is right for probe {base + i} per the stated reasoning.",
                "difficulty": 2,
            }
            for i in range(5)
        ]
        return json.dumps({"questions": qs})

    def request(self, model, system, user):
        self.systems.append(system)
        if system.startswith("You write ORIGINAL certification"):
            return self._questions()
        if system.startswith("You are a strict verifier"):
            return json.dumps({"supported": True, "ambiguous_option": None, "reason": "ok"})
        raise AssertionError(f"generator endpoint got an unexpected system: {system[:60]!r}")


class SolverLLM:
    """Serves ONLY the G3 blind solve. Records every system prompt it saw."""

    base_url = "https://solver.example/v1"

    def __init__(self, answer="A"):
        self.answer = answer
        self.systems = []

    def request(self, model, system, user):
        self.systems.append(system)
        if not system.startswith("You are an expert exam candidate"):
            raise AssertionError(f"solver endpoint got a non-blind system: {system[:60]!r}")
        return json.dumps(
            {"answer": self.answer, "confidence": "sure", "rationale": "blind reasoning"}
        )


def test_generator_and_solver_hit_distinct_endpoints(tmp_path):
    db = _mem()
    course, _ = _seed(db)
    gen = GeneratorLLM()
    solver = SolverLLM(answer="A")

    report = run_generation(
        db,
        course,
        new_user(),
        count=3,
        llm_transport=gen,
        solver_transport=solver,
        embed_transport=BowEmbed(),
        cache_dir=tmp_path,
    )

    assert report.accepted >= 1
    # Generator handled generation + grounding, never the blind solve.
    assert gen.systems, "generator was never called"
    assert all(not s.startswith("You are an expert exam candidate") for s in gen.systems)
    # Solver handled the blind solve, and ONLY the blind solve.
    assert solver.systems, "solver endpoint was never called — G3 never routed to it"
    assert all(s == _BLIND_SOLVE_SYSTEM for s in solver.systems)


def test_solver_dispute_routes_to_disputed(tmp_path):
    # A solver on a different endpoint that disagrees must still drive disputes.
    db = _mem()
    course, _ = _seed(db)
    report = run_generation(
        db,
        course,
        new_user(),
        count=3,
        llm_transport=GeneratorLLM(),
        solver_transport=SolverLLM(answer="B"),  # never matches key "A"
        embed_transport=BowEmbed(),
        cache_dir=tmp_path,
    )
    assert report.disputed >= 1
    assert report.accepted == 0


def test_default_solver_transport_uses_solver_env(monkeypatch):
    # get_solver_transport() must build from WEEKER_SOLVER_* independently of LLM.
    import importlib

    monkeypatch.setenv("WEEKER_LLM_BASE_URL", "https://llm.example/v1")
    monkeypatch.setenv("WEEKER_LLM_API_KEY", "llm-key")
    monkeypatch.setenv("WEEKER_SOLVER_BASE_URL", "https://groq.example/v1")
    monkeypatch.setenv("WEEKER_SOLVER_API_KEY", "groq-key")
    from weeker.core import config as cfg

    importlib.reload(cfg)
    import weeker.generate.gates as gates

    importlib.reload(gates)
    try:
        t = gates.get_solver_transport()
        assert t.base_url == "https://groq.example/v1"
        assert t.api_key == "groq-key"
        # distinct from the generator/LLM endpoint
        assert t.base_url != cfg.LLM_BASE_URL
    finally:
        # restore module state for the rest of the suite
        monkeypatch.undo()
        importlib.reload(cfg)
        importlib.reload(gates)


def test_solver_falls_back_to_llm_when_unset(monkeypatch):
    import importlib

    for var in ("WEEKER_SOLVER_BASE_URL", "WEEKER_SOLVER_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("WEEKER_LLM_BASE_URL", "https://only-llm.example/v1")
    monkeypatch.setenv("WEEKER_LLM_API_KEY", "shared-key")
    from weeker.core import config as cfg

    importlib.reload(cfg)
    try:
        assert cfg.SOLVER_BASE_URL == "https://only-llm.example/v1"
        assert cfg.SOLVER_API_KEY == "shared-key"
    finally:
        monkeypatch.undo()
        importlib.reload(cfg)
        import weeker.generate.gates as gates

        importlib.reload(gates)
