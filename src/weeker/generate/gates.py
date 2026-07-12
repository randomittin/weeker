"""Generation gate chain (012 §3).

Gates are the wall between a *generated* question and an *active* one. This module
holds the pure-code gates (G1/G4/G5); the LLM gates (G2/G3) live alongside them in
:mod:`weeker.generate.gates` too (added by T-23). Every gate returns a
:class:`GateResult` and records ``{"passed", "reason", ...}`` into the question's
``gate_log`` under its own key, so a rejection is always auditable.

* **G1 — structural invariants**: exactly 4 options keyed A–D, exactly one correct,
  the "all/none of the above" ban, and option distinctness (no two options within
  :data:`~weeker.core.config.OPTION_DISTINCT_MAX_COS`).
* **G4 — anti-plagiarism**: no verbatim :data:`~weeker.core.config.G4_NGRAM`-gram
  lifted from the (OCR-excluded) course corpus, and no stem within
  :data:`~weeker.core.config.G4_STEM_CHUNK_MAX_COS` cosine of a source chunk.
* **G5 — dedup / variant**: reject a stem within
  :data:`~weeker.core.config.G5_DUP_COS` of an existing stem; tag one within
  :data:`~weeker.core.config.G5_VARIANT_COS` as a variant so the scheduler can
  avoid serving near-twins together.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from weeker.core import config
from weeker.core.llm import Transport, complete
from weeker.ingest.textutil import cosine_matrix, load_prompt, render_prompt

_WORD = re.compile(r"[a-z0-9]+")
_BANNED_OPTION_PHRASES = ("all of the above", "none of the above")
OPTION_KEYS = ("A", "B", "C", "D")


@dataclass
class GateResult:
    """Outcome of one gate: pass/fail, a human reason, and any tags to persist."""

    gate: str
    passed: bool
    reason: str
    tags: dict = field(default_factory=dict)

    def record(self, gate_log: dict) -> GateResult:
        """Write this result under ``gate_log[self.gate]`` and return self."""
        gate_log[self.gate] = {"passed": self.passed, "reason": self.reason, **self.tags}
        return self


def _normalize_words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


# ── G1 ──────────────────────────────────────────────────────────────────────────
def check_g1(
    *,
    stem: str,
    options: Sequence[dict],
    correct_key: str,
    embed_fn: Callable[[list[str]], np.ndarray] | None = None,
    gate_log: dict,
) -> GateResult:
    """Structural invariants + option distinctness. Records into ``gate_log['G1']``."""
    if len(options) != 4:
        return GateResult("G1", False, f"must have exactly 4 options, got {len(options)}").record(
            gate_log
        )

    keys = [str(o.get("key", "")) for o in options]
    if sorted(keys) != list(OPTION_KEYS):
        return GateResult(
            "G1", False, f"option keys must be exactly A,B,C,D distinct, got {keys}"
        ).record(gate_log)

    # Exactly one correct: an explicit per-option 'correct' flag (if present) must be
    # singular and agree with correct_key; otherwise correct_key alone names the answer.
    flagged = [str(o.get("key")) for o in options if o.get("correct")]
    if flagged:
        if len(flagged) != 1:
            return GateResult(
                "G1", False, f"exactly one correct option required, {len(flagged)} flagged"
            ).record(gate_log)
        if flagged[0] != correct_key:
            return GateResult(
                "G1", False, f"flagged-correct {flagged[0]} disagrees with correct_key {correct_key}"
            ).record(gate_log)
    if correct_key not in keys:
        return GateResult(
            "G1", False, f"correct_key {correct_key} not among option keys"
        ).record(gate_log)

    for o in options:
        norm = " ".join(_normalize_words(str(o.get("text", ""))))
        for banned in _BANNED_OPTION_PHRASES:
            if banned in norm:
                return GateResult(
                    "G1", False, f"banned option phrase '{banned}' in option {o.get('key')}"
                ).record(gate_log)

    if not str(stem).strip():
        return GateResult("G1", False, "stem must be non-empty").record(gate_log)

    if embed_fn is not None:
        texts = [str(o.get("text", "")) for o in options]
        matrix = np.asarray(embed_fn(texts), dtype=np.float32)
        for i in range(len(texts)):
            sims = cosine_matrix(matrix[i], matrix)
            for j in range(len(texts)):
                if i != j and float(sims[j]) >= config.OPTION_DISTINCT_MAX_COS:
                    return GateResult(
                        "G1",
                        False,
                        f"options {keys[i]}/{keys[j]} not distinct "
                        f"(cos={float(sims[j]):.3f} ≥ {config.OPTION_DISTINCT_MAX_COS})",
                    ).record(gate_log)

    return GateResult("G1", True, "ok").record(gate_log)


# ── G4 ──────────────────────────────────────────────────────────────────────────
def _ngrams(words: list[str], n: int) -> set[str]:
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def build_ngram_index(texts: Sequence[str], n: int = config.G4_NGRAM) -> set[str]:
    """Normalized ``n``-gram set over a corpus (feed it OCR-excluded chunk text)."""
    index: set[str] = set()
    for t in texts:
        index |= _ngrams(_normalize_words(t), n)
    return index


def check_g4(
    stem: str,
    ngram_index: set[str],
    *,
    stem_vec: np.ndarray | None = None,
    chunk_matrix: np.ndarray | None = None,
    n: int = config.G4_NGRAM,
    gate_log: dict,
) -> GateResult:
    """Anti-plagiarism: verbatim n-gram lift + stem↔chunk cosine ceiling."""
    words = _normalize_words(stem)
    hits = _ngrams(words, n) & ngram_index
    if hits:
        return GateResult(
            "G4", False, f"verbatim {n}-gram lift from source: '{sorted(hits)[0]}'"
        ).record(gate_log)

    if stem_vec is not None and chunk_matrix is not None and np.asarray(chunk_matrix).size:
        sims = cosine_matrix(np.asarray(stem_vec, dtype=np.float32), np.asarray(chunk_matrix))
        top = float(sims.max()) if sims.size else 0.0
        if top >= config.G4_STEM_CHUNK_MAX_COS:
            return GateResult(
                "G4",
                False,
                f"stem too close to source chunk (cos={top:.3f} ≥ {config.G4_STEM_CHUNK_MAX_COS})",
            ).record(gate_log)

    return GateResult("G4", True, "ok").record(gate_log)


# ── G5 ──────────────────────────────────────────────────────────────────────────
def check_g5(
    stem_vec: np.ndarray,
    existing: Sequence[tuple[object, np.ndarray]],
    *,
    gate_log: dict,
) -> GateResult:
    """Dedup + variant tagging against existing stem vectors.

    ``existing`` is ``(id, vector)`` pairs. A max cosine ≥ ``G5_DUP_COS`` rejects
    the stem as a near-duplicate; a max cosine in ``[G5_VARIANT_COS, G5_DUP_COS)``
    passes but tags the variant group (the matched stem's id) so the scheduler
    keeps variants apart.
    """
    if not existing:
        return GateResult("G5", True, "no existing stems to compare").record(gate_log)

    matrix = np.vstack([np.asarray(v, dtype=np.float32) for _, v in existing])
    sims = cosine_matrix(np.asarray(stem_vec, dtype=np.float32), matrix)
    top_i = int(np.argmax(sims))
    top = float(sims[top_i])
    match_id = existing[top_i][0]

    if top >= config.G5_DUP_COS:
        return GateResult(
            "G5",
            False,
            f"near-duplicate of {match_id} (cos={top:.3f} ≥ {config.G5_DUP_COS})",
            tags={"duplicate_of": str(match_id)},
        ).record(gate_log)

    if top >= config.G5_VARIANT_COS:
        return GateResult(
            "G5",
            True,
            f"variant of {match_id} (cos={top:.3f} ≥ {config.G5_VARIANT_COS})",
            tags={"variant_of": str(match_id), "variant_group": str(match_id)},
        ).record(gate_log)

    return GateResult("G5", True, f"distinct (max cos={top:.3f})").record(gate_log)


# ── G2 (LLM grounding verifier) ─────────────────────────────────────────────────
G2_SCHEMA: dict = {
    "type": "object",
    "required": ["supported", "ambiguous_option", "reason"],
    "properties": {
        "supported": {"type": "boolean"},
        "ambiguous_option": {"type": ["string", "null"]},
        "reason": {"type": "string"},
    },
}


def _options_block(options: Sequence[dict]) -> str:
    """Render options as ``A) text`` lines with no correct-answer flag."""
    return "\n".join(f"{o.get('key')}) {o.get('text', '')}" for o in options)


def check_g2(
    *,
    stem: str,
    options: Sequence[dict],
    correct_key: str,
    chunk_texts: Sequence[str],
    model: str | None = None,
    transport: Transport | None = None,
    gate_log: dict,
) -> GateResult:
    """Grounding verifier (``prompts/grounding_verify.txt``).

    Passes only when the keyed answer is fully supported by the source *and* no
    other option is defensibly correct (``ambiguous_option`` is null).
    """
    system, user_t = load_prompt("grounding_verify")
    user = render_prompt(
        user_t,
        chunks="\n\n".join(chunk_texts) or "(no source provided)",
        stem=stem,
        options_with_keys_no_correct_flag=_options_block(options),
        correct_key=correct_key,
    )
    data = complete(
        model or config.MODEL_VERIFIER_CHEAP,
        system,
        user,
        json_schema=G2_SCHEMA,
        transport=transport,
    )
    assert isinstance(data, dict)
    supported = bool(data.get("supported"))
    ambiguous = data.get("ambiguous_option")
    if isinstance(ambiguous, str) and ambiguous.strip().lower() in ("null", "none", ""):
        ambiguous = None
    reason = str(data.get("reason", ""))
    passed = supported and ambiguous is None
    text = (
        "grounded, no ambiguity"
        if passed
        else (f"unsupported: {reason}" if not supported else f"option {ambiguous} also defensible")
    )
    return GateResult(
        "G2",
        passed,
        text,
        tags={"supported": supported, "ambiguous_option": ambiguous},
    ).record(gate_log)


# ── G3 (LLM blind solve + consensus / dispute routing) ──────────────────────────
G3_SCHEMA: dict = {
    "type": "object",
    "required": ["answer", "confidence", "rationale"],
    "properties": {
        "answer": {"type": "string", "enum": list(OPTION_KEYS)},
        "confidence": {"type": "string", "enum": ["sure", "unsure", "guessing"]},
        "rationale": {"type": "string"},
    },
}

_BLIND_SOLVE_SYSTEM = (
    "You are an expert exam candidate. Answer the multiple-choice question using "
    "ONLY your own knowledge and any scenario provided. Do not guess a pattern from "
    "the option wording. Output STRICT JSON: "
    '{"answer":"A|B|C|D","confidence":"sure|unsure|guessing","rationale":"..."}'
)


@dataclass(frozen=True)
class BlindSolve:
    """One blind-solver verdict on a question."""

    answer: str
    confidence: str
    rationale: str


def blind_solve(
    *,
    stem: str,
    options: Sequence[dict],
    scenario: str | None = None,
    model: str | None = None,
    transport: Transport | None = None,
) -> BlindSolve:
    """Solve one question blind (no keyed answer shown) on the solver model."""
    parts = []
    if scenario:
        parts.append(f"Scenario:\n{scenario}\n")
    parts.append(f"Question:\n{stem}\n")
    parts.append("Options:\n" + _options_block(options))
    data = complete(
        model or config.MODEL_SOLVER_BLIND,
        _BLIND_SOLVE_SYSTEM,
        "\n".join(parts),
        json_schema=G3_SCHEMA,
        transport=transport,
    )
    assert isinstance(data, dict)
    return BlindSolve(
        answer=str(data["answer"]).strip().upper(),
        confidence=str(data["confidence"]),
        rationale=str(data.get("rationale", "")),
    )


def check_g3(
    *,
    stem: str,
    options: Sequence[dict],
    correct_key: str,
    consensus: int = 1,
    scenario: str | None = None,
    model: str | None = None,
    transport: Transport | None = None,
    gate_log: dict,
) -> GateResult:
    """Blind-solve ``consensus`` times; pass iff every solve agrees with the key.

    Any disagreement — a lone dissent, a 2/3 split, or unanimous disagreement —
    routes the question to ``disputed`` for human adjudication (the keyed rationale
    vs. the blind solver's rationale). Records the votes and rationales so the
    dispute TUI can show both sides.
    """
    solves = [
        blind_solve(
            stem=stem,
            options=options,
            scenario=scenario,
            model=model,
            transport=transport,
        )
        for _ in range(max(1, consensus))
    ]
    votes = [s.answer for s in solves]
    unanimous_key = all(v == correct_key for v in votes)
    if unanimous_key:
        return GateResult(
            "G3",
            True,
            f"blind solver agrees with key ({len(votes)}/{len(votes)})",
            tags={"disputed": False, "votes": votes},
        ).record(gate_log)
    return GateResult(
        "G3",
        False,
        f"blind solver disagreement, votes={votes} vs key {correct_key} → disputed",
        tags={
            "disputed": True,
            "votes": votes,
            "solver_rationales": [s.rationale for s in solves],
            "solver_confidences": [s.confidence for s in solves],
        },
    ).record(gate_log)
