"""T-22 gate: G1 / G4 / G5 pure-code gates with crafted violations.

Each crafted violation must fail the *right* gate and record its reason into a
gate_log dict: a 3-option question and a two-correct question fail G1; a verbatim
8-gram lift and an over-close stem fail G4; a 0.91-cos near-dupe fails G5 while a
0.85-cos stem is tagged a variant (and passes).
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from weeker.core.config import (
    G4_STEM_CHUNK_MAX_COS,
    G5_DUP_COS,
    G5_VARIANT_COS,
    OPTION_DISTINCT_MAX_COS,
)
from weeker.generate import gates

_WORD = re.compile(r"[a-z0-9]+")


def _bow(text: str, dim: int = 64) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    for w in _WORD.findall(text.lower()):
        v[int(hashlib.sha256(w.encode()).hexdigest()[:8], 16) % dim] += 1.0
    return v


def _bow_embed(texts):
    return np.vstack([_bow(t) for t in texts])


def _opts(n=4, dup_key=False):
    keys = ["A", "B", "C", "D"][:n]
    if dup_key:
        keys[1] = "A"
    return [{"key": k, "text": f"option {k}"} for k in keys]


def test_g1_passes_a_clean_question():
    log: dict = {}
    res = gates.check_g1(
        stem="What is the NAV?",
        options=_opts(),
        correct_key="A",
        embed_fn=_bow_embed,
        gate_log=log,
    )
    assert res.passed
    assert log["G1"]["passed"] is True


def test_g1_three_options_fails():
    log: dict = {}
    res = gates.check_g1(stem="x?", options=_opts(3), correct_key="A", gate_log=log)
    assert not res.passed
    assert "4 options" in res.reason
    assert log["G1"]["passed"] is False


def test_g1_two_correct_fails():
    log: dict = {}
    opts = _opts()
    opts[0]["correct"] = True
    opts[1]["correct"] = True  # two flagged correct
    res = gates.check_g1(stem="x?", options=opts, correct_key="A", gate_log=log)
    assert not res.passed
    assert "one correct" in res.reason.lower()


def test_g1_all_of_the_above_banned():
    log: dict = {}
    opts = _opts()
    opts[3]["text"] = "All of the above"
    res = gates.check_g1(stem="x?", options=opts, correct_key="A", gate_log=log)
    assert not res.passed
    assert "above" in res.reason.lower()


def test_g1_non_distinct_options_fail():
    log: dict = {}
    opts = _opts()
    opts[0]["text"] = "the net asset value per unit of the fund"
    opts[1]["text"] = "the net asset value per unit of the fund today"  # near-identical
    res = gates.check_g1(
        stem="x?", options=opts, correct_key="A", embed_fn=_bow_embed, gate_log=log
    )
    assert not res.passed
    assert "distinct" in res.reason.lower()
    # sanity: those two options really are above the distinctness ceiling
    m = _bow_embed([opts[0]["text"], opts[1]["text"]])
    from weeker.ingest.textutil import cosine

    assert cosine(m[0], m[1]) >= OPTION_DISTINCT_MAX_COS


def test_g1_identical_option_text_fails_without_embedder():
    """Two options with identical text (modulo case/whitespace) are a hard defect,
    caught by the model-independent duplicate check even with no embedder."""
    log: dict = {}
    opts = _opts()
    opts[0]["text"] = "About 4% per annum"
    opts[1]["text"] = "  about   4%   PER Annum  "  # same text, different case/spacing
    res = gates.check_g1(stem="x?", options=opts, correct_key="A", gate_log=log)
    assert not res.passed
    assert "identical" in res.reason.lower()


def test_g1_sign_and_amount_distinct_options_pass_without_embedder():
    """Distinct-but-templated options must PASS the pure-structural G1 (no embedder):
    the duplicate check preserves signs/punctuation, so '4%' vs '-4%' and 'Rs. 5
    lakhs' vs 'Rs. 10 lakhs' are not collapsed into duplicates."""
    for a, b in (("About 4% per annum.", "About -4% per annum."),
                 ("Rs. 5 lakhs", "Rs. 10 lakhs")):
        log: dict = {}
        opts = _opts()
        opts[0]["text"] = a
        opts[1]["text"] = b
        res = gates.check_g1(stem="x?", options=opts, correct_key="A", gate_log=log)
        assert res.passed, f"{a!r} vs {b!r} wrongly flagged: {res.reason}"


def test_g4_verbatim_eight_gram_lift_fails():
    corpus = "the net asset value of a mutual fund is computed once every business day"
    idx = gates.build_ngram_index([corpus])
    log: dict = {}
    stem = "Explain how the net asset value of a mutual fund is computed once daily."
    res = gates.check_g4(stem, idx, gate_log=log)
    assert not res.passed
    assert "8-gram" in res.reason
    assert log["G4"]["passed"] is False


def test_g4_original_stem_passes():
    idx = gates.build_ngram_index(["completely unrelated corpus text about taxation rules"])
    res = gates.check_g4("How do you value a fund unit given assets and liabilities?", idx, gate_log={})
    assert res.passed


def test_g4_stem_too_close_to_chunk_fails():
    idx = gates.build_ngram_index(["nothing overlaps here at all friend"])
    stem = "How is net asset value computed for a fund unit daily basis"
    # A chunk that is a near-verbatim paraphrase of the stem sits above the cosine
    # ceiling even though it shares no full 8-gram window with the corpus index.
    chunk = "How is net asset value computed for a fund unit daily basis"
    stem_vec = _bow(stem)
    chunk_matrix = _bow_embed([chunk])
    log: dict = {}
    res = gates.check_g4(stem, idx, stem_vec=stem_vec, chunk_matrix=chunk_matrix, gate_log=log)
    assert not res.passed
    assert "cos" in res.reason.lower()
    from weeker.ingest.textutil import cosine_matrix

    assert float(cosine_matrix(stem_vec, chunk_matrix).max()) >= G4_STEM_CHUNK_MAX_COS


def test_g5_near_duplicate_fails():
    base = "How is the net asset value of a fund computed on a daily basis"
    dupe = "How is the net asset value of a fund computed on a daily basis today"
    stem_vec = _bow(dupe)
    existing = [("q-old", _bow(base))]
    log: dict = {}
    res = gates.check_g5(stem_vec, existing, gate_log=log)
    from weeker.ingest.textutil import cosine

    assert cosine(stem_vec, existing[0][1]) >= G5_DUP_COS
    assert not res.passed
    assert log["G5"]["passed"] is False


def test_g5_variant_is_tagged_and_passes():
    # Construct two vectors with a controlled cosine between the variant and dup ceilings.
    a = np.zeros(8, dtype=np.float32)
    a[0] = 1.0
    b = np.zeros(8, dtype=np.float32)
    b[0] = 0.84
    b[1] = np.sqrt(1 - 0.84**2)  # cosine(a,b) == 0.84 ∈ [VARIANT, DUP)
    from weeker.ingest.textutil import cosine

    assert G5_VARIANT_COS <= cosine(a, b) < G5_DUP_COS
    log: dict = {}
    res = gates.check_g5(b, [("q-orig", a)], gate_log=log)
    assert res.passed
    assert log["G5"]["variant_of"] == "q-orig"
    assert log["G5"]["variant_group"] == "q-orig"


def test_g5_distinct_stem_passes_clean():
    a = _bow("a totally different question about capital gains taxation indexation")
    b = _bow("what is the expense ratio of an index fund")
    res = gates.check_g5(a, [("q1", b)], gate_log={})
    assert res.passed
