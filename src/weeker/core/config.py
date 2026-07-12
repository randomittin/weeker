"""Single source of tunables for Weeker.

Every gate/threshold constant lives here and NOWHERE else (016 §T-03). Change a
value here, not at a call site. ``MODEL_*`` and ``EMBED_*`` are read from the
environment so providers/keys never get hard-coded; the documented defaults are
the ``…`` placeholders (models unset) and a conventional embedding dimension.
"""

from __future__ import annotations

import os

# ── ingestion ────────────────────────────────────────────────────────────────
CHUNK_TARGET_TOKENS = 900
CHUNK_MIN = 300
CHUNK_MAX = 1400
CHUNK_OVERLAP = 0.15
HEADER_FOOTER_PAGE_SHARE = 0.30
OCR_MIN_CHARS = 40
OCR_DPI = 200
CONCEPT_MERGE_COS = 0.92
CONCEPT_CHUNK_MIN_COS = 0.60
CONCEPT_CHUNK_TOPK = 6

# ── generation gates ───────────────────────────────────────────────────────────
GEN_BATCH = 5
GEN_OVERGEN_FACTOR = 1.3
G4_NGRAM = 8
G4_STEM_CHUNK_MAX_COS = 0.90
G5_DUP_COS = 0.90
G5_VARIANT_COS = 0.83
OPTION_DISTINCT_MAX_COS = 0.85

# ── adaptive (013) ─────────────────────────────────────────────────────────────
ELO_K_BASE = 0.6
ELO_K_MIN = 0.08
ELO_K_MAX = 0.6
ELO_KQ_BASE = 0.15
CONF_MULT = {
    ("correct", "sure"): 1.0,
    ("correct", "unsure"): 0.7,
    ("correct", "guessing"): 0.25,
    ("wrong", "sure"): 1.6,
    ("wrong", "unsure"): 1.0,
    ("wrong", "guessing"): 0.6,
}
MASTERY_TARGET_LOGIT = 0.3
STAB_GROW_BASE = 1.4
STAB_GROW_MASTERY = 1.8
STAB_FAIL_MULT = 0.35
STAB_MIN = 0.5
DUE_R_THRESHOLD = 0.85
PRIORITY_W = dict(weak=0.45, overdue=0.30, misconception=0.20, novelty=0.05)
SOFTMAX_T = 0.7
DESIRABLE_DIFFICULTY_OFFSET = 0.3
RESERVE_DAYS = 3
SESSION_MIX = dict(warmup=3, core=21, mixed=4, review=2)
PREDICT_SE_BASE = 1.2
PREDICT_BOOTSTRAP = 1000
PREDICT_MIN_COVERAGE = 0.30

# ── models (env-overridable; defaults are the documented "…" placeholders) ─────
MODEL_GENERATOR = os.getenv("WEEKER_MODEL_GENERATOR", "…")
MODEL_VERIFIER_CHEAP = os.getenv("WEEKER_MODEL_VERIFIER_CHEAP", "…")
MODEL_SOLVER_BLIND = os.getenv("WEEKER_MODEL_SOLVER_BLIND", "…-different-family")
MODEL_EMBED = os.getenv("WEEKER_MODEL_EMBED", "…")
EMBED_DIM = int(os.getenv("WEEKER_EMBED_DIM", "1536"))

# ── LLM / embed transport (env-overridable) ────────────────────────────────────
LLM_BASE_URL = os.getenv("WEEKER_LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("WEEKER_LLM_API_KEY", "")
EMBED_BASE_URL = os.getenv("WEEKER_EMBED_BASE_URL", "https://api.openai.com/v1")
EMBED_API_KEY = os.getenv("WEEKER_EMBED_API_KEY", "")
EMBED_BATCH = int(os.getenv("WEEKER_EMBED_BATCH", "64"))
CACHE_DIR = os.getenv("WEEKER_CACHE_DIR", ".weeker_cache")


def models_configured() -> bool:
    """True when the model identifiers have been set away from their placeholders."""
    return all(
        m and "…" not in m
        for m in (MODEL_GENERATOR, MODEL_VERIFIER_CHEAP, MODEL_SOLVER_BLIND, MODEL_EMBED)
    )


def llm_key_present() -> bool:
    """True when an LLM API key is configured in the environment."""
    return bool(LLM_API_KEY)


def embed_key_present() -> bool:
    """True when an embedding API key is configured in the environment."""
    return bool(EMBED_API_KEY)
