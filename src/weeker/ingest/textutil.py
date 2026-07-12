"""Shared ingestion text helpers: token counting, prompt loading, cosine.

Kept dependency-free (no tiktoken) so the ingestion gates run on any box. The
token counter is a single deterministic function used everywhere a token budget
is enforced (chunking, histograms) — its exact value is arbitrary but consistent,
which is what the gates rely on.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

# A word or a lone punctuation mark. Approximates a BPE token stream closely
# enough for budget math and, crucially, is stable across runs.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """Deterministic token count for budget/gate math."""
    return len(_TOKEN_RE.findall(text))


def _prompts_dir() -> Path:
    # repo_root/prompts — three parents up from src/weeker/ingest/textutil.py.
    return Path(__file__).resolve().parents[3] / "prompts"


def load_prompt(name: str) -> tuple[str, str]:
    """Load ``prompts/<name>.txt`` split into ``(system, user_template)``.

    The prompt files are stored as ``SYSTEM\\n...\\nUSER\\n...``. Placeholders in
    the user template are ``{like_this}`` but the templates also contain literal
    JSON braces, so callers substitute with :func:`render_prompt` (plain replace),
    never ``str.format``.
    """
    path = _prompts_dir() / f"{name}.txt"
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "SYSTEM":
        raise ValueError(f"prompt {name}: expected first line 'SYSTEM'")
    try:
        u = next(i for i, ln in enumerate(lines) if ln.strip() == "USER")
    except StopIteration as exc:
        raise ValueError(f"prompt {name}: missing 'USER' marker") from exc
    system = "\n".join(lines[1:u]).strip()
    user = "\n".join(lines[u + 1 :]).strip()
    return system, user


def render_prompt(template: str, **values: object) -> str:
    """Substitute ``{key}`` placeholders by literal replacement (brace-safe)."""
    out = template
    for key, val in values.items():
        out = out.replace("{" + key + "}", str(val))
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two 1-D vectors (0.0 if either is zero)."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine of ``query`` (dim,) against each row of ``matrix`` (n, dim)."""
    matrix = np.asarray(matrix, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32)
    if matrix.size == 0:
        return np.empty((0,), dtype=np.float32)
    qn = np.linalg.norm(query) or 1.0
    mn = np.linalg.norm(matrix, axis=1)
    mn[mn == 0] = 1.0
    return (matrix @ query) / (mn * qn)
