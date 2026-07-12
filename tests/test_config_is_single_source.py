"""Warn-only lint: config.py should be the single source of gate tunables.

Scans every module under ``src/weeker`` (except config.py) for bare numeric
literals whose value equals one of the gate constants defined in config.py. Any
hit is reported as a warning but does NOT fail the suite — this is a guardrail
to keep magic numbers from drifting out of config, not a hard gate.
"""

from __future__ import annotations

import ast
import warnings
from pathlib import Path

from weeker.core import config

SRC = Path(__file__).resolve().parents[1] / "src" / "weeker"

# Values that legitimately appear as structural literals everywhere.
_ALLOWED = {0, 1, 2, 3, -1, 100, 64, 60, 8, 32, 16, 24, 5, 6}


def _gate_values() -> set[float]:
    values: set[float] = set()
    for name in dir(config):
        if name.startswith("_") or not name.isupper():
            continue
        val = getattr(config, name)
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)):
            values.add(float(val))
        elif isinstance(val, dict):
            for v in val.values():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    values.add(float(v))
    return {v for v in values if v not in _ALLOWED}


def test_no_duplicate_gate_literals_outside_config():
    gate_values = _gate_values()
    findings: list[str] = []

    for path in SRC.rglob("*.py"):
        if path.name == "config.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                if isinstance(node.value, bool):
                    continue
                if float(node.value) in gate_values:
                    findings.append(f"{path.relative_to(SRC)}:{node.lineno} -> {node.value}")

    if findings:
        warnings.warn(
            "gate-valued numeric literals found outside config.py (move to config):\n"
            + "\n".join(findings),
            stacklevel=2,
        )
    # Warn-only: this test always passes.
    assert True
