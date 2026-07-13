"""Keyless loaders for hand-authored course content.

:mod:`weeker.seed.load_authored` ingests hand-authored concept/question/caselet
JSON (the ``courses/<course>/authored/chNN.json`` files) straight into the DB,
running the same PURE-CODE gates (G1/G4/G5) the generation pipeline runs. The
LLM gates (G2/G3) are skipped — authored content is pre-grounded — and marked as
such in each ``gate_log``.
"""

from __future__ import annotations

from weeker.seed.load_authored import ChapterLoad, LoadReport, load_authored

__all__ = ["ChapterLoad", "LoadReport", "load_authored"]
