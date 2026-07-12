"""T-24 — disputed-question review TUI.

A blind-solver dispute (G3) means the solver's answer differs from the authored
key: either the key is wrong or the question is genuinely hard. This Textual app
shows both sides — the keyed answer + its authored rationale (the explanation)
against the blind solver's vote(s) + rationale(s) recorded in ``gate_log['G3']`` —
and lets the reviewer resolve each one:

* ``k`` — **keep** the key as authored → the question goes ``active``;
* ``1``/``2``/``3``/``4`` — **fix key** to A/B/C/D → the question goes ``active``
  with the corrected key;
* ``x`` — **discard** → the question is ``retired``.

Every resolution is recorded under ``gate_log['dispute']`` and committed, so the
queue is resumable. ``weeker generate review`` launches it over a course's
disputed bank.
"""

from __future__ import annotations

from datetime import UTC, datetime

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Static

from weeker.core.db import get_session
from weeker.core.models import Question

_FIX_KEYS = {"1": "A", "2": "B", "3": "C", "4": "D"}


class DisputeApp(App):
    """Side-by-side dispute resolution over a list of disputed questions."""

    CSS = """
    Screen { layout: vertical; }
    #progress { color: $accent; height: 1; }
    #stem { padding: 1 0; text-style: bold; }
    #panels { height: auto; }
    #keyed { width: 1fr; padding: 0 1; color: $success; }
    #solver { width: 1fr; padding: 0 1; color: $warning; }
    #prompt { color: $text-muted; padding: 1 0; }
    """

    def __init__(self, questions, *, db=None, now: datetime | None = None):
        super().__init__()
        self.questions = list(questions)
        self.now = now or datetime.now(UTC)
        self._own_db = db is None
        self._db = db or get_session()
        self.index = 0
        self.resolutions: list[tuple[object, str]] = []
        self.phase = "review" if self.questions else "done"

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(id="progress")
            yield Static(id="stem")
            with Horizontal(id="panels"):
                yield Static(id="keyed")
                yield Static(id="solver")
            yield Static(id="prompt")

    def on_mount(self) -> None:
        self._render()

    def _set(self, wid: str, text: str) -> None:
        self.query_one(f"#{wid}", Static).update(text)

    @property
    def current(self) -> Question | None:
        if 0 <= self.index < len(self.questions):
            return self.questions[self.index]
        return None

    def _option_text(self, q: Question, key: str) -> str:
        for o in q.options or []:
            if o.get("key") == key:
                return str(o.get("text", ""))
        return ""

    def _render(self) -> None:
        if self.phase == "done":
            self._set("progress", "")
            self._set("stem", f"Dispute review complete — {len(self.resolutions)} resolved.")
            self._set("keyed", "")
            self._set("solver", "")
            self._set("prompt", "Press q to exit.")
            return
        q = self.current
        self._set("progress", f"Dispute {self.index + 1}/{len(self.questions)}")
        opts = "\n".join(f"  {o.get('key')}) {o.get('text', '')}" for o in q.options or [])
        self._set("stem", f"{q.stem}\n{opts}")

        g3 = (q.gate_log or {}).get("G3", {})
        votes = g3.get("votes", [])
        rationales = g3.get("solver_rationales", [])
        keyed = [
            f"KEYED ANSWER: {q.correct_key}) {self._option_text(q, q.correct_key)}",
            "",
            q.explanation or "(no authored rationale)",
        ]
        solver = [f"BLIND SOLVER votes: {votes}", ""]
        solver += [f"- {r}" for r in rationales] or ["(no rationale recorded)"]
        self._set("keyed", "\n".join(keyed))
        self._set("solver", "\n".join(solver))
        self._set(
            "prompt",
            "k keep key · 1-4 fix key to A-D · x discard · q quit",
        )

    def _resolve(self, resolution: str, new_key: str | None = None) -> None:
        q = self.current
        log = dict(q.gate_log or {})
        dispute = {"resolution": resolution, "resolved_at": self.now.isoformat()}
        if resolution == "keep":
            q.status = "active"
        elif resolution == "fix-key":
            dispute["old_key"] = q.correct_key
            dispute["new_key"] = new_key
            q.correct_key = new_key
            q.status = "active"
        elif resolution == "discard":
            q.status = "retired"
        log["dispute"] = dispute
        q.gate_log = log
        self._db.add(q)
        self._db.flush()
        self.resolutions.append((q.id, resolution))
        self.index += 1
        if self.index >= len(self.questions):
            self.phase = "done"
            if self._own_db:
                self._db.commit()
        self._render()

    def on_key(self, event) -> None:
        key = event.key
        if self.phase == "done":
            if key in ("q", "escape", "enter"):
                self.exit()
            return
        if key == "k":
            self._resolve("keep")
        elif key == "x":
            self._resolve("discard")
        elif key in _FIX_KEYS:
            self._resolve("fix-key", new_key=_FIX_KEYS[key])

    def on_unmount(self) -> None:
        if self._own_db:
            try:
                self._db.commit()
            finally:
                self._db.close()
