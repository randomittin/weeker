"""Local FastAPI backend for Weeker — a thin web layer over the study engine.

The package holds the FastAPI app (:mod:`weeker.api.app`), the request/response
schemas (:mod:`weeker.api.schemas`) and the ``weeker serve`` runner
(:mod:`weeker.api.serve`). It reuses the engine services in
:mod:`weeker.learn` verbatim — no learning logic is duplicated here — and never
returns a question's ``correct_key`` before the learner has answered.
"""
