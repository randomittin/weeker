"""``weeker serve`` runner — boots uvicorn on the FastAPI app and prints the URL.

Kept out of :mod:`weeker.cli.main` import time so the base CLI never imports
fastapi/uvicorn (the ``web`` extra) unless the learner actually runs ``serve``.
"""

from __future__ import annotations


def run_server(*, host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Run the Weeker web/API server, printing the URL it binds to."""
    import uvicorn

    url_host = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    print(f"Weeker running at http://{url_host}:{port}  (API under /api, docs at /docs)")
    # An import string is required for reload; the app object works otherwise.
    target = "weeker.api.app:app" if reload else _load_app()
    uvicorn.run(target, host=host, port=port, reload=reload)


def _load_app():
    from weeker.api.app import app

    return app
