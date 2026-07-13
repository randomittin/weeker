"""Embedding client with batching and an on-disk cache.

``embed`` returns an ``(n, dim)`` float32 array in input order. Each text is
cached on disk keyed by ``sha256(text)`` so repeat calls never re-hit the
transport. The transport is injectable (tests pass a fake); the default
:class:`HTTPTransport` calls an OpenAI-compatible embeddings endpoint.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, runtime_checkable

import httpx
import numpy as np

from weeker.core import config


class EmbedError(RuntimeError):
    """Raised when the embeddings transport returns an unusable response."""


@runtime_checkable
class EmbedTransport(Protocol):
    """Embeds a batch of texts into an ``(len(texts), dim)`` array."""

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HTTPTransport:
    """OpenAI-compatible embeddings transport (real HTTP path)."""

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ):
        self.model = model or config.MODEL_EMBED
        self.base_url = (base_url or config.EMBED_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else config.EMBED_API_KEY
        self._client = client
        self._timeout = timeout

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._http().post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": texts},
        )
        if resp.status_code >= 400:
            raise EmbedError(
                f"embed HTTP {resp.status_code} from {self.base_url}: {resp.text[:500]}"
            )
        payload = resp.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not data:
            raise EmbedError(f"unexpected embeddings envelope: {str(payload)[:500]}")
        rows = sorted(data, key=lambda d: d["index"])
        vectors = np.asarray([r["embedding"] for r in rows], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != config.EMBED_DIM:
            got = vectors.shape[1] if vectors.ndim == 2 else vectors.shape
            raise EmbedError(
                f"embedding dim {got} != EMBED_DIM {config.EMBED_DIM} "
                f"(model={self.model!r}); set WEEKER_EMBED_DIM to match the provider"
            )
        return vectors


_LOCAL_MODEL_CACHE: dict[str, object] = {}


class LocalModel2VecTransport:
    """Keyless local embeddings via a numpy-only model2vec static model.

    Loads ``config.EMBED_LOCAL_MODEL`` (default ``minishlab/potion-retrieval-32M``,
    512-dim) once and caches it process-wide. No API key, no torch — just numpy.
    Set ``WEEKER_EMBED_DIM`` to the model's dim (512 for the default model).
    """

    def __init__(self, model: str | None = None):
        self.model = model or config.EMBED_LOCAL_MODEL

    def _load(self) -> object:
        cached = _LOCAL_MODEL_CACHE.get(self.model)
        if cached is not None:
            return cached
        try:
            from model2vec import StaticModel
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise EmbedError(
                "WEEKER_EMBED_PROVIDER=local needs model2vec: pip install model2vec"
            ) from exc
        static = StaticModel.from_pretrained(self.model)
        _LOCAL_MODEL_CACHE[self.model] = static
        return static

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, config.EMBED_DIM), dtype=np.float32)
        vectors = np.asarray(self._load().encode(texts), dtype=np.float32)  # type: ignore[attr-defined]
        if vectors.ndim != 2 or vectors.shape[1] != config.EMBED_DIM:
            got = vectors.shape[1] if vectors.ndim == 2 else vectors.shape
            raise EmbedError(
                f"embedding dim {got} != EMBED_DIM {config.EMBED_DIM} "
                f"(model={self.model!r}); set WEEKER_EMBED_DIM to match the model"
            )
        return vectors


_default_transport: EmbedTransport | None = None


def _get_default_transport() -> EmbedTransport:
    global _default_transport
    if _default_transport is None:
        if config.EMBED_PROVIDER == "local":
            _default_transport = LocalModel2VecTransport()
        else:
            _default_transport = HTTPTransport()
    return _default_transport


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed(
    texts: list[str],
    transport: EmbedTransport | None = None,
    cache_dir: str | Path | None = None,
    batch_size: int | None = None,
) -> np.ndarray:
    """Embed ``texts`` (input order preserved), using the on-disk cache."""
    transport = transport or _get_default_transport()
    cache_dir = Path(cache_dir or Path(config.CACHE_DIR) / "embed")
    cache_dir.mkdir(parents=True, exist_ok=True)
    batch_size = batch_size or config.EMBED_BATCH

    results: list[np.ndarray | None] = [None] * len(texts)
    misses: list[int] = []

    for i, text in enumerate(texts):
        path = cache_dir / f"{_key(text)}.npy"
        if path.exists():
            results[i] = np.load(path)
        else:
            misses.append(i)

    for start in range(0, len(misses), batch_size):
        batch_idx = misses[start : start + batch_size]
        batch_texts = [texts[i] for i in batch_idx]
        vectors = transport.embed(batch_texts)
        for j, i in enumerate(batch_idx):
            vec = np.asarray(vectors[j], dtype=np.float32)
            results[i] = vec
            np.save(cache_dir / f"{_key(texts[i])}.npy", vec)

    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack([r for r in results if r is not None])
