"""Keyless local model2vec embedding transport (WEEKER_EMBED_PROVIDER=local).

Deterministic and tiny: asserts the transport returns an ``(n, dim)`` float32
array at the model's true dim, reuses the process-wide model cache, and that the
on-disk sha256 embedding cache short-circuits a repeat call. Skipped gracefully
when model2vec or the model snapshot is not present (e.g. offline CI).
"""

from __future__ import annotations

import numpy as np
import pytest

from weeker.core import config

MODEL = config.EMBED_LOCAL_MODEL


def _model_available() -> bool:
    try:
        import model2vec  # noqa: F401
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache(MODEL, "config.json"), str)


pytestmark = pytest.mark.skipif(
    not _model_available(), reason=f"model2vec model {MODEL!r} not downloaded"
)


def _dim() -> int:
    from model2vec import StaticModel

    return int(np.asarray(StaticModel.from_pretrained(MODEL).encode(["x"])).shape[-1])


def test_local_transport_dim_and_model_cache(monkeypatch):
    from weeker.core.embed import _LOCAL_MODEL_CACHE, LocalModel2VecTransport

    dim = _dim()
    monkeypatch.setattr(config, "EMBED_DIM", dim)
    _LOCAL_MODEL_CACHE.clear()

    t = LocalModel2VecTransport()
    out = t.embed(["hello world"])
    assert out.shape == (1, dim)
    assert out.dtype == np.float32
    # loaded once, cached process-wide, reused by a second instance (no reload)
    assert MODEL in _LOCAL_MODEL_CACHE
    assert LocalModel2VecTransport()._load() is t._load()


def test_local_transport_on_disk_cache_hit(monkeypatch, tmp_path):
    from weeker.core.embed import LocalModel2VecTransport, embed

    monkeypatch.setattr(config, "EMBED_DIM", _dim())

    class _Counting(LocalModel2VecTransport):
        calls = 0

        def embed(self, texts):
            _Counting.calls += 1
            return super().embed(texts)

    t = _Counting()
    a = embed(["cache me local"], transport=t, cache_dir=tmp_path)
    b = embed(["cache me local"], transport=t, cache_dir=tmp_path)
    assert np.array_equal(a, b)
    assert _Counting.calls == 1  # second call served from the on-disk cache
