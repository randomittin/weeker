"""embed: on-disk cache keyed by sha256(text) avoids repeat transport calls."""

from __future__ import annotations

import numpy as np

from atlas.core.embed import embed


class CountingTransport:
    """Deterministic fake embedder; counts how many texts it was asked to embed."""

    def __init__(self, dim=8):
        self.dim = dim
        self.embed_calls = 0
        self.texts_embedded = 0

    def embed(self, texts):
        self.embed_calls += 1
        self.texts_embedded += len(texts)
        # deterministic per-text vector so cached vs fresh compare equal
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i] = (len(t) % 7) + 1
        return out


def test_cache_hit_avoids_second_call(tmp_path):
    t = CountingTransport()
    v1 = embed(["alpha", "beta"], transport=t, cache_dir=tmp_path)
    assert v1.shape == (2, 8)
    assert t.texts_embedded == 2

    # second identical call: fully cached → transport not invoked again
    v2 = embed(["alpha", "beta"], transport=t, cache_dir=tmp_path)
    assert np.array_equal(v1, v2)
    assert t.texts_embedded == 2  # unchanged


def test_partial_cache_only_computes_new(tmp_path):
    t = CountingTransport()
    embed(["alpha"], transport=t, cache_dir=tmp_path)
    assert t.texts_embedded == 1
    embed(["alpha", "gamma"], transport=t, cache_dir=tmp_path)
    assert t.texts_embedded == 2  # only "gamma" is new


def test_order_and_values_preserved(tmp_path):
    t = CountingTransport()
    out = embed(["a", "bb", "ccc"], transport=t, cache_dir=tmp_path)
    assert out.shape == (3, 8)
    assert out[0][0] == (1 % 7) + 1
    assert out[1][0] == (2 % 7) + 1
