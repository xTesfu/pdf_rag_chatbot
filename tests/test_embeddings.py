import numpy as np

from app.embeddings import build_vector


def test_build_vector(monkeypatch):
    expected = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
    )

    def fake_encode(texts, normalize_embeddings):
        assert texts == ["Hello", "World"]
        assert normalize_embeddings is True
        return expected

    monkeypatch.setattr(
        "app.embeddings.embedder.encode",
        fake_encode,
    )

    chunks = [
        {"text": "Hello"},
        {"text": "World"},
    ]

    vectors = build_vector(chunks)

    assert np.array_equal(vectors, expected)

def test_build_vector_empty_chunks(monkeypatch):
    def fake_encode(texts, normalize_embeddings):
        assert texts == []
        assert normalize_embeddings is True
        return []

    monkeypatch.setattr(
        "app.embeddings.embedder.encode",
        fake_encode,
    )

    vectors = build_vector([])

    assert vectors == []