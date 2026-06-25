import numpy as np

from app.retriever import retrieve


class FakeIndex:
    def __init__(self, distances, indices):
        self.distances = distances
        self.indices = indices

    def search(self, q_vec, k):
        return self.distances, self.indices


def test_retrieve_returns_results(monkeypatch):
    monkeypatch.setattr(
        "app.retriever.build_vector",
        lambda _: np.array([[0.1, 0.2]], dtype="float32"),
    )

    monkeypatch.setattr(
        "app.retriever.get_all_documents",
        lambda: ["doc1"],
    )

    monkeypatch.setattr(
        "app.retriever.load_index",
        lambda _: FakeIndex([[0.9]], [[0]]),
    )

    monkeypatch.setattr(
        "app.retriever.load_chunks",
        lambda _: [
            {
                "text": "Python is a programming language.",
                "document": "file1.pdf",
                "page": 1,
            }
        ],
    )

    results = retrieve("What is Python?")

    assert len(results) == 1
    assert results[0]["text"] == "Python is a programming language."
    assert results[0]["document"] == "file1.pdf"
    assert results[0]["page"] == 1
    assert results[0]["score"] == 0.9


def test_retrieve_returns_empty_when_no_documents(monkeypatch):
    monkeypatch.setattr(
        "app.retriever.build_vector",
        lambda _: np.array([[0.1, 0.2]], dtype="float32"),
    )

    monkeypatch.setattr(
        "app.retriever.get_all_documents",
        lambda: [],
    )

    results = retrieve("query")

    assert results == []


def test_retrieve_skips_missing_indexes(monkeypatch):
    monkeypatch.setattr(
        "app.retriever.build_vector",
        lambda _: np.array([[0.1, 0.2]], dtype="float32"),
    )

    monkeypatch.setattr(
        "app.retriever.get_all_documents",
        lambda: ["doc1"],
    )

    monkeypatch.setattr(
        "app.retriever.load_index",
        lambda _: None,
    )

    monkeypatch.setattr(
        "app.retriever.load_chunks",
        lambda _: None,
    )

    results = retrieve("query")

    assert results == []


def test_retrieve_sorts_results_by_score(monkeypatch):
    monkeypatch.setattr(
        "app.retriever.build_vector",
        lambda _: np.array([[0.1, 0.2]], dtype="float32"),
    )

    monkeypatch.setattr(
        "app.retriever.get_all_documents",
        lambda: ["doc1"],
    )

    fake_index = FakeIndex(
        [[0.2, 0.9]],
        [[0, 1]],
    )

    monkeypatch.setattr(
        "app.retriever.load_index",
        lambda _: fake_index,
    )

    monkeypatch.setattr(
        "app.retriever.load_chunks",
        lambda _: [
            {
                "text": "Low score",
                "document": "file.pdf",
                "page": 1,
            },
            {
                "text": "High score",
                "document": "file.pdf",
                "page": 2,
            },
        ],
    )

    results = retrieve("query", k=2)

    assert len(results) == 2
    assert results[0]["text"] == "High score"
    assert results[1]["text"] == "Low score"


def test_retrieve_respects_top_k(monkeypatch):
    monkeypatch.setattr(
        "app.retriever.build_vector",
        lambda _: np.array([[0.1, 0.2]], dtype="float32"),
    )

    monkeypatch.setattr(
        "app.retriever.get_all_documents",
        lambda: ["doc1"],
    )

    fake_index = FakeIndex(
        [[0.9, 0.8, 0.7]],
        [[0, 1, 2]],
    )

    monkeypatch.setattr(
        "app.retriever.load_index",
        lambda _: fake_index,
    )

    monkeypatch.setattr(
        "app.retriever.load_chunks",
        lambda _: [
            {
                "text": "Chunk 1",
                "document": "file.pdf",
                "page": 1,
            },
            {
                "text": "Chunk 2",
                "document": "file.pdf",
                "page": 2,
            },
            {
                "text": "Chunk 3",
                "document": "file.pdf",
                "page": 3,
            },
        ],
    )

    results = retrieve("query", k=2)

    assert len(results) == 2
    assert results[0]["text"] == "Chunk 1"
    assert results[1]["text"] == "Chunk 2"


def test_retrieve_across_multiple_documents(monkeypatch):
    monkeypatch.setattr(
        "app.retriever.build_vector",
        lambda _: np.array([[0.1, 0.2]], dtype="float32"),
    )

    monkeypatch.setattr(
        "app.retriever.get_all_documents",
        lambda: ["doc1", "doc2"],
    )

    index1 = FakeIndex(
        [[0.8]],
        [[0]],
    )

    index2 = FakeIndex(
        [[0.9]],
        [[0]],
    )

    indexes = {
        "doc1": index1,
        "doc2": index2,
    }

    chunks = {
        "doc1": [
            {
                "text": "Result from doc1",
                "document": "doc1.pdf",
                "page": 1,
            }
        ],
        "doc2": [
            {
                "text": "Result from doc2",
                "document": "doc2.pdf",
                "page": 5,
            }
        ],
    }

    monkeypatch.setattr(
        "app.retriever.load_index",
        lambda doc_id: indexes[doc_id],
    )

    monkeypatch.setattr(
        "app.retriever.load_chunks",
        lambda doc_id: chunks[doc_id],
    )

    results = retrieve("query", k=2)

    assert len(results) == 2
    assert results[0]["document"] == "doc2.pdf"
    assert results[1]["document"] == "doc1.pdf"