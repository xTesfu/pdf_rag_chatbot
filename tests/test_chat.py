import os

import pytest
from streamlit.testing.v1 import AppTest


@pytest.fixture
def chat_app(monkeypatch):
    """Run chat.py with heavy dependencies mocked out.

    NOTE: AppTest.from_file() re-executes chat.py in its own isolated
    script-run context on every .run() - it is NOT the same module object
    you get from a plain `import chat`. So patching "chat.get_doc_id"
    etc. patches a different module instance than the one AppTest
    actually runs, and the mocks silently have no effect (the real
    app.* implementations get called instead).
    Patching the functions where they're *defined* (in the app.*
    modules) works instead, because chat.py does `from app.x import y`
    on every rerun and re-reads the attribute off the shared, cached
    app.x module each time.
    """
    monkeypatch.setattr("app.vector_store.get_doc_id", lambda b: "doc1")
    monkeypatch.setattr(
        "app.pdf_loader.load_pdf",
        lambda path: [{"page": 1, "text": "Hello world"}],
    )
    monkeypatch.setattr(
        "app.chunker.chunk_text",
        lambda pages, document_name="": [
            {
                "text": "Hello world",
                "page": 1,
                "document": document_name,
            }
        ],
    )
    monkeypatch.setattr("app.vector_store.load_index", lambda doc_id: None)
    monkeypatch.setattr("app.vector_store.load_chunks", lambda doc_id: None)
    monkeypatch.setattr("app.embeddings.build_vector", lambda chunks: [[0.1, 0.2]])
    monkeypatch.setattr("app.vector_store.build_index", lambda vectors: object())
    monkeypatch.setattr("app.vector_store.save_index", lambda idx, doc_id: None)
    monkeypatch.setattr("app.vector_store.save_chunks", lambda chunks, doc_id: None)
    monkeypatch.setattr(os, "remove", lambda path: None)

    at = AppTest.from_file("chat.py")
    at.run(timeout=30)
    return at


def test_chat_initial_load(chat_app):
    assert not chat_app.exception
    assert chat_app.title[0].value == "📚 Chat with your PDFs"
    assert chat_app.session_state.messages == []
    assert chat_app.session_state.documents == {}


def test_chat_upload_builds_index(chat_app):
    chat_app.sidebar.file_uploader[0].set_value(
        [("test.pdf", b"fake-pdf-content", "application/pdf")]
    )
    chat_app.run(timeout=30)

    assert not chat_app.exception
    assert "doc1" in chat_app.session_state.documents
    assert chat_app.session_state.documents["doc1"]["name"] == "test.pdf"
    assert chat_app.sidebar.success[0].value == "PDF(s) processed successfully!"


def test_chat_upload_uses_cache(monkeypatch):
    monkeypatch.setattr("app.vector_store.get_doc_id", lambda b: "doc1")
    monkeypatch.setattr(
        "app.pdf_loader.load_pdf",
        lambda path: (_ for _ in ()).throw(Exception("should not be called")),
    )
    monkeypatch.setattr("app.chunker.chunk_text", lambda pages, document_name="": None)
    monkeypatch.setattr("app.embeddings.build_vector", lambda chunks: None)
    monkeypatch.setattr("app.vector_store.build_index", lambda vectors: None)
    monkeypatch.setattr("app.vector_store.load_index", lambda doc_id: object())
    monkeypatch.setattr(
        "app.vector_store.load_chunks",
        lambda doc_id: [{"text": "cached chunk", "page": 1, "document": "test.pdf"}],
    )
    monkeypatch.setattr("app.vector_store.save_index", lambda idx, doc_id: None)
    monkeypatch.setattr("app.vector_store.save_chunks", lambda chunks, doc_id: None)
    monkeypatch.setattr(os, "remove", lambda path: None)

    at = AppTest.from_file("chat.py")
    at.run(timeout=30)
    at.sidebar.file_uploader[0].set_value(
        [("test.pdf", b"fake-pdf-content", "application/pdf")]
    )
    at.run(timeout=30)

    assert not at.exception
    assert at.session_state.documents["doc1"]["chunks"] == [
        {"text": "cached chunk", "page": 1, "document": "test.pdf"}
    ]


def test_chat_ask_without_pdfs(chat_app):
    chat_app.chat_input[0].set_value("What is Python?").run()

    assert not chat_app.exception
    assert chat_app.warning[0].value == "Please upload at least one PDF."
    assert chat_app.session_state.messages == []


def test_chat_ask_returns_answer(chat_app, monkeypatch):
    chat_app.session_state.documents = {
        "doc1": {
            "name": "file1.pdf",
            "index": object(),
            "chunks": [{"text": "Python is great", "page": 2, "document": "file1.pdf"}],
        }
    }
    chat_app.run(timeout=30)

    captured = {}

    def fake_retrieve(query):
        return [
            {
                "document": "file1.pdf",
                "page": 2,
                "text": "Python is great",
                "score": 0.9,
            }
        ]

    def fake_ask_llm(context, question):
        captured["context"] = context
        captured["question"] = question
        return "Python is a programming language."

    monkeypatch.setattr("app.retriever.retrieve", fake_retrieve)
    monkeypatch.setattr("app.llm.ask_llm", fake_ask_llm)

    chat_app.chat_input[0].set_value("What is Python?").run()

    assert not chat_app.exception
    assert captured["question"] == "What is Python?"
    assert "file1.pdf" in captured["context"]
    assert "page 2" in captured["context"]
    assert "Python is great" in captured["context"]
    assert chat_app.session_state.messages == [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
    ]


def test_chat_clear_messages(chat_app):
    chat_app.session_state.messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    chat_app.run(timeout=30)

    chat_app.sidebar.button[0].click().run()

    assert not chat_app.exception
    assert chat_app.session_state.messages == []


def test_chat_reset_all_pdfs(chat_app):
    chat_app.session_state.documents = {
        "doc1": {"name": "file1.pdf", "index": object(), "chunks": []}
    }
    chat_app.session_state.messages = [
        {"role": "user", "content": "Hi"},
    ]
    chat_app.run(timeout=30)

    chat_app.sidebar.button[1].click().run()

    assert not chat_app.exception
    assert chat_app.session_state.documents == {}
    assert chat_app.session_state.messages == []