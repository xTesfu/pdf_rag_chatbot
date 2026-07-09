import builtins
import tempfile

import main


class FakeFile:
    def __init__(self):
        self.data = b"pdf"

    def read(self):
        return self.data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def test_process_pdf_builds_index(monkeypatch):
    calls = {}

    monkeypatch.setattr("main.get_doc_id", lambda b: "doc1")

    monkeypatch.setattr("main.load_index", lambda doc_id: None)
    monkeypatch.setattr("main.load_chunks", lambda doc_id: None)

    monkeypatch.setattr(
        "main.load_pdf",
        lambda path: [
            {"page": 1, "text": "Hello world"}
        ],
    )

    monkeypatch.setattr(
        "main.chunk_text",
        lambda pages: [
            {"text": "Hello world", "page": 1, "document": "file.pdf"}
        ],
    )

    monkeypatch.setattr("main.build_vector", lambda chunks: [[0.1, 0.2]])

    monkeypatch.setattr("main.build_index", lambda vectors: object())

    monkeypatch.setattr(
        "main.save_index", 
        lambda idx, doc_id: calls.setdefault("saved_index", True)
    )
    monkeypatch.setattr(
        "main.save_chunks", 
        lambda chunks, doc_id: calls.setdefault("saved_chunks", True)
    )

    result = main.process_pdf("fake.pdf", b"pdf-bytes")

    assert result == "doc1"
    assert calls["saved_index"] is True
    assert calls["saved_chunks"] is True

def test_process_pdf_cache_hit(monkeypatch):
    monkeypatch.setattr("main.get_doc_id", lambda b: "doc1")

    monkeypatch.setattr("main.load_index", lambda doc_id: object())
    monkeypatch.setattr("main.load_chunks", lambda doc_id: [{"cached": True}])

    monkeypatch.setattr(
        "main.load_pdf", 
        lambda p: (_ for _ in ()).throw(Exception("should not be called"))
    )
    monkeypatch.setattr("main.chunk_text", lambda x: None)
    monkeypatch.setattr("main.build_vector", lambda x: None)
    monkeypatch.setattr("main.build_index", lambda x: None)

    result = main.process_pdf("fake.pdf", b"bytes")

    assert result == "doc1"

def test_main_processes_pdfs(monkeypatch):
    monkeypatch.setattr(main, "process_pdf", lambda p, b: "doc1")

    # FIXED open mock
    def fake_open(*args, **kwargs):
        return FakeFile()

    monkeypatch.setattr(builtins, "open", fake_open)

    # tempfile mock (keep simple)
    class FakeTemp:
        name = "tmp.pdf"

        def write(self, x): pass

        def __enter__(self): return self

        def __exit__(self, *a): pass

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", lambda *a, **k: FakeTemp())

    import os
    monkeypatch.setattr(os, "remove", lambda x: None)

    # exit loop immediately
    monkeypatch.setattr("builtins.input", lambda _: "exit")

    monkeypatch.setattr(main, "retrieve", lambda q, k=3: [])

    main.main(["file1.pdf"])

def test_context_formatting(monkeypatch):
    monkeypatch.setattr(main, "process_pdf", lambda p, b: "doc1")

    def fake_open(*args, **kwargs):
        return FakeFile()

    monkeypatch.setattr(builtins, "open", fake_open)

    class FakeTemp:
        name = "tmp.pdf"

        def write(self, x):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", lambda *a, **k: FakeTemp())

    import os

    monkeypatch.setattr(os, "remove", lambda x: None)

    # -------------------------
    # Fake retrieval result
    # -------------------------
    monkeypatch.setattr(
        main,
        "retrieve",
        lambda q, k=3: [
            {
                "document": "file1.pdf",
                "page": 2,
                "text": "Python is great",
                "score": 0.9,
            }
        ],
    )

    # Capture what goes into LLM
    captured = {}

    def fake_ask_llm(context, question):
        captured["context"] = context
        captured["question"] = question
        return "ANSWER"

    monkeypatch.setattr(main, "ask_llm", fake_ask_llm)

    # -------------------------
    # Stop loop after one run
    # -------------------------
    inputs = iter(["hello", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    # -------------------------
    # Run
    # -------------------------
    main.main(["file1.pdf"])

    # -------------------------
    # ASSERTIONS (REAL TEST)
    # -------------------------
    assert "file1.pdf" in captured["context"]
    assert "page 2" in captured["context"]
    assert "Python is great" in captured["context"]
    assert "hello" in captured["question"]


