
import numpy as np

from app.vector_store import (
    build_index,
    clear_document,
    get_all_documents,
    get_doc_id,
    get_doc_path,
    load_chunks,
    load_index,
    save_chunks,
    save_index,
)


def test_get_doc_id_is_deterministic():
    data = b"hello world"

    id1 = get_doc_id(data)
    id2 = get_doc_id(data)

    assert id1 == id2
    assert isinstance(id1, str)
    assert len(id1) == 32  # md5 length

def test_get_doc_path_creates_directory(tmp_path, monkeypatch):
    from app import vector_store

    monkeypatch.setattr(vector_store, "DATA_DIR", tmp_path)

    doc_id = "testdoc"
    path = get_doc_path(doc_id)

    assert path.exists()
    assert path.is_dir()
    assert path.name == doc_id

def test_build_index_creates_faiss_index():
    vectors = np.array([
        [1.0, 0.0],
        [0.5, 0.5],
    ], dtype="float32")

    index = build_index(vectors)

    assert index.ntotal == 2

def test_save_and_load_index(tmp_path, monkeypatch):
    import faiss

    from app import vector_store

    monkeypatch.setattr(vector_store, "DATA_DIR", tmp_path)

    vectors = np.array([[1.0, 0.0]], dtype="float32")
    index = faiss.IndexFlatIP(2)
    index.add(vectors)

    doc_id = "doc1"

    save_index(index, doc_id)

    loaded = load_index(doc_id)

    assert loaded is not None
    assert loaded.ntotal == 1

def test_load_index_returns_none(tmp_path, monkeypatch):
    from app import vector_store

    monkeypatch.setattr(vector_store, "DATA_DIR", tmp_path)

    result = load_index("missing_doc")

    assert result is None

def test_save_and_load_chunks(tmp_path, monkeypatch):
    from app import vector_store

    monkeypatch.setattr(vector_store, "DATA_DIR", tmp_path)

    doc_id = "doc1"
    chunks = [
        {"text": "hello", "page": 1},
        {"text": "world", "page": 2},
    ]

    save_chunks(chunks, doc_id)

    loaded = load_chunks(doc_id)

    assert loaded == chunks

def test_load_chunks_returns_none(tmp_path, monkeypatch):
    from app import vector_store

    monkeypatch.setattr(vector_store, "DATA_DIR", tmp_path)

    assert load_chunks("missing") is None

def test_clear_document_removes_files(tmp_path, monkeypatch):
    from app import vector_store

    monkeypatch.setattr(vector_store, "DATA_DIR", tmp_path)

    doc_id = "doc1"
    path = get_doc_path(doc_id)

    # create fake files
    (path / "index.bin").write_text("x")
    (path / "chunks.pkl").write_text("x")

    clear_document(doc_id)

    assert not path.exists()

def test_get_all_documents(tmp_path, monkeypatch):
    from app import vector_store

    monkeypatch.setattr(vector_store, "DATA_DIR", tmp_path)

    (tmp_path / "doc1").mkdir()
    (tmp_path / "doc2").mkdir()

    docs = get_all_documents()

    assert set(docs) == {"doc1", "doc2"}