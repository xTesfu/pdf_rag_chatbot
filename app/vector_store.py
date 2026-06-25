import hashlib
import pickle
from pathlib import Path

import faiss
import numpy as np

DATA_DIR = Path("data")


def get_doc_id(pdf_bytes):
    return hashlib.md5(pdf_bytes).hexdigest()


def get_doc_path(doc_id):
    path = DATA_DIR / doc_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_index(vectors):
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(vectors).astype("float32"))
    return index


def save_index(index, doc_id):
    path = get_doc_path(doc_id)
    faiss.write_index(index, str(path / "index.bin"))


def load_index(doc_id):
    path = get_doc_path(doc_id)
    file = path / "index.bin"

    if file.exists():
        return faiss.read_index(str(file))

    return None


# ----------------------------
# CHUNKS (NOW STRUCTURED)
# ----------------------------
def save_chunks(chunks, doc_id):
    path = get_doc_path(doc_id)
    with open(path / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)


def load_chunks(doc_id):
    path = get_doc_path(doc_id)
    file = path / "chunks.pkl"

    if file.exists():
        with open(file, "rb") as f:
            return pickle.load(f)

    return None


def clear_document(doc_id):
    path = get_doc_path(doc_id)

    if not path.exists():
        return
    
    for file in path.iterdir():
        if file.is_file():
            file.unlink()

    path.rmdir()


def get_all_documents():
    if not DATA_DIR.exists():
        return []

    return [d.name for d in DATA_DIR.iterdir() if d.is_dir()]