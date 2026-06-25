import tempfile

from app.chunker import chunk_text
from app.embeddings import build_vector
from app.llm import ask_llm
from app.pdf_loader import load_pdf
from app.retriever import retrieve
from app.vector_store import (
    build_index,
    get_doc_id,
    load_chunks,
    load_index,
    save_chunks,
    save_index,
)


def process_pdf(pdf_path, pdf_bytes):
    doc_id = get_doc_id(pdf_bytes)

    print(f"\n📄 Processing document: {doc_id}")

    index = load_index(doc_id)
    chunks = load_chunks(doc_id)

    # -------------------------
    # CACHE MISS → BUILD
    # -------------------------
    if index is None or chunks is None:
        print("🔨 Building new index...")

        pages = load_pdf(pdf_path)
        chunks = chunk_text(pages)

        vectors = build_vector(chunks)
        index = build_index(vectors)

        save_index(index, doc_id)
        save_chunks(chunks, doc_id)

    else:
        print("⚡ Loaded cached index.")

    return doc_id


def main(pdf_paths):
    print("🚀 Starting multi-PDF RAG system...\n")
    for path in pdf_paths:

        with open(path, "rb") as f:
            pdf_bytes = f.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        process_pdf(tmp_path, pdf_bytes)

        import os
        os.remove(tmp_path)

    print("\n✅ System ready! Ask questions.\n")

    # -------------------------
    # CHAT LOOP
    # -------------------------
    while True:
        query = input("Ask: ")

        if query.lower() in ["exit", "quit"]:
            break

        results = retrieve(query, k=3)

        context_text = "\n\n".join(
            [
                f"[{r['document']} | page {r['page']}]\n{r['text']}"
                for r in results
            ]
        )

        answer = ask_llm(context_text, query)

        print("\n--- Answer ---")
        print(answer)
        print("\n")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    pdf_files = [
        "data/file1.pdf",
        "data/file2.pdf",
        "data/file3.pdf",
    ]

    main(pdf_files)