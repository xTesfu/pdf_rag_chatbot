import os
import tempfile
from typing import Annotated

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from app.chunker import chunk_text
from app.embeddings import build_vector
from app.llm import ask_llm
from app.pdf_loader import load_pdf
from app.retriever import retrieve
from app.vector_store import (
    build_index,
    get_all_documents,
    get_doc_id,
    load_chunks,
    load_index,
    save_chunks,
    save_index,
)

app = FastAPI(
    title="PDF RAG API",
    version="1.1.0"
)


class Question(BaseModel):
    question: str


# -------------------------
# UPLOAD PDFs
# -------------------------
@app.post("/upload")
async def upload_pdfs(
    files: Annotated[list[UploadFile], File()]
):
    uploaded_documents = []

    for file in files:

        pdf_bytes = await file.read()
        doc_id = get_doc_id(pdf_bytes)

        index = load_index(doc_id)
        chunks = load_chunks(doc_id)

        if index is None or chunks is None:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                pdf_path = tmp.name

            try:
                pages = load_pdf(pdf_path)
                chunks = chunk_text(pages)

                vectors = build_vector(chunks)
                index = build_index(vectors)

                save_index(index, doc_id)
                save_chunks(chunks, doc_id)

            finally:
                os.remove(pdf_path)

        uploaded_documents.append(
            {
                "filename": file.filename,
                "document_id": doc_id,
            }
        )

    return {
        "message": "PDF(s) uploaded successfully",
        "documents": uploaded_documents,
        "count": len(uploaded_documents),
    }


# -------------------------
# ASK QUESTION
# -------------------------
@app.post("/ask")
def ask(question: Question):

    results = retrieve(question.question)

    context_text = "\n\n".join(
        [
            f"[{r['document']} | page {r['page']}]\n{r['text']}"
            for r in results
        ]
    )

    answer = ask_llm(
        context_text,
        question.question
    )

    return {
        "question": question.question,
        "answer": answer,
        "sources_found": len(results)
    }


# -------------------------
# LIST DOCUMENTS
# -------------------------
@app.get("/documents")
def documents():

    docs = get_all_documents()

    return {
        "documents": docs,
        "count": len(docs)
    }