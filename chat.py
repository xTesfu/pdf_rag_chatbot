import os
import tempfile

import streamlit as st

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

st.set_page_config(
    page_title="PDF-RAG Chat",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Chat with your PDFs")
st.markdown(
    "Upload multiple PDFs and ask questions across all of them."
)

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    """
    {
      doc_id: {
          "name": filename,
          "index": faiss_index,
          "chunks": [...]
      }
    }
    """
    st.session_state.documents = {}

# ---------------------------------------------------
# UPLOAD PDF(S)
# ---------------------------------------------------
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF(s)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    for uploaded_file in uploaded_files:

        pdf_bytes = uploaded_file.read()
        doc_id = get_doc_id(pdf_bytes)

        # Skip if already loaded
        if doc_id in st.session_state.documents:
            continue

        with st.spinner(f"Processing {uploaded_file.name}..."):

            # Save temporary PDF
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as tmp_file:
                tmp_file.write(pdf_bytes)
                pdf_path = tmp_file.name

            # Try loading cached version
            index = load_index(doc_id)
            saved_chunks = load_chunks(doc_id)

            # Build if cache doesn't exist
            if index is None or saved_chunks is None:

                # Extract text and chunks only when there's no cache to use
                pages = load_pdf(pdf_path)
                chunks = chunk_text(pages, document_name=uploaded_file.name)

                vectors = build_vector(chunks)
                index = build_index(vectors)

                save_index(index, doc_id)
                save_chunks(chunks, doc_id)

            else:
                chunks = saved_chunks

            # Store in session
            st.session_state.documents[doc_id] = {
                "name": uploaded_file.name,
                "index": index,
                "chunks": chunks,
            }

            os.remove(pdf_path)

    st.sidebar.success("PDF(s) processed successfully!")

# ---------------------------------------------------
# SHOW LOADED PDFS
# ---------------------------------------------------
if st.session_state.documents:

    st.sidebar.subheader("Loaded PDFs")

    for data in st.session_state.documents.values():
        st.sidebar.write(f"📄 {data['name']}")

# ---------------------------------------------------
# DISPLAY CHAT HISTORY
# ---------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------
if prompt := st.chat_input(
    "Ask something about your PDFs..."
):

    if not st.session_state.documents:
        st.warning("Please upload at least one PDF.")
        st.stop()

    # Display user message
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # Search across ALL PDFs
    results = retrieve(prompt)

    context_text = "\n\n".join(
        [
            f"[{r['document']} | page {r['page']}]\n{r['text']}"
            for r in results
        ]
    )

    # Ask LLM
    with st.spinner("Thinking..."):
        answer = ask_llm(
            context_text,
            prompt
        )

    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

# ---------------------------------------------------
# SIDEBAR ACTIONS
# ---------------------------------------------------
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []

if st.sidebar.button("Reset All PDFs"):
    st.session_state.documents = {}
    st.session_state.messages = []