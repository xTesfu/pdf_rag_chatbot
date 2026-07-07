# PDF RAG with Ollama (qwen2.5:7b)

A multi-document **Retrieval-Augmented Generation (RAG)** system that lets you upload PDFs and ask questions across all of them. Answers are grounded in your documents and include source citations (document name and page number).

The project supports three interfaces:

- **Streamlit UI** — chat-style web app for uploading PDFs and asking questions
- **FastAPI REST API** — programmatic upload and Q&A endpoints
- **CLI** — interactive terminal chat loop

LLM inference runs through [Ollama](https://ollama.com/) via the OpenAI-compatible API. Embeddings use [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) and vector search uses [FAISS](https://github.com/facebookresearch/faiss).

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Streamlit UI](#streamlit-ui)
  - [REST API](#rest-api)
  - [CLI](#cli)
- [Docker](#docker)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **Multi-PDF support** — index and search across many documents at once
- **Persistent caching** — FAISS indexes and text chunks are saved under `data/` so re-uploading the same PDF skips re-processing
- **Source citations** — the LLM is prompted to cite document name and page number in every answer
- **Three entry points** — Streamlit, FastAPI, and a CLI chat loop
- **Docker Compose** — run the API and Streamlit UI in containers with shared storage
- **Test suite** — pytest coverage for loaders, chunking, embeddings, retrieval, API, and CLI logic

---

## Architecture

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  chat.py    │   │   api.py    │   │   main.py   │
│ (Streamlit) │   │  (FastAPI)  │   │    (CLI)    │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
              ┌──────────▼──────────┐
              │   RAG pipeline      │
              │                     │
              │  pdf_loader         │
              │  chunker            │
              │  embeddings         │
              │  vector_store       │
              │  retriever          │
              │  llm (Ollama)       │
              └──────────┬──────────┘
                         │
              ┌──────────▼──────────┐
              │   data/<doc_id>/    │
              │   index.bin         │
              │   chunks.pkl        │
              └─────────────────────┘
```

| Component | Technology |
|-----------|------------|
| PDF parsing | `pypdf` |
| Text chunking | `langchain-text-splitters` (1000 chars, 200 overlap) |
| Embeddings | `sentence-transformers` — `BAAI/bge-small-en-v1.5` |
| Vector search | `faiss-cpu` — inner-product similarity |
| LLM | Ollama (OpenAI-compatible client) |
| Web UI | Streamlit |
| API | FastAPI + Uvicorn |

---

## Prerequisites

- **Python 3.12+** (local development)
- **[Ollama](https://ollama.com/download)** installed and running
- A model pulled in Ollama, e.g.:

  ```bash
  ollama pull qwen2.5:7b
  ```

- **Docker & Docker Compose** (optional, for containerized deployment)

On first run, `sentence-transformers` downloads the embedding model (~130 MB). Ensure you have network access and sufficient disk space.

---

## Quick Start

### 1. Clone and set up the environment

```bash
git clone <your-repo-url>
cd Ollama

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` if needed (defaults work for local Ollama):

```env
QWEN_API_KEY=ollama
MODEL_NAME=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

### 3. Start Ollama (if not already running)

```bash
ollama serve
```

### 4. Run the Streamlit UI

```bash
streamlit run chat.py
```

Open [http://localhost:8501](http://localhost:8501), upload PDFs in the sidebar, and start asking questions.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `QWEN_API_KEY` | `ollama` | API key passed to the OpenAI client. Ollama accepts any value. |
| `MODEL_NAME` | `qwen2.5:7b` | Ollama model tag used for chat completions |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible base URL for Ollama |
| `PDF_PATHS` | *(empty)* | Comma-separated PDF paths for the CLI (e.g. `data/a.pdf,data/b.pdf`) |

When running in Docker on Linux/WSL, `OLLAMA_BASE_URL` defaults to `http://host.docker.internal:11434/v1` so containers can reach Ollama on the host.

---

## Usage

### Streamlit UI

```bash
streamlit run chat.py
```

- Upload one or more PDFs from the sidebar
- Ask questions in the chat input
- View loaded documents in the sidebar
- Use **Clear Chat** to reset conversation history
- Use **Reset All PDFs** to clear session state (cached files in `data/` remain on disk)

### REST API

Start the server:

```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

#### `POST /upload`

Upload one or more PDF files for indexing.

```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf"
```

**Response:**

```json
{
  "message": "PDF(s) uploaded successfully",
  "documents": [
    { "filename": "document1.pdf", "document_id": "abc123..." }
  ],
  "count": 1
}
```

#### `POST /ask`

Ask a question against all indexed documents.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the report?"}'
```

**Response:**

```json
{
  "question": "What is the main topic of the report?",
  "answer": "...",
  "sources_found": 3
}
```

#### `GET /documents`

List all indexed document IDs.

```bash
curl http://localhost:8000/documents
```

**Response:**

```json
{
  "documents": ["abc123...", "def456..."],
  "count": 2
}
```

### CLI

Place PDFs in `data/` or set `PDF_PATHS`, then run:

```bash
python main.py
```

The CLI indexes any configured PDFs, then enters an interactive loop:

```
Ask: What are the key findings?
--- Answer ---
...

Ask: exit
```

Type `exit` or `quit` to leave. If no local PDFs are found and nothing is indexed in `data/`, the CLI prompts you to upload via the API or Streamlit UI first.

---

## Docker

Build and run the API and Streamlit services:

```bash
docker compose up --build
```

| Service | URL | Description |
|---------|-----|-------------|
| `api` | [http://localhost:8000](http://localhost:8000) | FastAPI backend |
| `streamlit` | [http://localhost:8501](http://localhost:8501) | Streamlit chat UI |

Both services share a `rag-data` Docker volume mounted at `/app/data` so indexes persist across restarts.

**Optional CLI service** (interactive, not started by default):

```bash
docker compose --profile cli run --rm cli
```

Ensure Ollama is running on the host and reachable at `host.docker.internal:11434`. On Linux, `extra_hosts: host-gateway` is configured in `docker-compose.yml`.

Environment variables can be overridden via a `.env` file in the project root or shell exports before `docker compose up`.

---

## Project Structure

```
Ollama/
├── app/
│   ├── chunker.py       # Split pages into overlapping text chunks
│   ├── embeddings.py    # Sentence-transformer encoding
│   ├── llm.py           # Ollama chat completions
│   ├── pdf_loader.py    # Extract text from PDF pages
│   ├── retriever.py     # Cross-document FAISS search
│   └── vector_store.py  # Index/chunk persistence (FAISS + pickle)
├── tests/               # pytest test suite
├── api.py               # FastAPI application
├── chat.py              # Streamlit UI
├── main.py              # CLI entry point
├── data/                # Generated indexes (gitignored)
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── pytest.ini
```

Indexed documents are stored as:

```
data/
└── <md5-hash-of-pdf>/
    ├── index.bin    # FAISS vector index
    └── chunks.pkl   # Chunk metadata (text, page, document name)
```

Document IDs are MD5 hashes of the raw PDF bytes, so identical files always map to the same cache entry.

---

## How It Works

1. **Ingest** — PDF pages are extracted with `pypdf`.
2. **Chunk** — Each page is split into ~1000-character chunks with 200-character overlap.
3. **Embed** — Chunks are encoded with `BAAI/bge-small-en-v1.5` (L2-normalized for cosine similarity via inner product).
4. **Index** — Embeddings are stored in a per-document FAISS `IndexFlatIP`.
5. **Retrieve** — At query time, the question is embedded and searched against every indexed document. Top results across all docs are merged and ranked by score (default `k=5`, CLI uses `k=3`).
6. **Generate** — Retrieved chunks are passed as context to the Ollama LLM, which answers and cites sources.

Re-uploading the same PDF skips steps 1–4 if a cached index already exists in `data/`.

---

## Testing

Run the full test suite:

```bash
pytest
```

Run with verbose output:

```bash
pytest -v
```

Tests cover PDF loading, chunking, embeddings, vector store operations, retrieval, the FastAPI endpoints, and CLI logic. External services (Ollama, model downloads) are mocked in unit tests.

---

## Troubleshooting

### Ollama connection errors

- Confirm Ollama is running: `curl http://localhost:11434/api/tags`
- Verify `OLLAMA_BASE_URL` and `MODEL_NAME` in `.env`
- Ensure the model is pulled: `ollama list`

### Empty or poor answers

- Check that PDFs contain extractable text (scanned images without OCR will produce empty chunks)
- Upload more relevant documents or ask more specific questions
- Try a larger or instruction-tuned model in Ollama

### Slow first run

- The embedding model downloads on first use
- Indexing large PDFs takes time; subsequent runs use the cache in `data/`

### Docker cannot reach Ollama

- Use `OLLAMA_BASE_URL=http://host.docker.internal:11434/v1`
- On Linux/WSL, ensure `extra_hosts` is set (already in `docker-compose.yml`)
- Confirm Ollama listens on `0.0.0.0`, not only `127.0.0.1`

### Port already in use

Change the host port in `docker-compose.yml` or pass a different port to Uvicorn/Streamlit:

```bash
uvicorn api:app --port 8001
streamlit run chat.py --server.port 8502
```

---

## License

This project is provided as-is for learning and experimentation. Add your preferred license file if you plan to distribute it.
