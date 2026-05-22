# GreenPack Industries — AI-Powered EPR Compliance System

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![FAISS](https://img.shields.io/badge/FAISS-1.8.0-FF6F00?logo=meta&logoColor=white)](https://faiss.ai)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.2%3A1b-black?logo=llama&logoColor=white)](https://ollama.com)
[![Groq](https://img.shields.io/badge/Groq-llama--3.3--70b-F55036?logo=groq&logoColor=white)](https://groq.com)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5-7952B3?logo=bootstrap&logoColor=white)](https://getbootstrap.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **A production-grade, AI-powered Extended Producer Responsibility (EPR) compliance workflow platform.**
> Combines a local-first hybrid LLM architecture, deterministic reconciliation engine, and a RAG-based compliance assistant — all deployable on a single machine.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture Overview](#architecture-overview)
- [Architecture Diagram](#architecture-diagram)
- [Why Hybrid LLM?](#why-hybrid-llm)
- [Deterministic vs LLM Logic](#deterministic-vs-llm-logic)
- [Hallucination Prevention](#hallucination-prevention)
- [RAG Pipeline](#rag-pipeline)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [API Endpoints](#api-endpoints)
- [Frontend Pages](#frontend-pages)
- [Docker](#docker)
- [Future Improvements](#future-improvements)
- [AI Coding Workflow](#ai-coding-workflow)

---

## Project Overview

GreenPack EPR Compliance System is an AI-assisted workflow platform built for plastic producers operating under India's Extended Producer Responsibility (EPR) framework. The system automates the three most critical compliance operations:

| Workflow | Description |
|---|---|
| **Monthly Declaration** | Producers submit plastic quantity declarations (rigid, flexible, multilayer) via a validated API. Stored in SQLite with UUID tracking. |
| **ERP Reconciliation** | Declared quantities are reconciled against ERP procurement data (CSV). Mismatches are computed deterministically; a local LLM narrates the findings. |
| **RAG Compliance Q&A** | A retrieval-augmented generation assistant answers EPR regulation questions, grounded exclusively in ingested compliance documents. |

**Core design principles:**

- **Local-first AI** — Ollama runs the LLM on-device; no data leaves the premises for primary workflows
- **Hybrid intelligence** — Local Ollama for fast summarization; cloud Groq LLM available for high-quality RAG reasoning
- **Deterministic engine** — All compliance math (mismatch %, status flags) runs in pure Python — never delegated to an LLM
- **Hallucination prevention** — FAISS retrieval threshold filtering ensures the LLM is never called without grounded context

---

## Architecture Overview

The system follows a clean layered architecture: thin HTTP routes → service layer → repository pattern → data stores. AI components are strictly isolated to narrative generation and Q&A — never to compliance calculations.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     GreenPack EPR — System Layers                        │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │  FRONTEND  (Bootstrap 5 + Jinja2)                                 │   │
│  │  Dashboard · Declaration Form · Reconciliation · Ask AI (RAG)     │   │
│  └───────────────────────┬───────────────────────────────────────────┘   │
│                           │ HTTP                                          │
│  ┌───────────────────────▼───────────────────────────────────────────┐   │
│  │  FASTAPI ROUTER LAYER  (thin routes — HTTP concerns only)         │   │
│  │  POST /submit · GET /summary/{id}/{month} · POST /ask · GET /health│  │
│  └──────────┬──────────────────────────┬────────────────────────────┘   │
│             │                          │                                  │
│  ┌──────────▼──────────┐  ┌────────────▼──────────────────────────────┐ │
│  │  DECLARATION SERVICE │  │  RECONCILIATION SERVICE                    │ │
│  │  Pydantic validation │  │  ERP CSV load · diff_pct (pure Python)    │ │
│  │  UUID generation     │  │  Status flags · LLM prompt assembly       │ │
│  └──────────┬──────────┘  └────────────┬──────────────────────────────┘ │
│             │                          │                                  │
│  ┌──────────▼──────────┐  ┌────────────▼──────────────────────────────┐ │
│  │  REPOSITORY LAYER    │  │  LLM SERVICE (async Ollama wrapper)       │ │
│  │  DB I/O only         │  │  httpx → Ollama :11434 (llama3.2:1b)     │ │
│  └──────────┬──────────┘  │  Retry logic · graceful degradation       │ │
│             │              └────────────┬──────────────────────────────┘ │
│  ┌──────────▼──────────┐               │                                 │
│  │  SQLite (greenpack.db)               │  ┌────────────────────────┐    │
│  └─────────────────────┘               └──│  Hybrid LLM Layer       │    │
│                                            │  Local: Ollama llama3.2 │    │
│  ┌────────────────────────────────────┐   │  Cloud: Groq llama-70b  │    │
│  │  RAG PIPELINE                       │   └────────────────────────┘    │
│  │  ingest.py (PyMuPDF + chunking)    │                                  │
│  │    ↓                                │                                  │
│  │  sentence-transformers embeddings  │                                  │
│  │  (all-MiniLM-L6-v2 · 384-dim)     │                                  │
│  │    ↓                                │                                  │
│  │  FAISS IndexFlatIP + metadata.json │                                  │
│  │    ↓                                │                                  │
│  │  retriever.py (threshold=0.45)     │                                  │
│  │    ↓                                │                                  │
│  │  LLM (grounded answer generation)  │                                  │
│  └────────────────────────────────────┘                                  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Diagram

> Simplified view of the five system layers and their interactions.

```
┌─────────────────────┐
│   Frontend Layer     │  Bootstrap 5 · Jinja2 templates · Chart.js
│  (4 pages)          │  Dashboard / Declare / Summary / Ask AI
└──────────┬──────────┘
           │ HTTP/REST
┌──────────▼──────────┐
│   Backend Layer      │  FastAPI · Pydantic v2 · uvicorn
│  (Router → Service → Repository)                        │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Deterministic       │  reconciliation_service.py
│  Engine              │  Pure Python maths — no LLM
│                      │  |declared − procured| / procured × 100
│                      │  ERP CSV via erp_loader.py
└──────────┬──────────┘
           │ pre-computed values only
┌──────────▼──────────┐
│   RAG Pipeline       │  ingest.py · vector_store.py · retriever.py
│                      │  FAISS IndexFlatIP · metadata.json
│                      │  sentence-transformers all-MiniLM-L6-v2
│                      │  Cosine similarity threshold = 0.45
└──────────┬──────────┘
           │ grounded context
┌──────────▼──────────┐
│   Hybrid LLM Layer   │  Local:  Ollama llama3.2:1b  (reconciliation narrative)
│                      │  Cloud:  Groq llama-3.3-70b  (RAG Q&A reasoning)
└─────────────────────┘
```

---

## Why Hybrid LLM?

The system deliberately uses **two different LLM tiers** for two different tasks — balancing speed, cost, quality, and privacy.

| Dimension | Local Ollama (llama3.2:1b) | Cloud Groq (llama-3.3-70b) |
|---|---|---|
| **Task** | Reconciliation narrative generation | RAG compliance Q&A |
| **Latency** | ~2–5 s on CPU | ~0.5–2 s via Groq API |
| **Quality** | Sufficient for structured summaries | High-quality for open-ended reasoning |
| **Cost** | Free — runs on-device | Pay-per-token (very cheap on Groq) |
| **Privacy** | Data never leaves the machine | Queries sent to Groq API |
| **Offline** | Fully offline capable | Requires internet |

**Philosophy — Local-First AI:**

- The reconciliation summary uses Ollama because the LLM is only narrating **pre-computed numbers** — a small 1B model is entirely sufficient and keeps sensitive producer data on-premises.
- The RAG assistant uses a larger model when available because answering open-ended regulatory questions benefits from stronger language understanding and longer context handling.
- The system degrades gracefully: if Ollama is down, the reconciliation endpoint still returns all computed data — only the AI narrative is replaced with a fallback message.

---

## Deterministic vs LLM Logic

This is one of the most important architectural decisions in the system.

```
┌──────────────────────────────┬─────────────────────────────────────────┐
│  Operation                   │  Implementation                          │
├──────────────────────────────┼─────────────────────────────────────────┤
│  Declaration validation       │  Pydantic v2 schema — 100% deterministic│
│  UUID generation              │  Python uuid4() — deterministic          │
│  Mismatch % calculation       │  Pure Python formula — deterministic     │
│  OK / MISMATCH status flag    │  5% threshold check — deterministic      │
│  Total kg aggregation         │  Python sum() — deterministic            │
│  ERP data loading             │  Pandas CSV read — deterministic         │
├──────────────────────────────┼─────────────────────────────────────────┤
│  Reconciliation narrative     │  Ollama llama3.2:1b — LLM               │
│  Compliance Q&A answers       │  Groq / Ollama — LLM                    │
└──────────────────────────────┴─────────────────────────────────────────┘
```

**Why validation and reconciliation calculations never use an LLM:**

1. **Auditability** — Regulatory bodies require reproducible, bit-identical results. LLMs are stochastic.
2. **Correctness** — LLMs hallucinate numbers. A 4.2% mismatch could be reported as 14.2%.
3. **Latency** — Removing LLM from the math path saves 2–10 seconds per reconciliation request.
4. **Determinism** — Same input must always yield the same compliance status. LLMs cannot guarantee this.

The LLM only receives **pre-computed, structured data** and is asked to write a human-readable narrative. It is explicitly forbidden from doing arithmetic.

---

## Hallucination Prevention

The RAG pipeline implements multiple layers of hallucination prevention:

### 1. Retrieval Threshold Filtering
```python
RETRIEVAL_THRESHOLD = 0.45   # cosine similarity minimum
```
Every retrieved chunk is scored by cosine similarity against the query embedding. Chunks below 0.45 are **discarded before the LLM is called**. If no chunk passes the threshold, the LLM is not invoked at all.

### 2. Hardcoded Fallback — Not LLM-Generated
```python
RAG_NO_CONTEXT_RESPONSE = "I do not know based on the provided documents."
```
The fallback string is a Python constant — it is **never generated by the LLM**. This guarantees that the fallback response is always identical and never varies.

### 3. Strict Grounded System Prompt
The RAG prompt template instructs the LLM:
- Answer **only** from the provided context chunks
- Cite the source document and chunk number for every claim
- If the answer is not in the context, say so explicitly

### 4. Source Citations in Every Response
Every `POST /ask` response returns a `citations` array:
```json
{
  "answer": "...",
  "citations": [
    { "source": "epr_guidelines_2024.pdf", "chunk": 3, "score": 0.72, "preview": "..." }
  ],
  "context_found": true
}
```

---

## RAG Pipeline

The RAG pipeline is implemented across three modules: `ingest.py`, `vector_store.py`, and `retriever.py`.

```
1. Document Ingestion (ingest.py)
   ├── Scan documents/epr_docs/ for .pdf and .txt files
   ├── Extract text via PyMuPDF (fitz)
   ├── Chunk text: CHUNK_SIZE=600 tokens, CHUNK_OVERLAP=80
   └── Track ingested files in metadata.json (avoids re-ingestion on restart)

2. Embedding Generation
   ├── Model: sentence-transformers/all-MiniLM-L6-v2
   ├── Dimension: 384
   ├── CPU-only (no GPU required), ~80 MB
   └── L2-normalized for cosine similarity via IndexFlatIP

3. Vector Indexing (vector_store.py)
   ├── FAISS IndexFlatIP (exact search, optimal for <100K chunks)
   ├── Persisted to disk: faiss_index/index.faiss + metadata.json
   └── metadata.json stores: source, chunk_index, text, text_preview

4. Semantic Retrieval (retriever.py)
   ├── Embed incoming question using same sentence-transformer model
   ├── Search FAISS for top-k=4 candidates
   ├── Filter: discard chunks with score < 0.45
   └── Return Citation objects: source, chunk, score, preview

5. Grounded Answer Generation
   ├── If no citations → return hardcoded fallback (NO LLM call)
   ├── Build numbered context block from full chunk texts
   ├── Format prompt from rag_prompt.txt template
   └── Call Ollama/Groq → return answer + citations
```

**Auto-ingestion on startup:** The FastAPI lifespan hook calls `ingest_documents()` at every startup. Only new files (not in `metadata.json`) are re-processed — existing indexed documents are skipped.

---

## Key Engineering Decisions

| Decision | Rationale |
|---|---|
| **Repository pattern** | `repository.py` isolates all DB I/O. Services never call SQLAlchemy directly — testable, swappable. |
| **Async Ollama via httpx** | All LLM calls are `async` with configurable timeout (60s) and retry logic (2 retries + exponential backoff). |
| **`metadata.json` instead of pickle** | FAISS index metadata stored as JSON — human-readable, version-control friendly, avoids pickle security risks. |
| **Retrieval threshold at 0.45** | Tuned empirically on EPR docs. Prevents hallucination while maintaining recall on genuine EPR queries. |
| **Modular prompt templates** | Prompts stored in `app/prompts/*.txt` — editable without touching Python code. Separate templates for summary and RAG. |
| **Pydantic-settings config** | All config validated at startup via `pydantic-settings`. Fail-fast on misconfiguration rather than silent runtime errors. |
| **Service layer separation** | Routes handle HTTP concerns only (status codes, error mapping). All business logic lives in `services/`. |
| **Local-first AI philosophy** | Ollama runs on-device. Sensitive producer data (declared quantities) never leaves the premises. |
| **streaming=False for Ollama** | Chosen for simplicity. Streaming with SSE is listed as a future improvement for perceived latency. |
| **FAISS IndexFlatIP** | Exact search — no approximation errors. Appropriate for <100K chunk scale. IVF index planned for 1M+ scale. |

---

## Project Structure

```
greenpack-epr/
│
├── app/
│   ├── main.py                         # FastAPI app factory + lifespan orchestration
│   │
│   ├── api/                            # Thin route handlers (HTTP concerns only)
│   │   ├── declarations.py             # POST /submit
│   │   ├── summary.py                  # GET /summary/{producer_id}/{month}
│   │   ├── rag.py                      # POST /ask
│   │   ├── health.py                   # GET /health
│   │   └── pages.py                    # GET /, /declare, /summary, /ask (Jinja2)
│   │
│   ├── services/                       # Business logic layer
│   │   ├── declaration_service.py      # Validation, UUID gen, DB persist
│   │   ├── reconciliation_service.py   # ERP diff engine + LLM narrative
│   │   └── llm_service.py              # Async Ollama wrapper (retry, timeout, fallback)
│   │
│   ├── core/
│   │   ├── config.py                   # pydantic-settings — all env config validated here
│   │   └── constants.py                # Shared constants (thresholds, status strings, fallbacks)
│   │
│   ├── schemas/                        # Pydantic v2 request/response models
│   │   ├── declaration.py              # DeclarationRequest / DeclarationResponse
│   │   ├── summary.py                  # ReconciliationItem / ReconciliationResponse
│   │   └── rag.py                      # AskRequest / AskResponse / Citation
│   │
│   ├── database/                       # Data access layer
│   │   ├── models.py                   # SQLAlchemy ORM model (Declaration table)
│   │   ├── repository.py               # Repository pattern — all DB queries here
│   │   └── session.py                  # SessionLocal factory + init_db()
│   │
│   ├── rag/                            # RAG pipeline modules
│   │   ├── ingest.py                   # Document loader, chunker, embedder, FAISS builder
│   │   ├── vector_store.py             # VectorStore class (FAISS + metadata.json)
│   │   └── retriever.py                # Query embedding, FAISS search, threshold filter
│   │
│   ├── prompts/                        # LLM prompt templates (editable without code changes)
│   │   ├── summary_prompt.txt          # Reconciliation narrative prompt
│   │   └── rag_prompt.txt              # RAG grounded Q&A prompt
│   │
│   ├── templates/                      # Jinja2 HTML templates (Bootstrap 5)
│   │   ├── base.html                   # Shared layout with nav
│   │   ├── index.html                  # Dashboard with Chart.js
│   │   ├── declare.html                # Declaration submission form
│   │   ├── summary.html                # Reconciliation results view
│   │   └── ask.html                    # RAG assistant chat interface
│   │
│   ├── static/                         # CSS and JavaScript assets
│   └── utils/
│       ├── erp_loader.py               # ERP CSV loader (pandas) → dict
│       └── logger.py                   # Structured logging config
│
├── documents/
│   └── epr_docs/                       # Source PDFs/TXTs for RAG ingestion
│
├── faiss_index/                        # Auto-created on first run
│   ├── index.faiss                     # FAISS binary index
│   └── metadata.json                   # Chunk metadata (source, text, preview)
│
├── mock_data/
│   └── erp_feed.csv                    # Mock ERP procurement data
│
├── Architecture Diagram/               # PlantUML diagrams
├── .env                                # Environment configuration
├── requirements.txt
├── run.py                              # Simple startup script
└── Dockerfile
```

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed locally
- (Optional) Groq API key for cloud LLM fallback

---

### Step 1 — Clone and create virtual environment

```bash
git clone <repo-url>
cd greenpack-epr

python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Configure environment

Copy `.env` and update as needed:

```bash
# .env (already included — review before running)
DB_URL=sqlite:///./greenpack.db
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=llama3.2:1b
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
FAISS_INDEX_PATH=./faiss_index
DOCS_PATH=./documents/epr_docs
ERP_CSV_PATH=./mock_data/erp_feed.csv
RETRIEVAL_TOP_K=4
RETRIEVAL_THRESHOLD=0.45
```

### Step 4 — Install Ollama and pull model

```bash
# Install Ollama from https://ollama.com/download

# Pull the local LLM
ollama pull llama3.2:1b

# Start the Ollama server (runs on http://localhost:11434)
ollama serve
```

### Step 5 — Add EPR documents (optional)

Place `.pdf` or `.txt` compliance documents in `documents/epr_docs/`. The app automatically ingests them on every startup — previously indexed files are skipped.

### Step 6 — Run the application

```bash
# Development (with hot-reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or via the convenience script
python run.py
```

### Step 7 — Access the application

| URL | Description |
|---|---|
| `http://localhost:8000` | Dashboard |
| `http://localhost:8000/declare` | Declaration submission form |
| `http://localhost:8000/summary` | Reconciliation summary viewer |
| `http://localhost:8000/ask` | RAG compliance assistant |
| `http://localhost:8000/docs` | Auto-generated Swagger UI |
| `http://localhost:8000/health` | System health check |

---

## API Endpoints

### `POST /submit` — Submit Monthly EPR Declaration

Submit a monthly plastic quantity declaration for a producer. All validation is deterministic — **no LLM is used**.

```bash
curl -X POST http://localhost:8000/submit \
  -H "Content-Type: application/json" \
  -d '{
    "producer_id": "GREENPACK-001",
    "month": "2026-04",
    "declared_quantities_kg": {
      "rigid_plastic": 12000,
      "flexible_plastic": 8500,
      "multilayer_plastic": 3200
    }
  }'
```

**Sample Response (201 Created):**
```json
{
  "record_id": "d4f3a1b2-8e9c-4d7f-a2b1-3c5e6f7d8a9b",
  "producer_id": "GREENPACK-001",
  "month": "2026-04",
  "rigid_plastic_kg": 12000.0,
  "flexible_plastic_kg": 8500.0,
  "multilayer_plastic_kg": 3200.0,
  "submitted_at": "2026-04-15T10:32:11.432Z"
}
```

---

### `GET /summary/{producer_id}/{month}` — ERP Reconciliation

Retrieve the reconciliation summary for a producer and month. Computes mismatch percentages deterministically, then generates an AI narrative via Ollama.

```bash
curl http://localhost:8000/summary/GREENPACK-001/2026-04
```

**Sample Response (200 OK):**
```json
{
  "producer_id": "GREENPACK-001",
  "month": "2026-04",
  "record_id": "d4f3a1b2-8e9c-4d7f-a2b1-3c5e6f7d8a9b",
  "items": [
    {
      "category": "rigid_plastic",
      "display_name": "Rigid Plastic",
      "declared_kg": 12000.0,
      "procured_kg": 11500.0,
      "difference_kg": 500.0,
      "difference_percent": 4.35,
      "status": "OK"
    },
    {
      "category": "flexible_plastic",
      "display_name": "Flexible Plastic",
      "declared_kg": 8500.0,
      "procured_kg": 7200.0,
      "difference_kg": 1300.0,
      "difference_percent": 18.06,
      "status": "MISMATCH"
    }
  ],
  "total_declared_kg": 23700.0,
  "total_procured_kg": 22200.0,
  "overall_status": "MISMATCH",
  "ai_summary": "GreenPack's April 2026 declaration shows an 18.06% discrepancy in flexible plastic...",
  "recommendations": "Investigate procurement records for flexible plastic. Consider reconciling..."
}
```

---

### `POST /ask` — RAG Compliance Q&A

Ask an EPR compliance question. The system retrieves relevant document chunks and generates a grounded answer. If no relevant context is found, the LLM is not called.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the EPR obligations for plastic producers under Indian regulations?"}'
```

**Sample Response — Context Found (200 OK):**
```json
{
  "question": "What are the EPR obligations for plastic producers under Indian regulations?",
  "answer": "Under the Plastic Waste Management Rules 2016 (amended 2022), producers are required to...",
  "citations": [
    {
      "source": "epr_guidelines_2024.pdf",
      "chunk": 3,
      "score": 0.74,
      "preview": "Producers, importers and brand owners shall be responsible for..."
    }
  ],
  "context_found": true
}
```

**Sample Response — No Context (200 OK):**
```json
{
  "question": "What is the capital of France?",
  "answer": "I do not know based on the provided documents.",
  "citations": [],
  "context_found": false
}
```

---

### `GET /health` — System Health Check

Check connectivity to all three external dependencies: SQLite, Ollama, and FAISS.

```bash
curl http://localhost:8000/health
```

**Sample Response — All Healthy:**
```json
{
  "status": "ok",
  "database": "connected",
  "ollama": "connected",
  "faiss": "loaded",
  "faiss_chunks": 47
}
```

**Sample Response — Degraded (Ollama down):**
```json
{
  "status": "degraded",
  "database": "connected",
  "ollama": "disconnected",
  "faiss": "loaded",
  "faiss_chunks": 47
}
```

---

### Swagger UI

Full interactive API documentation is auto-generated at:

```
http://localhost:8000/docs
```

---

## Frontend Pages

| Page | Route | Description |
|---|---|---|
| **Dashboard** | `GET /` | Overview with Chart.js bar charts comparing declared vs ERP quantities |
| **Declaration** | `GET /declare` | Multi-field form to submit monthly plastic declarations |
| **Reconciliation** | `GET /summary` | Producer/month lookup with status table and AI narrative |
| **RAG Assistant** | `GET /ask` | Chat interface for EPR compliance Q&A with source citations |

> All pages are rendered server-side via Jinja2 and styled with Bootstrap 5 dark theme. No JavaScript framework — lightweight and fast.

---

## Docker

```bash
# Build the image
docker build -t greenpack-epr .

# Run (Ollama must be accessible at host.docker.internal:11434)
docker run -p 8000:8000 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  -e GROQ_API_KEY=your_groq_key \
  -v $(pwd)/faiss_index:/app/faiss_index \
  -v $(pwd)/documents:/app/documents \
  greenpack-epr
```

> **Note:** On Windows Docker Desktop, use `host.docker.internal` to reference the host Ollama server. On Linux, use `--network=host` instead.

---

## Future Improvements

### Infrastructure
- [ ] **PostgreSQL support** — Replace SQLite with async PostgreSQL via `asyncpg` for concurrent write scale
- [ ] **Redis caching** — Cache reconciliation summaries to avoid redundant ERP CSV reads and LLM calls
- [ ] **Kubernetes deployment** — Helm chart with liveness/readiness probes wired to `/health`

### AI & RAG
- [ ] **Streaming Ollama responses** — Server-Sent Events (SSE) for perceived latency improvement
- [ ] **Semantic chunking** — Replace fixed-size chunking with spaCy sentence-boundary detection
- [ ] **FAISS IVF index** — Approximate nearest-neighbor for 1M+ chunk scale
- [ ] **OCR for scanned PDFs** — Add Tesseract/AWS Textract fallback for image-based regulatory documents
- [ ] **Hybrid retrieval** — Combine dense (FAISS) + sparse (BM25) retrieval for better recall

### Compliance Features
- [ ] **Multi-producer authentication** — JWT-based auth for producer portal
- [ ] **EPR target calculation engine** — Auto-compute annual recovery targets from declaration history
- [ ] **Certificate procurement tracking** — Track and reconcile EPR certificate purchases
- [ ] **Multi-tenant compliance support** — Isolated data partitions per producer organization
- [ ] **Audit log export** — PDF/CSV export of reconciliation reports for regulatory submission

---

## AI Coding Workflow

This project was built using an **AI-augmented engineering workflow**:

| Tool | Role |
|---|---|
| **Antigravity (Google DeepMind)** | Primary coding assistant — scaffolding, refactoring, code review |
| **Cursor / Claude Code** | Inline code generation and iterative debugging |
| **Human architect** | All architectural decisions, system design, and engineering trade-offs |

**Key AI-assisted activities:**
- Initial FastAPI scaffolding and module structure generation
- Pydantic schema drafting and refinement
- Async httpx retry logic boilerplate
- Jinja2 template generation with Bootstrap 5
- Docstring and inline comment quality improvement

**Human-led decisions:**
- Hybrid LLM architecture choice (local Ollama vs cloud Groq)
- Deterministic-only reconciliation engine (no LLM for math)
- FAISS over managed vector DBs (Chroma, Weaviate, Pinecone)
- Repository pattern adoption
- Retrieval threshold tuning (0.45)
- `metadata.json` over pickle for FAISS metadata

> AI tools accelerated scaffolding by ~60%. All critical design decisions, threshold tuning, and architecture reviews were performed by a human engineer.

---

## Design Trade-offs

| Decision | What We Chose | What We Traded Away |
|---|---|---|
| SQLite over PostgreSQL | Zero-config, single-file DB | Concurrent write scale |
| Fixed chunking over semantic | Fewer dependencies | Context coherence at sentence boundaries |
| FAISS over Chroma/Weaviate | No external server, lightweight | Rich metadata query API |
| IndexFlatIP over IVF | Exact search, no approx error | Speed at >100K chunks |
| Ollama local over OpenAI | Data privacy, zero API cost | Response quality ceiling |
| `streaming=False` | Code simplicity | Perceived latency for long responses |

---

*Built with FastAPI · SQLAlchemy · FAISS · sentence-transformers · Ollama · Groq · Bootstrap 5 · PyMuPDF · Pydantic v2*
