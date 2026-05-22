"""
app/main.py
-----------
FastAPI application factory with modern lifespan management.

Architecture note: We use the @asynccontextmanager lifespan pattern (FastAPI 0.93+)
instead of deprecated @app.on_event("startup") decorators. The lifespan context
manager guarantees that startup and shutdown logic are co-located and that
resources are properly cleaned up even on unexpected shutdowns.

Startup responsibilities (in order):
  1. Configure logging
  2. Initialize database tables
  3. Load or create FAISS index
  4. Auto-ingest documents from docs/ directory

Middleware:
  - Request/response logging with timing
  - CORS (permissive for local development)
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app.api import declarations, health, pages, rag, summary
from app.core.config import settings
from app.database.session import init_db
from app.rag.ingest import ingest_documents
from app.rag.vector_store import get_vector_store
from app.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


# =============================================================================
# Lifespan — startup and shutdown orchestration
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI lifespan context manager.

    Replaces deprecated @app.on_event("startup") / @app.on_event("shutdown").
    Everything before `yield` runs at startup; after `yield` on shutdown.
    """
    # --- Startup ---
    configure_logging()
    logger.info("=" * 60)
    logger.info("GreenPack EPR Compliance System — Starting Up")
    logger.info("=" * 60)

    # Step 1: Initialize database
    logger.info("[1/3] Initializing database...")
    init_db()

    # Step 2: Load or create FAISS index
    logger.info("[2/3] Loading FAISS vector index...")
    store = get_vector_store()
    loaded = store.load(Path(settings.FAISS_INDEX_PATH))
    if loaded:
        logger.info("      FAISS index loaded (%d chunks)", store.total_chunks)
    else:
        logger.info("      No existing FAISS index — will build from documents")

    # Step 3: Auto-ingest documents
    logger.info("[3/3] Scanning documents for ingestion...")
    new_docs = ingest_documents(
        docs_path=Path(settings.DOCS_PATH),
        index_dir=Path(settings.FAISS_INDEX_PATH),
    )
    logger.info(
        "      Ingestion complete: %d new document(s) processed, "
        "%d total vectors in index",
        new_docs,
        get_vector_store().total_chunks,
    )

    logger.info("=" * 60)
    logger.info("System ready — serving at http://localhost:8000")
    logger.info("API docs:   http://localhost:8000/docs")
    logger.info("Health:     http://localhost:8000/health")
    logger.info("=" * 60)

    yield  # --- Application runs here ---

    # --- Shutdown ---
    logger.info("GreenPack EPR System shutting down gracefully")


# =============================================================================
# Application factory
# =============================================================================

app = FastAPI(
    title="GreenPack Industries — EPR Compliance System",
    description=(
        "AI-powered EPR (Extended Producer Responsibility) compliance workflow system. "
        "Features: monthly declaration submission, ERP reconciliation with AI summary, "
        "and RAG-based compliance Q&A using local Ollama + FAISS.\n\n"
        "**All AI computations run locally — no cloud APIs required.**"
    ),
    version="1.0.0",
    contact={
        "name": "GreenPack Industries Compliance Team",
        "email": "compliance@greenpack.in",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan,
)


# =============================================================================
# Middleware
# =============================================================================

# CORS — permissive for local development; restrict origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """
    Log all HTTP requests with method, path, status code, and processing time.
    This gives full request visibility without a separate APM tool.
    """
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "%s %s → %d [%.1fms]",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# =============================================================================
# Exception handlers
# =============================================================================

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors with a structured response."""
    logger.warning("Validation error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": exc.errors(),
            "path": str(request.url.path),
        },
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Return structured JSON for 404 errors on API routes."""
    if request.url.path.startswith("/docs") or request.url.path.startswith("/openapi"):
        raise exc
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "detail": f"No route matches '{request.method} {request.url.path}'",
        },
    )


# =============================================================================
# Static files and routers
# =============================================================================

# Mount static assets (CSS, JS)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# API routers — exact endpoint names per assignment spec
app.include_router(declarations.router)   # POST /submit
app.include_router(summary.router)        # GET /summary/{producer_id}/{month}
app.include_router(rag.router)            # POST /ask
app.include_router(health.router)         # GET /health
app.include_router(pages.router)          # GET /, /declare, /summary, /ask
