"""
app/api/health.py
-----------------
System health check endpoint.

Route: GET /health

Validates connectivity to all three external dependencies:
  - SQLite database
  - Ollama LLM API
  - FAISS vector index

Architecture note: Health checks are essential for Docker/k8s liveness probes
and for debugging deployment issues. Each component is checked independently
so partial degradation is surfaced clearly.
"""

from fastapi import APIRouter
from sqlalchemy import text

from app.core.constants import (
    HEALTH_CONNECTED,
    HEALTH_DEGRADED,
    HEALTH_DISCONNECTED,
    HEALTH_LOADED,
    HEALTH_NOT_LOADED,
    HEALTH_OK,
)
from app.database.session import SessionLocal
from app.rag.vector_store import get_vector_store
from app.services.llm_service import check_ollama_health
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="System Health Check",
    description=(
        "Check connectivity to database, Ollama LLM, and FAISS vector index. "
        "Returns 'ok' if all components are healthy, 'degraded' otherwise."
    ),
)
async def health_check() -> dict:
    """
    GET /health — System health check for all dependencies.

    Returns a JSON object with individual component status and an overall
    system status. HTTP 200 is always returned (status field indicates health).
    """
    results = {}

    # --- Check 1: SQLite database ---
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        results["database"] = HEALTH_CONNECTED
    except Exception as exc:
        logger.error("Database health check failed: %s", exc)
        results["database"] = HEALTH_DISCONNECTED

    # --- Check 2: Ollama LLM ---
    try:
        ollama_ok = await check_ollama_health()
        results["ollama"] = HEALTH_CONNECTED if ollama_ok else HEALTH_DISCONNECTED
    except Exception as exc:
        logger.error("Ollama health check failed: %s", exc)
        results["ollama"] = HEALTH_DISCONNECTED

    # --- Check 3: FAISS vector index ---
    try:
        store = get_vector_store()
        if store.is_loaded and store.total_chunks > 0:
            results["faiss"] = HEALTH_LOADED
            results["faiss_chunks"] = store.total_chunks
        else:
            results["faiss"] = HEALTH_NOT_LOADED
            results["faiss_chunks"] = 0
    except Exception as exc:
        logger.error("FAISS health check failed: %s", exc)
        results["faiss"] = HEALTH_NOT_LOADED
        results["faiss_chunks"] = 0

    # --- Overall status ---
    all_healthy = (
        results["database"] == HEALTH_CONNECTED
        and results["ollama"] == HEALTH_CONNECTED
        and results["faiss"] == HEALTH_LOADED
    )
    results["status"] = HEALTH_OK if all_healthy else HEALTH_DEGRADED

    logger.info("Health check result: %s", results)
    return results
