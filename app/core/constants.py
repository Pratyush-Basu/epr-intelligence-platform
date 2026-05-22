"""
app/core/constants.py
---------------------
Application-wide constants.

Architecture note: All magic strings, threshold values, and repeated literals
are stored here. This makes the codebase easier to audit and maintain — if a
threshold changes, it changes in exactly one place.
"""

from typing import Final

# ---------------------------------------------------------------------------
# Plastic category identifiers (must match DB column names and CSV headers)
# ---------------------------------------------------------------------------
PLASTIC_CATEGORIES: Final[list[str]] = [
    "rigid_plastic",
    "flexible_plastic",
    "multilayer_plastic",
]

CATEGORY_DISPLAY_NAMES: Final[dict[str, str]] = {
    "rigid_plastic": "Rigid Plastic",
    "flexible_plastic": "Flexible Plastic",
    "multilayer_plastic": "Multilayer Plastic",
}

# ---------------------------------------------------------------------------
# Compliance thresholds
# ---------------------------------------------------------------------------
MISMATCH_THRESHOLD_PERCENT: Final[float] = 5.0
"""
Declared vs procured difference above this % triggers MISMATCH status.
This is a deterministic business rule — never delegated to the LLM.
"""

# ---------------------------------------------------------------------------
# Reconciliation status labels
# ---------------------------------------------------------------------------
STATUS_OK: Final[str] = "OK"
STATUS_MISMATCH: Final[str] = "MISMATCH"

# ---------------------------------------------------------------------------
# RAG / LLM constants
# ---------------------------------------------------------------------------
RAG_NO_CONTEXT_RESPONSE: Final[str] = (
    "I do not know based on the provided documents"
)
"""
Exact fallback string returned when retrieval confidence is below threshold.
Hardcoded to prevent hallucination — the LLM is NOT invoked in this case.
"""

LLM_UNAVAILABLE_RESPONSE: Final[str] = (
    "AI summary is currently unavailable. "
    "Please ensure Ollama is running with model llama3.2:1b."
)

# ---------------------------------------------------------------------------
# Health check status labels
# ---------------------------------------------------------------------------
HEALTH_OK: Final[str] = "ok"
HEALTH_DEGRADED: Final[str] = "degraded"
HEALTH_CONNECTED: Final[str] = "connected"
HEALTH_DISCONNECTED: Final[str] = "disconnected"
HEALTH_LOADED: Final[str] = "loaded"
HEALTH_NOT_LOADED: Final[str] = "not_loaded"

# ---------------------------------------------------------------------------
# FAISS metadata file names
# ---------------------------------------------------------------------------
FAISS_INDEX_FILE: Final[str] = "index.faiss"
FAISS_METADATA_FILE: Final[str] = "metadata.json"

# ---------------------------------------------------------------------------
# Supported document extensions for ingestion
# ---------------------------------------------------------------------------
SUPPORTED_DOC_EXTENSIONS: Final[set[str]] = {".pdf", ".txt"}
