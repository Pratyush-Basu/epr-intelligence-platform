"""
app/rag/ingest.py
-----------------
Document ingestion pipeline: file reading → chunking → embedding → FAISS indexing.

Supports PDF files (via PyMuPDF) and plain text files (.txt).

Architecture note: Ingestion tracks processed files by a simple hash set stored
in ingested_files.json. This prevents re-embedding documents that haven't changed
across server restarts. The hash is computed from the file path + modification
time, so re-uploading a modified file correctly triggers re-ingestion.

Design tradeoff: We use fixed-size character chunking with overlap rather than
sentence-aware chunking. This keeps the dependency list minimal and avoids NLTK
punkt model downloads. For production, consider semantic chunking with spaCy.
"""

import hashlib
import json
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.constants import FAISS_METADATA_FILE, SUPPORTED_DOC_EXTENSIONS
from app.rag.vector_store import VectorStore, get_vector_store
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level embedding model (loaded once, reused across ingest + retrieval)
# ---------------------------------------------------------------------------
_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """
    Lazy-load the sentence-transformers model.

    Loading is deferred to first call so that import time remains fast.
    The model is ~80MB and CPU-compatible — no GPU required.
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model: sentence-transformers/all-MiniLM-L6-v2")
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Embedding model loaded successfully")
    return _embedding_model


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def extract_text_from_pdf(file_path: Path) -> str:
    """
    Extract all text from a PDF file using PyMuPDF.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Concatenated text from all pages, separated by newlines.

    Raises:
        RuntimeError: If the PDF cannot be opened or parsed.
    """
    try:
        doc = fitz.open(str(file_path))
        pages_text = [page.get_text("text") for page in doc]
        doc.close()
        return "\n".join(pages_text)
    except Exception as exc:
        raise RuntimeError(f"Failed to extract text from '{file_path}': {exc}") from exc


def extract_text_from_txt(file_path: Path) -> str:
    """Read a plain text file and return its contents."""
    return file_path.read_text(encoding="utf-8", errors="replace")


def extract_text(file_path: Path) -> str:
    """
    Route text extraction based on file extension.

    Raises:
        ValueError: If the file extension is not supported.
    """
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".txt":
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {SUPPORTED_DOC_EXTENSIONS}"
        )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Split text into fixed-size overlapping character chunks.

    Args:
        text: The full document text.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Character overlap between consecutive chunks.

    Returns:
        List of text chunks. Empty list if text is empty.
    """
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


# ---------------------------------------------------------------------------
# File tracking (skip already-ingested files)
# ---------------------------------------------------------------------------

INGESTED_REGISTRY_FILE = "ingested_files.json"


def _get_file_hash(file_path: Path) -> str:
    """Compute a unique hash for a file based on path + modification time."""
    stat = file_path.stat()
    key = f"{file_path.resolve()}|{stat.st_mtime}"
    return hashlib.md5(key.encode()).hexdigest()


def _load_ingested_registry(index_dir: Path) -> set[str]:
    """Load the set of already-ingested file hashes from disk."""
    registry_path = index_dir / INGESTED_REGISTRY_FILE
    if registry_path.exists():
        with open(registry_path, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def _save_ingested_registry(index_dir: Path, hashes: set[str]) -> None:
    """Persist the set of ingested file hashes to disk."""
    index_dir.mkdir(parents=True, exist_ok=True)
    registry_path = index_dir / INGESTED_REGISTRY_FILE
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(list(hashes), f, indent=2)


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------


def ingest_documents(
    docs_path: Path | None = None,
    index_dir: Path | None = None,
    force: bool = False,
) -> int:
    """
    Ingest all supported documents from docs_path into the FAISS vector store.

    Steps:
      1. Scan docs_path for supported files (.pdf, .txt)
      2. Skip files already in the ingested registry (unless force=True)
      3. Extract text → chunk → embed → add to FAISS index
      4. Persist updated index and registry to disk

    Args:
        docs_path: Directory containing source documents. Defaults to settings.DOCS_PATH.
        index_dir: Directory to store FAISS index files. Defaults to settings.FAISS_INDEX_PATH.
        force: If True, re-ingest all files even if already indexed.

    Returns:
        Number of new documents ingested in this run.
    """
    docs_dir = docs_path or Path(settings.DOCS_PATH)
    idx_dir = index_dir or Path(settings.FAISS_INDEX_PATH)

    if not docs_dir.exists():
        logger.warning("Documents directory '%s' does not exist — skipping ingest", docs_dir)
        return 0

    # Collect supported files
    files: list[Path] = [
        f for f in docs_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_DOC_EXTENSIONS
    ]

    if not files:
        logger.warning("No supported documents found in '%s'", docs_dir)
        return 0

    logger.info("Found %d document(s) in '%s'", len(files), docs_dir)

    # Load embedding model and vector store
    model = get_embedding_model()
    store: VectorStore = get_vector_store()

    # Load or create the FAISS index
    if not store.is_loaded:
        loaded = store.load(idx_dir)
        if not loaded:
            logger.info("Creating new FAISS index")

    # Load file tracking registry
    ingested_hashes = _load_ingested_registry(idx_dir) if not force else set()
    new_count = 0

    for file_path in files:
        file_hash = _get_file_hash(file_path)

        if file_hash in ingested_hashes:
            logger.debug("Skipping already-ingested file: %s", file_path.name)
            continue

        logger.info("Ingesting: %s", file_path.name)

        try:
            raw_text = extract_text(file_path)
        except Exception as exc:
            logger.error("Failed to extract text from '%s': %s", file_path.name, exc)
            continue

        chunks = chunk_text(
            raw_text,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

        if not chunks:
            logger.warning("No chunks produced from '%s' — skipping", file_path.name)
            continue

        logger.info("  → %d chunks produced", len(chunks))

        # Generate embeddings for all chunks at once (batched for efficiency)
        embeddings: np.ndarray = model.encode(
            chunks,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2-normalize for cosine similarity via IndexFlatIP
        )

        # Build metadata entries
        metadata_entries = [
            {
                "chunk_id": store.total_chunks + i,
                "source": file_path.name,
                "chunk_index": i,
                "text_preview": chunk[:150],
                "text": chunk,
            }
            for i, chunk in enumerate(chunks)
        ]

        store.add_vectors(embeddings.astype(np.float32), metadata_entries)
        ingested_hashes.add(file_hash)
        new_count += 1
        logger.info("  → Ingested '%s' (%d chunks)", file_path.name, len(chunks))

    # Persist updated index and registry
    if new_count > 0:
        store.save(idx_dir)
        _save_ingested_registry(idx_dir, ingested_hashes)
        logger.info(
            "Ingestion complete: %d new document(s), %d total vectors in index",
            new_count,
            store.total_chunks,
        )
    else:
        logger.info("No new documents to ingest (all files already indexed)")

    return new_count
