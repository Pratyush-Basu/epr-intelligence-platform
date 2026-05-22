"""
app/rag/vector_store.py
-----------------------
FAISS index lifecycle management — load, save, and state tracking.

Architecture note: Metadata is stored as human-readable JSON (not pickle)
alongside the FAISS binary index. This makes the vector store debuggable —
you can inspect metadata.json directly to see what chunks are indexed,
their sources, and text previews without running any Python code.

Design tradeoff: We use IndexFlatIP (inner product) with L2-normalized
vectors, which gives cosine similarity scores in [-1, 1]. This is preferable
to IndexFlatL2 for retrieval quality comparisons and threshold filtering,
because cosine similarity has intuitive semantics (1.0 = identical).
"""

import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from app.core.constants import FAISS_INDEX_FILE, FAISS_METADATA_FILE
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    Wrapper around a FAISS IndexFlatIP index with JSON metadata sidecar.

    Each vector in the FAISS index corresponds to one metadata entry at the
    same integer position. Adding vectors and metadata must always be done
    together via add_vectors() to maintain this invariant.
    """

    def __init__(self, dimension: int = 384) -> None:
        """
        Initialize an in-memory FAISS index.

        Args:
            dimension: Embedding dimension. all-MiniLM-L6-v2 produces 384-d vectors.
        """
        self.dimension = dimension
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self.metadata: list[dict] = []  # Parallel list to FAISS index entries
        self._is_loaded: bool = False

    @property
    def is_loaded(self) -> bool:
        """True if the index has been loaded from disk or has vectors added."""
        return self._is_loaded

    @property
    def total_chunks(self) -> int:
        """Total number of vectors (chunks) stored in this index."""
        return self.index.ntotal

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata_entries: list[dict],
    ) -> None:
        """
        Add normalized vectors and their metadata to the store.

        Args:
            vectors: Float32 numpy array of shape (N, dimension), L2-normalized.
            metadata_entries: List of N metadata dicts with keys:
                              chunk_id, source, chunk_index, text_preview.

        Raises:
            ValueError: If vectors and metadata_entries have mismatched lengths.
        """
        if len(vectors) != len(metadata_entries):
            raise ValueError(
                f"Vector count ({len(vectors)}) must match "
                f"metadata count ({len(metadata_entries)})"
            )

        self.index.add(vectors.astype(np.float32))
        self.metadata.extend(metadata_entries)
        self._is_loaded = True
        logger.debug("Added %d vectors to FAISS index (total: %d)", len(vectors), self.index.ntotal)

    def search(
        self, query_vector: np.ndarray, top_k: int
    ) -> list[tuple[float, dict]]:
        """
        Search the index for the most similar vectors.

        Args:
            query_vector: L2-normalized float32 array of shape (1, dimension).
            top_k: Maximum number of results to return.

        Returns:
            List of (score, metadata_dict) tuples ordered by score descending.
            Score is cosine similarity in range [-1, 1].
        """
        if self.index.ntotal == 0:
            logger.warning("FAISS search called on empty index")
            return []

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vector.astype(np.float32), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for padding when k > ntotal
                continue
            results.append((float(score), self.metadata[idx]))

        return results

    def save(self, index_dir: Path) -> None:
        """
        Persist the FAISS index and metadata to disk.

        Saves two files:
          - {index_dir}/index.faiss  — binary FAISS index
          - {index_dir}/metadata.json — human-readable chunk metadata
        """
        index_dir.mkdir(parents=True, exist_ok=True)

        index_path = index_dir / FAISS_INDEX_FILE
        metadata_path = index_dir / FAISS_METADATA_FILE

        faiss.write_index(self.index, str(index_path))

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        logger.info(
            "FAISS index saved to %s (%d vectors, %d metadata entries)",
            index_dir,
            self.index.ntotal,
            len(self.metadata),
        )

    def load(self, index_dir: Path) -> bool:
        """
        Load a persisted FAISS index and metadata from disk.

        Args:
            index_dir: Directory containing index.faiss and metadata.json.

        Returns:
            True if successfully loaded, False if files do not exist.
        """
        index_path = index_dir / FAISS_INDEX_FILE
        metadata_path = index_dir / FAISS_METADATA_FILE

        if not index_path.exists() or not metadata_path.exists():
            logger.info("No persisted FAISS index found at %s", index_dir)
            return False

        self.index = faiss.read_index(str(index_path))

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self._is_loaded = True
        logger.info(
            "FAISS index loaded from %s (%d vectors)",
            index_dir,
            self.index.ntotal,
        )
        return True


# ---------------------------------------------------------------------------
# Module-level singleton — shared across the application lifecycle
# ---------------------------------------------------------------------------
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """
    Return the application-level VectorStore singleton.

    Created lazily on first call; subsequent calls return the same instance.
    This ensures there is exactly one FAISS index in memory per process.
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore(dimension=384)
    return _vector_store
