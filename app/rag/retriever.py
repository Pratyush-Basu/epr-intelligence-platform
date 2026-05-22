"""
app/rag/retriever.py
--------------------
Query-time retrieval: embed question → search FAISS → threshold filter → return chunks.

Architecture note: Retrieval confidence thresholding is enforced here, not in the
API layer. If no chunk scores above RETRIEVAL_THRESHOLD, the API receives an empty
list and must return the hardcoded fallback string — the LLM is NOT called.

This is a critical hallucination prevention mechanism: the LLM only generates
an answer when there is sufficient documentary evidence to ground its response.
"""

import numpy as np

from app.core.config import settings
from app.rag.ingest import get_embedding_model
from app.rag.vector_store import VectorStore, get_vector_store
from app.schemas.rag import Citation
from app.utils.logger import get_logger

logger = get_logger(__name__)


def retrieve_relevant_chunks(
    question: str,
    top_k: int | None = None,
    threshold: float | None = None,
) -> list[Citation]:
    """
    Retrieve the most relevant document chunks for a given question.

    Pipeline:
      1. Embed the question using sentence-transformers/all-MiniLM-L6-v2
      2. Search the FAISS index for top-k most similar chunks
      3. Filter results below the confidence threshold
      4. Return structured Citation objects

    Args:
        question: The user's compliance question.
        top_k: Number of candidates to retrieve before threshold filtering.
               Defaults to settings.RETRIEVAL_TOP_K.
        threshold: Minimum cosine similarity to include a chunk.
                   Defaults to settings.RETRIEVAL_THRESHOLD.

    Returns:
        List of Citation objects with source, chunk index, score, and preview.
        Empty list if no chunks pass the confidence threshold.
    """
    k = top_k or settings.RETRIEVAL_TOP_K
    min_score = threshold if threshold is not None else settings.RETRIEVAL_THRESHOLD

    store: VectorStore = get_vector_store()

    if not store.is_loaded or store.total_chunks == 0:
        logger.warning(
            "Retrieval attempted on empty/unloaded FAISS index. "
            "Ensure ingest_documents() was called at startup."
        )
        return []

    # Embed the query (L2-normalized for cosine similarity)
    model = get_embedding_model()
    query_embedding: np.ndarray = model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Search FAISS index
    raw_results = store.search(query_embedding, top_k=k)

    logger.debug(
        "FAISS search returned %d candidates for question: '%s...'",
        len(raw_results),
        question[:60],
    )

    # Apply confidence threshold filter
    citations: list[Citation] = []
    for score, meta in raw_results:
        logger.debug(
            "  Chunk '%s' (chunk %d): score=%.4f (threshold=%.4f) → %s",
            meta.get("source", "?"),
            meta.get("chunk_index", -1),
            score,
            min_score,
            "PASS" if score >= min_score else "REJECT",
        )

        if score < min_score:
            continue

        citations.append(
            Citation(
                source=meta.get("source", "unknown"),
                chunk=meta.get("chunk_index", 0),
                score=round(score, 4),
                preview=meta.get("text_preview", "")[:150],
            )
        )

    logger.info(
        "Retrieval result: %d/%d chunks passed threshold %.2f for question '%s...'",
        len(citations),
        len(raw_results),
        min_score,
        question[:50],
    )

    return citations


def get_chunk_texts(citations: list[Citation]) -> list[str]:
    """
    Retrieve the full text of chunks corresponding to the given citations.

    This is a secondary lookup into the vector store metadata to get the
    complete chunk text (the Citation only stores a 150-char preview).

    Args:
        citations: List of Citation objects from retrieve_relevant_chunks().

    Returns:
        List of full chunk texts in the same order as citations.
    """
    store: VectorStore = get_vector_store()
    texts = []

    for citation in citations:
        # Find the metadata entry matching this source + chunk_index
        chunk_text = ""
        for meta in store.metadata:
            if (
                meta.get("source") == citation.source
                and meta.get("chunk_index") == citation.chunk
            ):
                chunk_text = meta.get("text", meta.get("text_preview", ""))
                break

        if not chunk_text:
            logger.warning(
                "Could not find full text for %s chunk %d",
                citation.source,
                citation.chunk,
            )

        texts.append(chunk_text)

    return texts
