"""
app/api/rag.py
--------------
API router for RAG-based EPR compliance Q&A.

Route: POST /ask

Workflow:
  1. Embed the question
  2. Search FAISS for top-k relevant chunks
  3. Filter by confidence threshold
  4. If no chunks pass threshold → return hardcoded fallback (NO LLM call)
  5. If chunks available → build prompt → call Ollama → return answer + citations

Architecture note: The LLM is NOT called when no sufficient context is found.
This is the key hallucination prevention mechanism in the RAG pipeline.
The fallback string is hardcoded in constants.RAG_NO_CONTEXT_RESPONSE.
"""

from fastapi import APIRouter, HTTPException, status

from app.core.constants import RAG_NO_CONTEXT_RESPONSE
from app.rag.retriever import get_chunk_texts, retrieve_relevant_chunks
from app.schemas.rag import AskRequest, AskResponse, Citation
from app.services.llm_service import ask_ollama, load_prompt_template
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["RAG Q&A"])


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask an EPR Compliance Question",
    description=(
        "RAG-powered Q&A using local EPR compliance documents. "
        "Embeds question → searches FAISS → filters by confidence threshold → "
        "answers using Ollama llama3.2:1b grounded in retrieved context. "
        "Returns 'I do not know based on the provided documents' "
        "if no sufficient context is found."
    ),
)
async def ask_question(request: AskRequest) -> AskResponse:
    """
    POST /ask — RAG-based EPR compliance question answering.

    The system will ONLY answer using retrieved document context.
    If the retrieved context is insufficient (score below threshold),
    the LLM is not called and the fallback message is returned directly.
    """
    try:
        logger.info("RAG query received: '%s...'", request.question[:80])

        # Step 1 & 2: Embed + retrieve
        citations: list[Citation] = retrieve_relevant_chunks(request.question)

        # Step 3: No context above threshold → return fallback (no LLM)
        if not citations:
            logger.info(
                "No chunks above confidence threshold for question: '%s...'",
                request.question[:60],
            )
            return AskResponse(
                question=request.question,
                answer=RAG_NO_CONTEXT_RESPONSE,
                citations=[],
                context_found=False,
            )

        # Step 4: Retrieve full chunk texts for context
        chunk_texts = get_chunk_texts(citations)

        # Build numbered context block for the prompt
        context_parts = []
        for i, (citation, text) in enumerate(zip(citations, chunk_texts), start=1):
            context_parts.append(
                f"[{i}] Source: {citation.source} (chunk {citation.chunk}, "
                f"relevance: {citation.score:.2f})\n{text}"
            )
        context_block = "\n\n---\n\n".join(context_parts)

        # Step 5: Build prompt and call Ollama
        prompt_template = load_prompt_template("rag_prompt.txt")
        prompt = prompt_template.format(
            context=context_block,
            question=request.question,
        )

        logger.info(
            "Calling Ollama with %d context chunks for question: '%s...'",
            len(citations),
            request.question[:60],
        )

        answer = await ask_ollama(prompt)

        # If LLM returned the fallback string (Ollama unavailable), mark context_found=False
        context_found = answer != RAG_NO_CONTEXT_RESPONSE

        return AskResponse(
            question=request.question,
            answer=answer,
            citations=citations if context_found else [],
            context_found=context_found,
        )

    except Exception as exc:
        logger.error("Unexpected error in RAG Q&A: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during question answering.",
        ) from exc
