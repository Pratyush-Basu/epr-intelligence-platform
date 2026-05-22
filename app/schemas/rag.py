"""
app/schemas/rag.py
------------------
Pydantic schemas for the RAG Q&A endpoint (POST /ask).
"""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """
    A structured source citation from the RAG retrieval pipeline.

    Architecture note: Returning structured citations (not just filenames)
    allows the frontend to render rich citation cards and helps users trace
    exactly which document chunk was used to generate the answer.
    """

    source: str = Field(description="Source document filename", examples=["epr_guide.txt"])
    chunk: int = Field(description="Zero-based chunk index within the document")
    score: float = Field(
        description="Cosine similarity score (0.0-1.0, higher = more relevant)",
        examples=[0.82],
    )
    preview: str = Field(
        description="First 120 characters of the chunk text for context",
        examples=["EPR obligations require producers to register on the CPCB portal..."],
    )


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    question: str = Field(
        min_length=5,
        max_length=1000,
        description="The compliance question to answer using EPR documents",
        examples=["What are EPR obligations for plastic producers?"],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What are EPR obligations for plastic producers?"
                }
            ]
        }
    }


class AskResponse(BaseModel):
    """Response from POST /ask."""

    question: str = Field(description="The original question")
    answer: str = Field(
        description=(
            "LLM answer grounded in retrieved document context, "
            "or 'I do not know based on the provided documents' if no context found"
        )
    )
    citations: list[Citation] = Field(
        description="Source chunks used to generate the answer",
        default_factory=list,
    )
    context_found: bool = Field(
        description="Whether sufficient context was retrieved above the confidence threshold"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What are EPR obligations for plastic producers?",
                    "answer": "All plastic producers must register with CPCB...",
                    "citations": [
                        {
                            "source": "epr_guide.txt",
                            "chunk": 3,
                            "score": 0.82,
                            "preview": "All plastic producers must register on the EPR portal...",
                        }
                    ],
                    "context_found": True,
                }
            ]
        }
    }
