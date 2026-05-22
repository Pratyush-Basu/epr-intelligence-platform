"""
app/api/declarations.py
-----------------------
API router for EPR declaration submission.

Architecture: Routes are intentionally thin — they handle HTTP concerns
(request parsing, response formatting, error codes) and delegate all
business logic to the service layer.

Route: POST /submit
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.schemas.declaration import DeclarationRequest, DeclarationResponse
from app.services import declaration_service
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Declarations"])


@router.post(
    "/submit",
    response_model=DeclarationResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit Monthly EPR Declaration",
    description=(
        "Submit a monthly plastic quantity declaration for a producer. "
        "Validates format, prevents negatives, generates UUID, persists to SQLite. "
        "**No LLM is used in this endpoint** — all validation is deterministic."
    ),
)
async def submit_declaration(
    request: DeclarationRequest,
    db: Session = Depends(get_db),
) -> DeclarationResponse:
    """
    POST /submit — Submit a monthly EPR plastic quantity declaration.

    This endpoint is intentionally LLM-free. Compliance data submission
    must be deterministic, auditable, and fast. The LLM is reserved for
    narrative generation in the /summary endpoint.
    """
    try:
        logger.info(
            "Declaration submission received: producer=%s month=%s",
            request.producer_id,
            request.month,
        )
        return declaration_service.create_declaration(request, db)

    except ValueError as exc:
        logger.warning("Declaration validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.error("Unexpected error in declaration submission: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while saving the declaration.",
        ) from exc
