"""
app/api/summary.py
------------------
API router for ERP reconciliation summary.

Route: GET /summary/{producer_id}/{month}

This endpoint triggers the most complex workflow in the system:
  1. DB lookup (synchronous via repository)
  2. CSV read (synchronous via erp_loader)
  3. Math computation (pure Python)
  4. LLM call (async httpx to Ollama)

All steps 1-3 are deterministic. Step 4 is the only LLM-powered operation.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.schemas.summary import ReconciliationResponse
from app.services import reconciliation_service
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Reconciliation"])


@router.get(
    "/summary/{producer_id}/{month}",
    response_model=ReconciliationResponse,
    summary="Get ERP Reconciliation Summary",
    description=(
        "Compare declared plastic quantities against ERP procurement data. "
        "Returns per-category comparison with mismatch flags (>5% threshold) "
        "and an AI-generated compliance narrative from Ollama llama3.2:1b. "
        "**Math is computed in Python — the LLM only narrates pre-computed results.**"
    ),
)
async def get_summary(
    producer_id: str,
    month: str,
    db: Session = Depends(get_db),
) -> ReconciliationResponse:
    """
    GET /summary/{producer_id}/{month} — ERP reconciliation and AI compliance summary.

    Path parameters:
      - producer_id: e.g. 'GREENPACK-001'
      - month: YYYY-MM format, e.g. '2026-04'
    """
    try:
        logger.info(
            "Reconciliation requested: producer=%s month=%s",
            producer_id,
            month,
        )
        return await reconciliation_service.get_reconciliation_summary(
            producer_id=producer_id,
            month=month,
            db=db,
        )

    except ValueError as exc:
        # Declaration not found or ERP data missing
        logger.warning("Reconciliation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        # ERP CSV file missing
        logger.error("ERP data file not found: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.error("Unexpected error in reconciliation: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred during reconciliation.",
        ) from exc
