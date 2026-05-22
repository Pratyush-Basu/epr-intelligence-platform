"""
app/services/declaration_service.py
------------------------------------
Business logic for EPR monthly declaration submission.

Architecture note: No LLM is used here. Declaration submission is a
deterministic data operation: validate → generate ID → persist → return.
Using an LLM for any part of this process would introduce non-determinism
and latency where none is needed. Pydantic validation in the schema layer
catches all input errors before they reach this service.
"""

import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from app.database.models import Declaration
from app.database import repository
from app.schemas.declaration import (
    DeclarationRequest,
    DeclarationResponse,
    DeclaredQuantities,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


def create_declaration(
    request: DeclarationRequest,
    db: Session,
) -> DeclarationResponse:
    """
    Create and persist a new monthly EPR declaration.

    Steps:
      1. Generate a UUID record_id
      2. Build the ORM model instance
      3. Persist via repository (no direct ORM calls in services)
      4. Return a structured response schema

    Args:
        request: Validated DeclarationRequest from the API route.
        db: Active database session (injected by FastAPI Depends).

    Returns:
        DeclarationResponse with the persisted record details.
    """
    record_id = str(uuid.uuid4())
    submitted_at = datetime.now(ZoneInfo("Asia/Kolkata"))

    logger.info(
        "Creating declaration: producer=%s month=%s record_id=%s",
        request.producer_id,
        request.month,
        record_id,
    )

    # Build the ORM model (no business logic — pure data mapping)
    declaration = Declaration(
        record_id=record_id,
        producer_id=request.producer_id,
        month=request.month,
        rigid_plastic_kg=request.declared_quantities_kg.rigid_plastic,
        flexible_plastic_kg=request.declared_quantities_kg.flexible_plastic,
        multilayer_plastic_kg=request.declared_quantities_kg.multilayer_plastic,
        submitted_at=submitted_at,
    )

    # Persist via repository — service never touches Session directly
    saved = repository.save_declaration(db, declaration)

    logger.info(
        "Declaration persisted successfully: record_id=%s",
        saved.record_id,
    )

    # Map ORM model back to Pydantic response
    return DeclarationResponse(
        record_id=saved.record_id,
        producer_id=saved.producer_id,
        month=saved.month,
        declared_quantities_kg=DeclaredQuantities(
            rigid_plastic=saved.rigid_plastic_kg,
            flexible_plastic=saved.flexible_plastic_kg,
            multilayer_plastic=saved.multilayer_plastic_kg,
        ),
        submitted_at=saved.submitted_at,
    )


def get_recent_declarations(
    db: Session,
    limit: int = 10,
) -> list[DeclarationResponse]:
    """
    Retrieve the most recent declarations for the dashboard.

    Args:
        db: Active database session.
        limit: Maximum number of records to return.

    Returns:
        List of DeclarationResponse objects ordered by submission time (desc).
    """
    declarations = repository.get_all_declarations(db, limit=limit)

    return [
        DeclarationResponse(
            record_id=d.record_id,
            producer_id=d.producer_id,
            month=d.month,
            declared_quantities_kg=DeclaredQuantities(
                rigid_plastic=d.rigid_plastic_kg,
                flexible_plastic=d.flexible_plastic_kg,
                multilayer_plastic=d.multilayer_plastic_kg,
            ),
            submitted_at=d.submitted_at,
        )
        for d in declarations
    ]
