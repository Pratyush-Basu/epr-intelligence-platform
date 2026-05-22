"""
app/database/repository.py
--------------------------
Data access layer — all database read/write operations live here.

Architecture note: Repository pattern isolates all DB I/O from the service
layer. Services call repository functions; they never access the ORM session
directly. This separation means:
  1. Business logic is testable without a real DB.
  2. Swapping SQLite for PostgreSQL requires changes only in this file.
  3. Query logic is co-located and easy to audit.
"""

from datetime import datetime

from sqlalchemy.orm import Session

from app.database.models import Declaration
from app.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Declaration repository functions
# ---------------------------------------------------------------------------


def save_declaration(db: Session, declaration: Declaration) -> Declaration:
    """
    Persist a new Declaration record to the database.

    Args:
        db: Active SQLAlchemy session (injected via Depends).
        declaration: Pre-constructed Declaration ORM instance.

    Returns:
        The persisted Declaration with any DB-generated fields populated.
    """
    db.add(declaration)
    db.commit()
    db.refresh(declaration)
    logger.info(
        "Declaration saved: record_id=%s producer=%s month=%s",
        declaration.record_id,
        declaration.producer_id,
        declaration.month,
    )
    return declaration


def get_declaration(
    db: Session, producer_id: str, month: str
) -> Declaration | None:
    """
    Retrieve the most recent declaration for a producer/month pair.

    Args:
        db: Active SQLAlchemy session.
        producer_id: Producer identifier.
        month: Month string in 'YYYY-MM' format.

    Returns:
        The Declaration ORM instance, or None if not found.
    """
    result = (
        db.query(Declaration)
        .filter(
            Declaration.producer_id == producer_id,
            Declaration.month == month,
        )
        .order_by(Declaration.submitted_at.desc())
        .first()
    )

    if result:
        logger.debug(
            "Declaration found: record_id=%s for producer=%s month=%s",
            result.record_id,
            producer_id,
            month,
        )
    else:
        logger.warning(
            "No declaration found for producer=%s month=%s",
            producer_id,
            month,
        )

    return result


def get_all_declarations(db: Session, limit: int = 20) -> list[Declaration]:
    """
    Retrieve the most recent declarations across all producers.

    Args:
        db: Active SQLAlchemy session.
        limit: Maximum number of records to return.

    Returns:
        List of Declaration ORM instances ordered by submission time (desc).
    """
    return (
        db.query(Declaration)
        .order_by(Declaration.submitted_at.desc())
        .limit(limit)
        .all()
    )


def count_declarations(db: Session) -> int:
    """Return total count of all declaration records in the database."""
    return db.query(Declaration).count()


def declaration_exists(db: Session, producer_id: str, month: str) -> bool:
    """Check whether a declaration exists for a given producer/month."""
    return (
        db.query(Declaration.record_id)
        .filter(
            Declaration.producer_id == producer_id,
            Declaration.month == month,
        )
        .first()
        is not None
    )
