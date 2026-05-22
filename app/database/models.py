"""
app/database/models.py
----------------------
SQLAlchemy ORM models for the GreenPack EPR compliance system.

Architecture note: Models are kept strictly as data definitions (columns,
relationships, table names). No business logic lives here. Validation is
handled by Pydantic schemas in app/schemas/.
"""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, String
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""
    pass


class Declaration(Base):
    """
    Stores monthly plastic quantity declarations submitted by producers.

    Each declaration covers one producer for one calendar month. Quantities
    are stored per plastic category as defined in constants.PLASTIC_CATEGORIES.
    """

    __tablename__ = "declarations"

    # Primary key — UUID generated at submission time, not by DB
    record_id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True,
    )

    # Producer and period identifiers
    producer_id = Column(String(50), nullable=False, index=True)
    month = Column(String(7), nullable=False, index=True)  # Format: YYYY-MM

    # Declared plastic quantities in kilograms
    rigid_plastic_kg = Column(Float, nullable=False, default=0.0)
    flexible_plastic_kg = Column(Float, nullable=False, default=0.0)
    multilayer_plastic_kg = Column(Float, nullable=False, default=0.0)

    # Audit timestamp — set at insert time, never updated
    submitted_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    def __repr__(self) -> str:
        return (
            f"<Declaration record_id={self.record_id!r} "
            f"producer={self.producer_id!r} month={self.month!r}>"
        )
