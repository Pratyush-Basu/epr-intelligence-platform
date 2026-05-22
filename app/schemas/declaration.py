"""
app/schemas/declaration.py
--------------------------
Pydantic schemas for declaration submission and response.

Validation rules:
  - month must match YYYY-MM format (regex enforced)
  - All quantity values must be >= 0 (non-negative)
  - producer_id must be non-empty
"""

import re
import uuid
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Nested model for declared quantities
# ---------------------------------------------------------------------------

class DeclaredQuantities(BaseModel):
    """
    Plastic quantities declared by the producer, in kilograms.
    All values must be zero or positive — negative quantities are invalid.
    """

    rigid_plastic: Annotated[
        float,
        Field(
            ge=0,
            description="Declared quantity of rigid plastic in kg",
            examples=[12000],
        ),
    ]
    flexible_plastic: Annotated[
        float,
        Field(
            ge=0,
            description="Declared quantity of flexible plastic in kg",
            examples=[8500],
        ),
    ]
    multilayer_plastic: Annotated[
        float,
        Field(
            ge=0,
            description="Declared quantity of multilayer plastic in kg",
            examples=[3200],
        ),
    ]


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class DeclarationRequest(BaseModel):
    """
    Incoming payload for POST /submit.
    All validation is deterministic — no LLM involved.
    """

    producer_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=50,
            description="Unique producer identifier",
            examples=["GREENPACK-001"],
        ),
    ]

    month: Annotated[
        str,
        Field(
            description="Declaration month in YYYY-MM format",
            examples=["2026-04"],
        ),
    ]

    declared_quantities_kg: DeclaredQuantities

    @field_validator("month")
    @classmethod
    def validate_month_format(cls, v: str) -> str:
        """Enforce YYYY-MM format with a valid month number (01-12)."""
        pattern = r"^\d{4}-(0[1-9]|1[0-2])$"
        if not re.match(pattern, v):
            raise ValueError(
                f"month must be in YYYY-MM format (e.g. '2026-04'), got: '{v}'"
            )
        return v

    @field_validator("producer_id")
    @classmethod
    def validate_producer_id(cls, v: str) -> str:
        """Strip whitespace and ensure producer_id is not blank."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("producer_id must not be empty or whitespace")
        return stripped

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "producer_id": "GREENPACK-001",
                    "month": "2026-04",
                    "declared_quantities_kg": {
                        "rigid_plastic": 12000,
                        "flexible_plastic": 8500,
                        "multilayer_plastic": 3200,
                    },
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------

class DeclarationResponse(BaseModel):
    """
    Response returned after a successful POST /submit.
    Includes the generated record_id and submission timestamp.
    """

    record_id: str = Field(description="Generated UUID for this declaration")
    producer_id: str
    month: str
    declared_quantities_kg: DeclaredQuantities
    submitted_at: datetime

    model_config = {"from_attributes": True}
