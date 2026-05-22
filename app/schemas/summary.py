"""
app/schemas/summary.py
----------------------
Pydantic schemas for ERP reconciliation summary response.

The reconciliation response contains:
  - Per-category comparison items (declared vs procured, diff %, status)
  - An AI-generated narrative summary from Ollama
  - Structured recommendations
"""

from pydantic import BaseModel, Field


class ReconciliationItem(BaseModel):
    """
    A single category's reconciliation result.

    Architecture note: All numeric fields (declared, procured, difference_percent)
    are computed in pure Python — never delegated to the LLM. The LLM only
    receives these pre-computed values to narrate in human language.
    """

    category: str = Field(description="Plastic category name", examples=["rigid_plastic"])
    display_name: str = Field(description="Human-readable category name", examples=["Rigid Plastic"])
    declared_kg: float = Field(description="Quantity declared by producer (kg)")
    procured_kg: float = Field(description="Quantity from ERP procurement data (kg)")
    difference_kg: float = Field(description="Absolute difference (kg)")
    difference_percent: float = Field(description="Percentage difference relative to procured")
    status: str = Field(
        description="'OK' if difference <= 5%, else 'MISMATCH'",
        examples=["OK", "MISMATCH"],
    )


class ReconciliationResponse(BaseModel):
    """
    Full reconciliation response for GET /summary/{producer_id}/{month}.
    """

    producer_id: str
    month: str
    record_id: str = Field(description="Declaration record UUID")
    items: list[ReconciliationItem] = Field(
        description="Per-category reconciliation results"
    )
    total_declared_kg: float = Field(description="Sum of all declared quantities")
    total_procured_kg: float = Field(description="Sum of all ERP procured quantities")
    overall_status: str = Field(
        description="'OK' if all categories pass, else 'MISMATCH'",
        examples=["MISMATCH"],
    )
    ai_summary: str = Field(
        description="LLM-generated 3-5 sentence compliance narrative"
    )
    recommendations: str = Field(
        description="LLM-generated actionable recommendations"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "producer_id": "GREENPACK-001",
                    "month": "2026-04",
                    "record_id": "a1b2c3d4-...",
                    "items": [
                        {
                            "category": "rigid_plastic",
                            "display_name": "Rigid Plastic",
                            "declared_kg": 12000,
                            "procured_kg": 11800,
                            "difference_kg": 200,
                            "difference_percent": 1.69,
                            "status": "OK",
                        }
                    ],
                    "total_declared_kg": 23700,
                    "total_procured_kg": 24050,
                    "overall_status": "MISMATCH",
                    "ai_summary": "GreenPack Industries has demonstrated...",
                    "recommendations": "We recommend investigating...",
                }
            ]
        }
    }
