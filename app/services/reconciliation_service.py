"""
app/services/reconciliation_service.py
---------------------------------------
ERP reconciliation business logic — compare declared vs procured quantities.

Critical architecture note: ALL numeric computations (difference_percent,
total quantities, status flags) are performed in pure Python here.
The LLM is invoked ONLY after all computations are complete, receiving
pre-computed values to narrate. This separation is intentional:

  - Deterministic logic: never delegate to LLM
  - LLM role: narrative summarization only
  - This prevents calculation hallucinations and ensures auditability

The 5% mismatch threshold is defined in constants.MISMATCH_THRESHOLD_PERCENT
and applied uniformly across all categories.
"""

from sqlalchemy.orm import Session

from app.core.constants import (
    CATEGORY_DISPLAY_NAMES,
    MISMATCH_THRESHOLD_PERCENT,
    PLASTIC_CATEGORIES,
    STATUS_MISMATCH,
    STATUS_OK,
)
from app.database import repository
from app.schemas.summary import ReconciliationItem, ReconciliationResponse
from app.services.llm_service import ask_ollama, load_prompt_template
from app.utils.erp_loader import load_erp_data
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _compute_difference_percent(declared: float, procured: float) -> float:
    """
    Compute the percentage difference between declared and procured quantities.

    Formula: |declared - procured| / procured * 100

    Returns 0.0 if procured is zero to avoid division by zero.
    The LLM must never be used for this calculation.
    """
    if procured == 0.0:
        return 0.0 if declared == 0.0 else 100.0
    return abs(declared - procured) / procured * 100.0


def _format_reconciliation_for_prompt(items: list[ReconciliationItem]) -> str:
    """Format reconciliation items as a readable string for the LLM prompt."""
    lines = []
    for item in items:
        lines.append(
            f"  {item.display_name}: "
            f"Declared={item.declared_kg:,.0f} kg, "
            f"Procured={item.procured_kg:,.0f} kg, "
            f"Difference={item.difference_percent:.1f}%, "
            f"Status={item.status}"
        )
    return "\n".join(lines)


async def get_reconciliation_summary(
    producer_id: str,
    month: str,
    db: Session,
) -> ReconciliationResponse:
    """
    Generate a full ERP reconciliation report for a producer/month pair.

    Steps:
      1. Retrieve declaration from SQLite (via repository)
      2. Load ERP procurement data from CSV (via erp_loader)
      3. Compute per-category differences (pure Python, NO LLM)
      4. Determine status flags (OK / MISMATCH)
      5. Call LLM ONLY for narrative summary and recommendations

    Args:
        producer_id: Producer identifier.
        month: Month string in YYYY-MM format.
        db: Active database session.

    Returns:
        ReconciliationResponse with computed data and AI narrative.

    Raises:
        ValueError: If no declaration is found for the given producer/month.
    """
    # Step 1: Retrieve declaration from DB
    declaration = repository.get_declaration(db, producer_id, month)
    if declaration is None:
        raise ValueError(
            f"No declaration found for producer '{producer_id}' in month '{month}'. "
            "Please submit a declaration first via POST /submit."
        )

    logger.info(
        "Reconciling declaration %s for producer=%s month=%s",
        declaration.record_id,
        producer_id,
        month,
    )

    # Step 2: Load ERP procurement data
    erp_data = load_erp_data(producer_id, month)

    # Step 3 & 4: Compute differences and status (NO LLM involved)
    declared_values = {
        "rigid_plastic": declaration.rigid_plastic_kg,
        "flexible_plastic": declaration.flexible_plastic_kg,
        "multilayer_plastic": declaration.multilayer_plastic_kg,
    }

    items: list[ReconciliationItem] = []
    has_mismatch = False

    for category in PLASTIC_CATEGORIES:
        declared_kg = declared_values[category]
        procured_kg = erp_data[category]
        diff_kg = abs(declared_kg - procured_kg)
        diff_pct = _compute_difference_percent(declared_kg, procured_kg)

        # Deterministic threshold check — never LLM
        status = STATUS_MISMATCH if diff_pct > MISMATCH_THRESHOLD_PERCENT else STATUS_OK
        if status == STATUS_MISMATCH:
            has_mismatch = True

        items.append(
            ReconciliationItem(
                category=category,
                display_name=CATEGORY_DISPLAY_NAMES[category],
                declared_kg=declared_kg,
                procured_kg=procured_kg,
                difference_kg=diff_kg,
                difference_percent=round(diff_pct, 2),
                status=status,
            )
        )
        logger.debug(
            "Category=%s declared=%.0f procured=%.0f diff=%.2f%% status=%s",
            category, declared_kg, procured_kg, diff_pct, status,
        )

    total_declared = sum(i.declared_kg for i in items)
    total_procured = sum(i.procured_kg for i in items)
    overall_status = STATUS_MISMATCH if has_mismatch else STATUS_OK

    logger.info(
        "Reconciliation computed: overall_status=%s total_declared=%.0f total_procured=%.0f",
        overall_status, total_declared, total_procured,
    )

    # Step 5: Call LLM ONLY for narrative (receives pre-computed values)
    mismatched = [i for i in items if i.status == STATUS_MISMATCH]
    mismatched_text = (
        "\n".join(
            f"  - {i.display_name}: {i.difference_percent:.1f}% difference"
            for i in mismatched
        )
        if mismatched
        else "  None — all categories within acceptable range."
    )

    try:
        prompt_template = load_prompt_template("summary_prompt.txt")
        prompt = prompt_template.format(
            producer_id=producer_id,
            month=month,
            overall_status=overall_status,
            reconciliation_data=_format_reconciliation_for_prompt(items),
            mismatched_categories=mismatched_text,
            total_declared_kg=f"{total_declared:,.0f}",
            total_procured_kg=f"{total_procured:,.0f}",
        )

        llm_response = await ask_ollama(prompt)

        # Split LLM response into summary + recommendations
        # Simple heuristic: first 3 sentences = summary, rest = recommendations
        sentences = [s.strip() for s in llm_response.replace("\n", " ").split(".") if s.strip()]
        summary_sentences = sentences[:3]
        recommendation_sentences = sentences[3:]

        ai_summary = ". ".join(summary_sentences) + ("." if summary_sentences else "")
        recommendations = ". ".join(recommendation_sentences) + ("." if recommendation_sentences else "")

        if not recommendations:
            recommendations = llm_response  # Use full response if splitting fails

    except Exception as exc:
        logger.error("Failed to generate AI summary: %s", exc)
        ai_summary = "AI summary unavailable — please ensure Ollama is running."
        recommendations = "Please review the reconciliation data manually."

    return ReconciliationResponse(
        producer_id=producer_id,
        month=month,
        record_id=declaration.record_id,
        items=items,
        total_declared_kg=total_declared,
        total_procured_kg=total_procured,
        overall_status=overall_status,
        ai_summary=ai_summary,
        recommendations=recommendations,
    )
