"""
app/utils/erp_loader.py
------------------------
Utility for reading and filtering ERP procurement data from CSV.

Architecture note: File I/O for the ERP feed is isolated here rather than
inline in the service layer. This makes it trivial to swap the CSV source
for a live ERP API or a database table without touching business logic.
"""

from pathlib import Path

import pandas as pd

from app.core.config import settings
from app.core.constants import PLASTIC_CATEGORIES
from app.utils.logger import get_logger

logger = get_logger(__name__)


def load_erp_data(producer_id: str, month: str) -> dict[str, float]:
    """
    Load ERP procurement quantities for a given producer and month.

    Args:
        producer_id: The producer identifier (e.g. 'GREENPACK-001').
        month: Month string in 'YYYY-MM' format.

    Returns:
        A dict mapping category name → procured_kg for the requested
        producer/month. Returns 0.0 for any category not found in the CSV.

    Raises:
        FileNotFoundError: If the ERP CSV file does not exist.
        ValueError: If the CSV is malformed or missing required columns.
    """
    csv_path = Path(settings.ERP_CSV_PATH)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"ERP CSV not found at '{csv_path}'. "
            "Please ensure mock_data/erp_feed.csv exists."
        )

    logger.debug("Loading ERP CSV from %s", csv_path)

    try:
        df = pd.read_csv(csv_path, dtype=str)
    except Exception as exc:
        raise ValueError(f"Failed to parse ERP CSV: {exc}") from exc

    required_columns = {"producer_id", "month", "category", "procured_kg"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"ERP CSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Filter rows for the requested producer and month
    mask = (df["producer_id"] == producer_id) & (df["month"] == month)
    filtered = df[mask].copy()

    if filtered.empty:
        logger.warning(
            "No ERP data found for producer='%s', month='%s'",
            producer_id,
            month,
        )
        return {cat: 0.0 for cat in PLASTIC_CATEGORIES}

    # Convert procured_kg to float safely
    filtered["procured_kg"] = pd.to_numeric(
        filtered["procured_kg"], errors="coerce"
    ).fillna(0.0)

    # Build category → quantity mapping
    result: dict[str, float] = {cat: 0.0 for cat in PLASTIC_CATEGORIES}
    for _, row in filtered.iterrows():
        category = str(row["category"]).strip()
        if category in result:
            result[category] = float(row["procured_kg"])
        else:
            logger.warning("Unknown ERP category '%s' — skipping row", category)

    logger.info(
        "ERP data loaded for producer='%s', month='%s': %s",
        producer_id,
        month,
        result,
    )
    return result
