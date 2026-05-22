"""
app/api/pages.py
----------------
Jinja2 template page routes for the Bootstrap 5 frontend.

These routes serve HTML pages rendered server-side using FastAPI's Jinja2
TemplateResponse. API data is fetched client-side via JavaScript fetch() calls
to keep page routes simple and the frontend interactive.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.database import repository
from app.database.session import get_db
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Pages"])
templates = Jinja2Templates(directory="app/templates")


@router.get("/", include_in_schema=False)
async def home_page(request: Request, db: Session = Depends(get_db)):
    """
    GET / — Home dashboard showing system stats and recent declarations.
    """
    total_count = repository.count_declarations(db)
    recent = repository.get_all_declarations(db, limit=5)

    logger.info("Dashboard page loaded — %d total declarations", total_count)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "total_declarations": total_count,
            "recent_declarations": recent,
            "page_title": "Dashboard",
        },
    )


@router.get("/declare", include_in_schema=False)
async def declare_page(request: Request):
    """
    GET /declare — EPR declaration submission form.
    """
    logger.info("Declare page accessed")
    return templates.TemplateResponse(
        "declare.html",
        {
            "request": request,
            "page_title": "Submit Declaration",
        },
    )


@router.get("/summary", include_in_schema=False)
async def summary_page(
    request: Request,
    producer_id: Optional[str] = Query(default=None, description="Pre-fill producer ID"),
    month: Optional[str] = Query(default=None, description="Pre-fill month (YYYY-MM)"),
):
    """
    GET /summary — ERP reconciliation dashboard.
    Accepts optional ?producer_id=...&month=... query params for pre-population.
    All reconciliation data is fetched client-side via GET /summary/{producer_id}/{month}.
    """
    logger.info(
        "Summary page accessed — prefill: producer_id=%s month=%s",
        producer_id or "(none)",
        month or "(none)",
    )
    return templates.TemplateResponse(
        "summary.html",
        {
            "request": request,
            "page_title": "Reconciliation Summary",
            "prefill_producer_id": producer_id or "",
            "prefill_month": month or "",
        },
    )


@router.get("/ask", include_in_schema=False)
async def ask_page(request: Request):
    """
    GET /ask — RAG-powered EPR compliance chatbot page.
    """
    logger.info("Ask page accessed")
    return templates.TemplateResponse(
        "ask.html",
        {
            "request": request,
            "page_title": "EPR Compliance Assistant",
        },
    )
