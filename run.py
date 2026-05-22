"""
run.py
------
Application entrypoint. Start the FastAPI server with uvicorn.

Usage:
    python run.py

Or with auto-reload for development:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import uvicorn

from app.core.config import settings

if __name__ == "__main__":
    log_level = settings.LOG_LEVEL.lower()

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,          # Set True for development hot-reload
        log_level=log_level,
        access_log=False,      # We use custom logging middleware instead
    )
