"""
app/database/session.py
-----------------------
SQLAlchemy engine, session factory, and FastAPI dependency injection.

Architecture note: The engine and SessionLocal are created once at module
import time. The get_db() dependency yields a session per request and
guarantees cleanup via the finally block — preventing connection leaks
even when exceptions occur mid-request.
"""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.database.models import Base
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------
# connect_args={"check_same_thread": False} is required for SQLite when used
# with FastAPI's async request handling — SQLite connections are not thread-safe
# by default, but SQLAlchemy manages thread-local sessions correctly.
engine = create_engine(
    settings.DB_URL,
    connect_args={"check_same_thread": False},
    echo=False,  # Set to True for SQL query logging during development
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def init_db() -> None:
    """
    Create all database tables defined in the ORM models.
    Called once during application startup lifespan.
    Safe to call multiple times — CREATE TABLE IF NOT EXISTS semantics.
    """
    logger.info("Initializing database at: %s", settings.DB_URL)
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified successfully")


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a SQLAlchemy session per request.

    Usage in routes:
        @router.get("/endpoint")
        def my_endpoint(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
