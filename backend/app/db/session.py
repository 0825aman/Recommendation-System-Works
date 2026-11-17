"""SQLAlchemy engine, session factory, and FastAPI dependency.

The database URL is read from application settings — never hardcoded.
Use get_db() as a FastAPI dependency to obtain a session per request.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

if TYPE_CHECKING:
    from collections.abc import Generator

from app.core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

engine = create_engine(
    settings.database_url,
    # Pool configuration appropriate for a development monolith
    pool_pre_ping=True,  # Verify connections before use
    pool_size=5,
    max_overflow=10,
    echo=False,  # Set to True temporarily for SQL debugging
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db() -> Generator[Session]:
    """FastAPI dependency that provides a database session per request.

    Yields a SQLAlchemy Session and ensures it is closed after the request,
    even if an exception occurs.

    Usage in a route:
        @router.get("/items")
        def list_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_database_connection() -> dict[str, str]:
    """Test the database connection by executing a lightweight query.

    Returns:
        dict with "status" key: "connected" or "error", and optional "detail".
        Never includes connection credentials or internal stack traces.
    """
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"status": "connected"}
    except Exception as exc:
        logger.warning("Database connectivity check failed: %s", type(exc).__name__)
        return {"status": "error", "detail": "Database is not reachable."}
