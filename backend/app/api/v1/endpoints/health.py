"""Health check endpoints.

GET /api/v1/health     — application liveness check
GET /api/v1/health/db  — database connectivity check

These endpoints intentionally expose NO secrets, credentials,
or internal configuration values.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import get_settings
from app.db.session import check_database_connection

router = APIRouter(tags=["health"])
settings = get_settings()


class HealthResponse(BaseModel):
    """Response schema for the application health endpoint."""

    status: str
    version: str
    environment: str


class DatabaseHealthResponse(BaseModel):
    """Response schema for the database health endpoint."""

    status: str
    detail: str | None = None


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Application health",
    description="Returns the application liveness status. Always responds if the server is up.",
)
def get_health() -> HealthResponse:
    """Return application health status.

    This endpoint is always available as long as the server process is running.
    It does NOT test database connectivity.
    """
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        environment=settings.app_env,
    )


@router.get(
    "/health/db",
    response_model=DatabaseHealthResponse,
    summary="Database connectivity check",
    description=(
        "Tests the actual database connection. "
        "Returns 'connected' when PostgreSQL is reachable, 'error' otherwise. "
        "Never exposes credentials or connection strings."
    ),
)
def get_database_health() -> DatabaseHealthResponse:
    """Return database connectivity status.

    Performs a live SELECT 1 against the configured database.
    Returns a safe status object — no credentials are included.
    """
    result = check_database_connection()
    return DatabaseHealthResponse(**result)
