"""System information endpoint.

GET /api/v1/system/info — safe, non-sensitive application metadata

SECURITY: This endpoint must NEVER expose:
- Secret keys or tokens
- Database passwords or connection strings
- Environment variable contents
- Internal file paths
- Any other credentials
"""

from __future__ import annotations

import platform
import sys

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import get_settings

router = APIRouter(tags=["system"])
settings = get_settings()


class SystemInfoResponse(BaseModel):
    """Response schema for the system information endpoint.

    Contains only safe, non-sensitive application metadata.
    """

    app_name: str
    version: str
    environment: str
    python_version: str
    platform_system: str
    api_version: str


@router.get(
    "/system/info",
    response_model=SystemInfoResponse,
    summary="System information",
    description=(
        "Returns safe, non-sensitive application metadata. "
        "No credentials, secrets, or configuration values are included."
    ),
)
def get_system_info() -> SystemInfoResponse:
    """Return safe application metadata.

    This endpoint exposes only information that is safe to make public.
    It never returns environment variable contents, secrets, or credentials.
    """
    return SystemInfoResponse(
        app_name="Hybrid Recommendation Intelligence Platform",
        version=settings.app_version,
        environment=settings.app_env,
        python_version=sys.version.split(" ")[0],
        platform_system=platform.system(),
        api_version="v1",
    )
