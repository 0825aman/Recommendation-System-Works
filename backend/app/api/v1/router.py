"""API v1 router — aggregates all version 1 endpoint routers."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.system import router as system_router

router = APIRouter(prefix="/api/v1")

router.include_router(health_router)
router.include_router(system_router)
