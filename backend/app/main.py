"""FastAPI application factory.

This module creates and configures the FastAPI application instance.
Import the `app` object for use with uvicorn:

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import router as v1_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

settings = get_settings()

# Configure logging before anything else
configure_logging(settings.backend_log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Manage application startup and shutdown lifecycle.

    Code before yield runs at startup; code after yield runs at shutdown.
    """
    logger.info(
        "Starting Hybrid Recommendation Intelligence Platform v%s [%s]",
        settings.app_version,
        settings.app_env,
    )
    yield
    logger.info("Application shutting down.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Hybrid Recommendation Intelligence Platform",
        description=(
            "Backend API for a production-oriented hybrid recommendation system "
            "combining content-based filtering, collaborative filtering, "
            "candidate generation, ML ranking, and feedback loops."
        ),
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS — allow the Vite dev server to call the backend in development
    # In production this must be tightened to the actual frontend domain
    if settings.is_development:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
            allow_credentials=False,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["Content-Type", "Accept"],
        )

    # Register centralized exception handlers
    register_exception_handlers(app)

    # Register API routers
    app.include_router(v1_router)

    return app


app = create_app()
