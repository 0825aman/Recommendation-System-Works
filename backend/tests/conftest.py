"""Pytest configuration and shared fixtures.

Provides:
- TestClient for making HTTP requests to the FastAPI app in tests
- Database session override for isolated test database access
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Return a TestClient wrapping the FastAPI application.

    Scoped to the session so the app is only initialized once per test run.
    """
    return TestClient(app)
