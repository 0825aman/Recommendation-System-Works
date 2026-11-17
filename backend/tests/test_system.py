"""Tests for the system information endpoint.

Covers:
- GET /api/v1/system/info
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


class TestSystemInfo:
    """Tests for GET /api/v1/system/info."""

    def test_system_info_returns_200(self, client: TestClient) -> None:
        """System info endpoint must return HTTP 200."""
        response = client.get("/api/v1/system/info")
        assert response.status_code == 200

    def test_system_info_returns_app_name(self, client: TestClient) -> None:
        """System info must include the application name."""
        data = client.get("/api/v1/system/info").json()
        assert "app_name" in data
        assert "Hybrid Recommendation" in data["app_name"]

    def test_system_info_returns_version(self, client: TestClient) -> None:
        """System info must include a version string."""
        data = client.get("/api/v1/system/info").json()
        assert "version" in data
        assert isinstance(data["version"], str)

    def test_system_info_returns_python_version(self, client: TestClient) -> None:
        """System info must include the Python version."""
        data = client.get("/api/v1/system/info").json()
        assert "python_version" in data
        assert data["python_version"].startswith("3.")

    def test_system_info_returns_api_version(self, client: TestClient) -> None:
        """System info must include the API version."""
        data = client.get("/api/v1/system/info").json()
        assert "api_version" in data
        assert data["api_version"] == "v1"

    def test_system_info_does_not_expose_secrets(self, client: TestClient) -> None:
        """System info endpoint must not expose any sensitive values."""
        data = client.get("/api/v1/system/info").json()
        response_text = str(data).lower()
        # Ensure no secret-like values appear in the response
        assert "secret" not in response_text
        assert "password" not in response_text
        assert "postgresql://" not in response_text
        assert "database_url" not in response_text

    def test_system_info_does_not_expose_internal_paths(
        self, client: TestClient
    ) -> None:
        """System info must not expose internal file system paths."""
        data = client.get("/api/v1/system/info").json()
        response_text = str(data)
        # Should not contain Windows-style absolute paths
        assert "C:\\" not in response_text
        assert "/home/" not in response_text

    def test_system_info_required_schema(self, client: TestClient) -> None:
        """System info response must contain all required fields."""
        data = client.get("/api/v1/system/info").json()
        required_fields = {
            "app_name",
            "version",
            "environment",
            "python_version",
            "platform_system",
            "api_version",
        }
        assert required_fields.issubset(data.keys())
