"""Tests for health check endpoints.

Covers:
- GET /api/v1/health        (application liveness)
- GET /api/v1/health/db     (database connectivity)
"""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestApplicationHealth:
    """Tests for GET /api/v1/health."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health endpoint must return HTTP 200."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, client: TestClient) -> None:
        """Health endpoint must return status 'ok'."""
        data = client.get("/api/v1/health").json()
        assert data["status"] == "ok"

    def test_health_returns_version(self, client: TestClient) -> None:
        """Health endpoint must include a version string."""
        data = client.get("/api/v1/health").json()
        assert "version" in data
        assert isinstance(data["version"], str)
        assert len(data["version"]) > 0

    def test_health_returns_environment(self, client: TestClient) -> None:
        """Health endpoint must include the environment name."""
        data = client.get("/api/v1/health").json()
        assert "environment" in data
        assert isinstance(data["environment"], str)

    def test_health_response_schema(self, client: TestClient) -> None:
        """Health endpoint response must contain required fields only."""
        data = client.get("/api/v1/health").json()
        required_fields = {"status", "version", "environment"}
        assert required_fields.issubset(data.keys())

    def test_health_does_not_expose_secrets(self, client: TestClient) -> None:
        """Health endpoint must not expose secret_key or db_password."""
        data = client.get("/api/v1/health").json()
        response_text = str(data).lower()
        assert "secret" not in response_text
        assert "password" not in response_text
        assert "database_url" not in response_text


class TestDatabaseHealth:
    """Tests for GET /api/v1/health/db."""

    def test_db_health_returns_200(self, client: TestClient) -> None:
        """Database health endpoint must always return HTTP 200.

        Even when the database is unreachable, the endpoint returns 200
        with a status of 'error' — it does not crash with 500.
        """
        response = client.get("/api/v1/health/db")
        assert response.status_code == 200

    def test_db_health_returns_status_field(self, client: TestClient) -> None:
        """Database health response must include a 'status' field."""
        data = client.get("/api/v1/health/db").json()
        assert "status" in data

    def test_db_health_status_is_valid_value(self, client: TestClient) -> None:
        """Database health status must be 'connected' or 'error'."""
        data = client.get("/api/v1/health/db").json()
        assert data["status"] in {"connected", "error"}

    def test_db_health_does_not_expose_credentials(self, client: TestClient) -> None:
        """Database health endpoint must not expose connection credentials."""
        data = client.get("/api/v1/health/db").json()
        response_text = str(data).lower()
        assert "password" not in response_text
        assert "postgresql://" not in response_text
        assert "secret" not in response_text
