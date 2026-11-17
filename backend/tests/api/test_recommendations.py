"""Tests for Phase 4 recommendation API endpoints.

Tests GET /api/v1/recommendations/results
     GET /api/v1/recommendations/algorithms
     GET /api/v1/recommendations/results/{algorithm_name}
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


class TestRecommendationResults:
    def test_results_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/recommendations/results")
        assert response.status_code == 200

    def test_results_schema_has_required_fields(self, client: TestClient) -> None:
        response = client.get("/api/v1/recommendations/results")
        body = response.json()
        assert "total_experiments" in body
        assert "algorithms" in body
        assert "primary_metric" in body
        assert isinstance(body["algorithms"], list)
        assert isinstance(body["total_experiments"], int)

    def test_results_total_experiments_is_non_negative(self, client: TestClient) -> None:
        response = client.get("/api/v1/recommendations/results")
        body = response.json()
        assert body["total_experiments"] >= 0

    def test_results_primary_metric_is_ndcg(self, client: TestClient) -> None:
        response = client.get("/api/v1/recommendations/results")
        body = response.json()
        assert body["primary_metric"] == "ndcg_at_10"

    def test_results_algorithms_have_correct_shape(self, client: TestClient) -> None:
        response = client.get("/api/v1/recommendations/results")
        body = response.json()
        for alg in body["algorithms"]:
            assert "algorithm" in alg
            assert "phase" in alg
            # Optional metric fields — either float or null
            for field in ["ndcg_at_10", "precision_at_10", "recall_at_10"]:
                assert field in alg

    def test_results_does_not_expose_secrets(self, client: TestClient) -> None:
        response = client.get("/api/v1/recommendations/results")
        text = response.text.lower()
        for forbidden in ["password", "secret_key", "db_password", "token"]:
            assert forbidden not in text


class TestRecommendationAlgorithms:
    def test_algorithms_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/recommendations/algorithms")
        assert response.status_code == 200

    def test_algorithms_schema(self, client: TestClient) -> None:
        response = client.get("/api/v1/recommendations/algorithms")
        body = response.json()
        assert "algorithms" in body
        assert "results_available" in body
        assert isinstance(body["algorithms"], list)
        assert isinstance(body["results_available"], int)

    def test_algorithms_results_available_matches_results_endpoint(
        self, client: TestClient
    ) -> None:
        alg_resp = client.get("/api/v1/recommendations/algorithms").json()
        res_resp = client.get("/api/v1/recommendations/results").json()
        assert alg_resp["results_available"] == res_resp["total_experiments"]


class TestRecommendationAlgorithmDetail:
    def test_unknown_algorithm_returns_404(self, client: TestClient) -> None:
        response = client.get(
            "/api/v1/recommendations/results/NonExistentAlgorithmXYZ"
        )
        assert response.status_code == 404

    def test_404_response_does_not_expose_secrets(self, client: TestClient) -> None:
        response = client.get(
            "/api/v1/recommendations/results/NonExistentAlgorithmXYZ"
        )
        text = response.text.lower()
        for forbidden in ["password", "secret_key", "db_password"]:
            assert forbidden not in text
