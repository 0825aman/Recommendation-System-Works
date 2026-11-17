"""Recommendation endpoints — Phase 4.

GET /api/v1/recommendations/results   — evaluation summary from experiment files
GET /api/v1/recommendations/algorithms — list of evaluated algorithms

These endpoints serve pre-computed evaluation results from the experiments/results/
directory. They do NOT perform live inference at request time (Phase 5).

Security: no secrets, no credentials, no internal paths are exposed.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

_PROJECT_ROOT = Path(__file__).resolve().parents[6]
_RESULTS_DIR = _PROJECT_ROOT / "experiments" / "results"


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class AlgorithmResult(BaseModel):
    """Evaluation metrics for a single algorithm."""

    algorithm: str
    phase: str
    ndcg_at_10: float | None = None
    precision_at_10: float | None = None
    recall_at_10: float | None = None
    map_at_10: float | None = None
    catalog_coverage: float | None = None
    n_eval_users: int | None = None


class RecommendationResultsResponse(BaseModel):
    """Aggregated evaluation results across all recorded experiments."""

    total_experiments: int
    algorithms: list[AlgorithmResult]
    primary_metric: str = "ndcg_at_10"


class AlgorithmsResponse(BaseModel):
    """List of algorithms with recorded experiment results."""

    algorithms: list[str]
    results_available: int


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _load_all_results() -> list[dict]:  # type: ignore[type-arg]
    """Load all experiment JSON files from experiments/results/."""
    if not _RESULTS_DIR.exists():
        return []
    results = []
    for path in sorted(_RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            results.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return results


def _extract_metric(result: dict, key: str) -> float | None:  # type: ignore[type-arg]
    """Safely extract a metric value from a result dict."""
    try:
        agg = result.get("results", {}).get("aggregate", {})
        val = agg.get(key)
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/results",
    response_model=RecommendationResultsResponse,
    summary="Recommendation evaluation results",
    description=(
        "Returns a summary of offline evaluation metrics for all "
        "recommendation algorithms with recorded experiment results. "
        "Reads from pre-computed experiment JSON files."
    ),
)
def get_recommendation_results() -> RecommendationResultsResponse:
    """Return evaluation results for all recorded experiments.

    Results are read from experiments/results/*.json and summarised.
    Returns an empty list if no experiment files are found.
    """
    all_results = _load_all_results()

    algorithm_results = []
    for r in all_results:
        alg_result = AlgorithmResult(
            algorithm=r.get("algorithm", "unknown"),
            phase=str(r.get("phase", "?")),
            ndcg_at_10=_extract_metric(r, "ndcg_at_10_mean"),
            precision_at_10=_extract_metric(r, "precision_at_10_mean"),
            recall_at_10=_extract_metric(r, "recall_at_10_mean"),
            map_at_10=_extract_metric(r, "map_at_10_mean"),
            catalog_coverage=r.get("results", {}).get("catalog_coverage"),
            n_eval_users=r.get("results", {}).get("n_eval_users"),
        )
        algorithm_results.append(alg_result)

    # Sort by NDCG@10 descending (None last)
    algorithm_results.sort(
        key=lambda x: x.ndcg_at_10 if x.ndcg_at_10 is not None else -1.0,
        reverse=True,
    )

    return RecommendationResultsResponse(
        total_experiments=len(all_results),
        algorithms=algorithm_results,
    )


@router.get(
    "/algorithms",
    response_model=AlgorithmsResponse,
    summary="List evaluated recommendation algorithms",
    description="Returns the names of all algorithms with recorded experiment results.",
)
def get_algorithms() -> AlgorithmsResponse:
    """Return names of algorithms with recorded experiment results."""
    all_results = _load_all_results()
    names = sorted({r.get("algorithm", "unknown") for r in all_results})
    return AlgorithmsResponse(
        algorithms=names,
        results_available=len(all_results),
    )


@router.get(
    "/results/{algorithm_name}",
    response_model=AlgorithmResult,
    summary="Get results for a specific algorithm",
    description=(
        "Returns evaluation metrics for the most recent experiment "
        "recorded for the given algorithm name."
    ),
)
def get_algorithm_result(algorithm_name: str) -> AlgorithmResult:
    """Return the most recent result for a given algorithm name.

    Args:
        algorithm_name: Exact algorithm name (URL-encoded if it contains special chars).

    Raises:
        404 if no results are found for the given algorithm.
    """
    all_results = _load_all_results()
    matching = [r for r in all_results if r.get("algorithm") == algorithm_name]

    if not matching:
        raise HTTPException(
            status_code=404,
            detail=f"No experiment results found for algorithm '{algorithm_name}'.",
        )

    # Return the most recently recorded result (last in sorted file order)
    r = matching[-1]
    return AlgorithmResult(
        algorithm=r.get("algorithm", "unknown"),
        phase=str(r.get("phase", "?")),
        ndcg_at_10=_extract_metric(r, "ndcg_at_10_mean"),
        precision_at_10=_extract_metric(r, "precision_at_10_mean"),
        recall_at_10=_extract_metric(r, "recall_at_10_mean"),
        map_at_10=_extract_metric(r, "map_at_10_mean"),
        catalog_coverage=r.get("results", {}).get("catalog_coverage"),
        n_eval_users=r.get("results", {}).get("n_eval_users"),
    )
