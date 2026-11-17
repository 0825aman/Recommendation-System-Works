"""Offline evaluation metrics for recommendation systems.

All metrics operate on pure Python lists/sets. No numpy or external
dependencies required — deliberate choice to avoid version conflicts
and keep the evaluation layer lightweight.

Reference metrics:
    - Precision@K  : fraction of recommended items that are relevant
    - Recall@K     : fraction of relevant items that were recommended
    - NDCG@K       : normalized discounted cumulative gain (ranking quality)
    - MAP@K        : mean average precision at K
    - Hit Rate@K   : binary — 1 if any top-K recommendation is relevant
    - Catalog Coverage : breadth of recommendations across all users
    - Intra-List Diversity : category variety within a single recommendation list
    - Novelty       : how non-obvious the recommendations are (self-information)

Usage:
    from ml.evaluation.metrics import precision_at_k, ndcg_at_k, hit_rate_at_k
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Per-user ranking metrics
# ---------------------------------------------------------------------------


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Compute Precision@K.

    Args:
        recommended: Ordered list of recommended item IDs (best first).
        relevant:    Set of ground-truth relevant item IDs for this user.
        k:           Cutoff rank.

    Returns:
        Fraction of top-K recommendations that are relevant. Range [0, 1].
        Returns 0.0 if recommended is empty or k == 0.
    """
    if k <= 0 or not recommended:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Compute Recall@K.

    Args:
        recommended: Ordered list of recommended item IDs (best first).
        relevant:    Set of ground-truth relevant item IDs for this user.
        k:           Cutoff rank.

    Returns:
        Fraction of relevant items retrieved in top-K. Range [0, 1].
        Returns 0.0 if relevant is empty.
    """
    if not relevant or k <= 0 or not recommended:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Compute NDCG@K (Normalized Discounted Cumulative Gain).

    Uses binary relevance (item is relevant or not).
    Ideal DCG (IDCG) is computed assuming all relevant items are ranked first.

    Args:
        recommended: Ordered list of recommended item IDs (best first).
        relevant:    Set of ground-truth relevant item IDs for this user.
        k:           Cutoff rank.

    Returns:
        NDCG@K score. Range [0, 1]. Returns 0.0 if relevant is empty.
    """
    if not relevant or k <= 0 or not recommended:
        return 0.0

    top_k = recommended[:k]

    # DCG: sum of 1/log2(rank+1) for relevant items in recommended
    dcg = sum(
        1.0 / math.log2(rank + 2)  # rank is 0-indexed → log2(rank+2)
        for rank, item in enumerate(top_k)
        if item in relevant
    )

    # IDCG: best possible DCG — all relevant items ranked first
    n_ideal = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(n_ideal))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def average_precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Compute Average Precision@K for a single user.

    Args:
        recommended: Ordered list of recommended item IDs (best first).
        relevant:    Set of ground-truth relevant item IDs for this user.
        k:           Cutoff rank.

    Returns:
        AP@K score. Range [0, 1]. Returns 0.0 if relevant is empty.
    """
    if not relevant or k <= 0 or not recommended:
        return 0.0

    top_k = recommended[:k]
    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(top_k, 1):
        if item in relevant:
            hits += 1
            precision_sum += hits / rank

    if hits == 0:
        return 0.0
    return precision_sum / min(len(relevant), k)


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Compute Hit Rate@K (binary relevance indicator).

    Returns 1.0 if at least one of the top-K recommendations is relevant,
    otherwise 0.0. Also known as Recall@1-binary or any-hit rate.

    Args:
        recommended: Ordered list of recommended item IDs (best first).
        relevant:    Set of ground-truth relevant item IDs for this user.
        k:           Cutoff rank.

    Returns:
        1.0 if any top-K item is relevant, else 0.0.
        Returns 0.0 if recommended is empty, relevant is empty, or k == 0.
    """
    if k <= 0 or not recommended or not relevant:
        return 0.0
    return float(any(item in relevant for item in recommended[:k]))


# ---------------------------------------------------------------------------
# Aggregate / list-level metrics
# ---------------------------------------------------------------------------


def catalog_coverage(
    all_recommendations: list[list[int]],
    catalog_size: int,
) -> float:
    """Fraction of catalog items that appear in at least one recommendation list.

    Args:
        all_recommendations: List of per-user recommendation lists.
        catalog_size:        Total number of items in the catalog.

    Returns:
        Coverage fraction. Range [0, 1]. Returns 0.0 if catalog_size == 0.
    """
    if catalog_size <= 0:
        return 0.0
    recommended_items: set[int] = set()
    for recs in all_recommendations:
        recommended_items.update(recs)
    return len(recommended_items) / catalog_size


def intra_list_diversity(
    recommendations: list[int],
    item_to_category: dict[int, str],
) -> float:
    """Average pairwise category distance within a recommendation list.

    Distance between two items = 0 if same category, 1 if different category.
    This is equivalent to 1 - (category concentration) for binary categories.

    Args:
        recommendations:   Ordered list of recommended item IDs.
        item_to_category:  Mapping from item ID to category string.

    Returns:
        Diversity score. Range [0, 1].
        Returns 0.0 if fewer than 2 items or all items have no category.
    """
    n = len(recommendations)
    if n < 2:
        return 0.0

    cats = [item_to_category.get(item) for item in recommendations]
    # Only consider pairs where both have a known category
    pairs = 0
    different = 0
    for i in range(n):
        for j in range(i + 1, n):
            if cats[i] is not None and cats[j] is not None:
                pairs += 1
                if cats[i] != cats[j]:
                    different += 1

    if pairs == 0:
        return 0.0
    return different / pairs


def novelty(
    recommendations: list[int],
    item_popularity: dict[int, float],
    n_users: int,
) -> float:
    """Average self-information (novelty) of recommended items.

    Novelty = -log2(p(item)) where p(item) = popularity / n_users.
    Less popular items have higher novelty.

    Args:
        recommendations:  Ordered list of recommended item IDs.
        item_popularity:  Mapping from item ID to interaction count.
        n_users:          Total number of users in the system.

    Returns:
        Mean novelty across recommendations. Higher = less popular = more novel.
        Returns 0.0 if recommendations is empty or n_users == 0.
    """
    if not recommendations or n_users <= 0:
        return 0.0

    scores = []
    for item in recommendations:
        pop = item_popularity.get(item, 0)
        if pop <= 0:
            # Item with no interactions: maximally novel
            scores.append(math.log2(n_users + 1))
        else:
            p = pop / n_users
            scores.append(-math.log2(p))

    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Aggregate over all users
# ---------------------------------------------------------------------------


def aggregate_metrics(
    per_user: list[dict[str, float]],  # type: ignore[type-arg]
) -> dict[str, float]:  # type: ignore[type-arg]
    """Compute mean, median, std, p10, p90 for each metric across users.

    Args:
        per_user: List of per-user metric dicts (each has the same keys).

    Returns:
        Dict with aggregated statistics for each metric key.
    """
    if not per_user:
        return {}

    keys = list(per_user[0].keys())
    result: dict[str, float] = {}  # type: ignore[type-arg]

    for key in keys:
        values = sorted(v[key] for v in per_user)
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std = math.sqrt(variance)
        median = values[n // 2] if n % 2 == 1 else (values[n // 2 - 1] + values[n // 2]) / 2
        p10 = values[max(0, int(n * 0.10))]
        p90 = values[min(n - 1, int(n * 0.90))]

        result[f"{key}_mean"] = round(mean, 6)
        result[f"{key}_median"] = round(median, 6)
        result[f"{key}_std"] = round(std, 6)
        result[f"{key}_p10"] = round(p10, 6)
        result[f"{key}_p90"] = round(p90, 6)

    return result
