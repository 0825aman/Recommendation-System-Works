"""Evaluation runner for Phase 3 baseline recommenders.

Runs a fitted recommender against the test split and computes
Precision@K, Recall@K, NDCG@K, MAP@K, Coverage, Diversity, Novelty.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import polars as pl

from ml.evaluation.metrics import (
    aggregate_metrics,
    average_precision_at_k,
    catalog_coverage,
    intra_list_diversity,
    ndcg_at_k,
    novelty,
    precision_at_k,
    recall_at_k,
)

if TYPE_CHECKING:
    from ml.baselines.base import BaseRecommender


class Evaluator:
    """Offline evaluator for recommendation algorithms.

    Usage:
        evaluator = Evaluator(config=config)
        results = evaluator.evaluate(recommender, train_df, test_df, item_categories_df)
    """

    def __init__(
        self,
        k_values: list[int] | None = None,
        relevance_threshold: float = 0.5,
        min_test_interactions: int = 1,
    ) -> None:
        """Initialise the evaluator.

        Args:
            k_values:               K cutoffs to evaluate at. Default [5, 10, 20].
            relevance_threshold:    watch_ratio threshold for binary relevance (ADR-P3-02).
            min_test_interactions:  Minimum relevant test interactions to include a user.
        """
        self.k_values = k_values or [5, 10, 20]
        self.relevance_threshold = relevance_threshold
        self.min_test_interactions = min_test_interactions

    def evaluate(
        self,
        recommender: BaseRecommender,
        train_df: pl.DataFrame,
        test_df: pl.DataFrame,
        item_to_category: dict[int, str] | None = None,
    ) -> dict:  # type: ignore[type-arg]
        """Run full evaluation of a fitted recommender.

        Args:
            recommender:      A fitted BaseRecommender instance.
            train_df:         Training split (used to build seen-item sets).
            test_df:          Test split (ground truth).
            item_to_category: Optional mapping for diversity computation.

        Returns:
            Dict with per-user metrics, aggregate metrics, coverage, runtime.
        """
        t0 = time.perf_counter()
        k_max = max(self.k_values)
        item_to_cat: dict[int, str] = item_to_category or {}

        # Build seen-item sets (from training data only — no leakage)
        seen_per_user: dict[int, set[int]] = {}
        for row in train_df.select(["user_id", "video_id"]).iter_rows():
            uid, vid = row
            seen_per_user.setdefault(uid, set()).add(vid)

        # Build item popularity from training data (for novelty computation)
        item_pop_df = (
            train_df.group_by("video_id")
            .agg(pl.len().alias("count"))
        )
        item_popularity: dict[int, float] = dict(
            zip(item_pop_df["video_id"].to_list(), item_pop_df["count"].to_list())
        )
        n_train_users = train_df["user_id"].n_unique()

        # Build ground-truth relevant sets from test data
        relevant_per_user: dict[int, set[int]] = {}
        for row in test_df.filter(
            pl.col("watch_ratio") >= self.relevance_threshold
        ).select(["user_id", "video_id"]).iter_rows():
            uid, vid = row
            relevant_per_user.setdefault(uid, set()).add(vid)

        # Identify evaluable users (have enough relevant test interactions)
        eval_users = [
            uid for uid, rel in relevant_per_user.items()
            if len(rel) >= self.min_test_interactions
        ]

        per_user_metrics: list[dict[str, float]] = []  # type: ignore[type-arg]
        all_recommendations: list[list[int]] = []

        for uid in eval_users:
            seen = seen_per_user.get(uid, set())
            relevant = relevant_per_user[uid]

            recs = recommender.recommend(uid, n=k_max, exclude_seen=seen)
            all_recommendations.append(recs)

            user_metrics: dict[str, float] = {}  # type: ignore[type-arg]
            for k in self.k_values:
                user_metrics[f"precision_at_{k}"] = precision_at_k(recs, relevant, k)
                user_metrics[f"recall_at_{k}"] = recall_at_k(recs, relevant, k)
                user_metrics[f"ndcg_at_{k}"] = ndcg_at_k(recs, relevant, k)
                user_metrics[f"map_at_{k}"] = average_precision_at_k(recs, relevant, k)

            # Diversity and novelty at primary K=10
            k_primary = 10 if 10 in self.k_values else self.k_values[-1]
            user_metrics["diversity"] = intra_list_diversity(recs[:k_primary], item_to_cat)
            user_metrics["novelty"] = novelty(recs[:k_primary], item_popularity, n_train_users)

            per_user_metrics.append(user_metrics)

        # Catalog
        catalog = list(item_popularity.keys())
        coverage = catalog_coverage(all_recommendations, len(catalog))

        # Aggregate
        agg = aggregate_metrics(per_user_metrics)

        elapsed = time.perf_counter() - t0

        return {
            "recommender": recommender.name,
            "n_eval_users": len(eval_users),
            "n_total_test_users": test_df["user_id"].n_unique(),
            "k_values": self.k_values,
            "relevance_threshold": self.relevance_threshold,
            "catalog_coverage": round(coverage, 6),
            "catalog_size": len(catalog),
            "aggregate": agg,
            "runtime_seconds": round(elapsed, 3),
        }
