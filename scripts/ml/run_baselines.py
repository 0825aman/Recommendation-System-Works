"""Run all Phase 3 baseline recommenders and record results.

Usage (from project root):
    backend\\.venv\\Scripts\\python.exe scripts/ml/run_baselines.py

Loads the train/test splits, fits each of the 5 baseline recommenders,
evaluates at K in {5, 10, 20}, records results to experiments/results/,
and prints a comparison table.

Prerequisites:
    - Run prepare_splits.py first to create split files
    - Run run_eda.py first (optional, recommended)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.baselines.category_popularity_recommender import CategoryPopularityRecommender
from ml.baselines.popularity_recommender import PopularityRecommender
from ml.baselines.random_recommender import RandomRecommender
from ml.baselines.trending_recommender import TrendingRecommender
from ml.baselines.user_preference_recommender import UserPreferenceRecommender
from ml.data.splits import load_splits
from ml.evaluation.evaluator import Evaluator
from ml.experiments.recorder import record_experiment
from ml.features.item_features import load_item_category_map
from scripts.data.common import get_logger

logger = get_logger(__name__)

_SPLITS_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "data" / "processed" / "kuairec" / "splits" / "split_manifest.json"
)
_RESULTS_DIR = Path(__file__).resolve().parents[2] / "experiments" / "results"


def load_split_manifest() -> dict:  # type: ignore[type-arg]
    """Load the split manifest for dataset provenance."""
    if _SPLITS_MANIFEST.exists():
        with _SPLITS_MANIFEST.open(encoding="utf-8") as f:
            return json.load(f)
    return {}


def run_all_baselines() -> None:
    """Fit, evaluate, and record all 5 Phase 3 baseline recommenders."""
    logger.info("=" * 60)
    logger.info("Phase 3 Baseline Evaluation")
    logger.info("=" * 60)

    # Load splits
    logger.info("Loading splits ...")
    train_df, val_df, test_df = load_splits()
    logger.info(
        "  Train: %d rows | Val: %d rows | Test: %d rows",
        train_df.height, val_df.height, test_df.height,
    )

    # Load item category map
    logger.info("Loading item category map ...")
    item_to_category = load_item_category_map()
    logger.info("  %d items with category labels", len(item_to_category))

    # Build evaluator
    evaluator = Evaluator(k_values=[5, 10, 20], relevance_threshold=0.5)

    # Dataset info for all runs
    manifest = load_split_manifest()
    dataset_info = {
        "name": "KuaiRec 2.0 small_matrix",
        "split": "temporal",
        "train_rows": train_df.height,
        "val_rows": val_df.height,
        "test_rows": test_df.height,
        "split_manifest": manifest,
    }

    # Define all 5 recommenders
    recommenders = [
        RandomRecommender(random_seed=42),
        PopularityRecommender(popularity_metric="weighted_score"),
        CategoryPopularityRecommender(
            item_to_category=item_to_category, n_top_categories=3
        ),
        TrendingRecommender(recency_window_days=14),
        UserPreferenceRecommender(
            item_to_category=item_to_category, n_top_categories=3
        ),
    ]

    all_results = []

    for rec in recommenders:
        logger.info("")
        logger.info("--- %s ---", rec.name)

        t0 = time.perf_counter()
        logger.info("  Fitting ...")
        rec.fit(train_df)
        fit_time = time.perf_counter() - t0

        logger.info("  Evaluating ...")
        result = evaluator.evaluate(
            rec, train_df, test_df, item_to_category=item_to_category
        )
        total_time = time.perf_counter() - t0

        logger.info(
            "  Users evaluated: %d | Fit: %.1fs | Total: %.1fs",
            result["n_eval_users"], fit_time, total_time,
        )

        agg = result["aggregate"]
        for k in [5, 10, 20]:
            logger.info(
                "  @%-3d  P=%.4f  R=%.4f  NDCG=%.4f  MAP=%.4f",
                k,
                agg.get(f"precision_at_{k}_mean", 0.0),
                agg.get(f"recall_at_{k}_mean", 0.0),
                agg.get(f"ndcg_at_{k}_mean", 0.0),
                agg.get(f"map_at_{k}_mean", 0.0),
            )
        logger.info(
            "  Coverage@10=%.4f  Diversity@10=%.4f  Novelty@10=%.4f",
            result["catalog_coverage"],
            agg.get("diversity_mean", 0.0),
            agg.get("novelty_mean", 0.0),
        )

        config = {
            "random_seed": rec.random_seed,
            "k_values": evaluator.k_values,
            "relevance_threshold": evaluator.relevance_threshold,
            "fit_seconds": round(fit_time, 3),
        }

        result_path = record_experiment(
            algorithm_name=rec.name,
            config=config,
            dataset_info=dataset_info,
            evaluation_results=result,
            results_dir=_RESULTS_DIR,
        )
        logger.info("  Saved: %s", result_path.name)
        all_results.append((rec.name, result))

    # Sanity check: Popularity should outperform Random on NDCG@10
    logger.info("")
    logger.info("=== SANITY CHECK ===")
    names = [r[0] for r in all_results]
    if "RandomRecommender" in names and "PopularityRecommender(weighted_score)" in names:
        random_ndcg = all_results[0][1]["aggregate"].get("ndcg_at_10_mean", 0.0)
        pop_ndcg = all_results[1][1]["aggregate"].get("ndcg_at_10_mean", 0.0)
        passed = pop_ndcg >= random_ndcg
        logger.info(
            "  Popularity NDCG@10 (%.4f) >= Random NDCG@10 (%.4f): %s",
            pop_ndcg, random_ndcg, "PASS" if passed else "FAIL"
        )

    # Comparison table
    logger.info("")
    logger.info("=== BASELINE COMPARISON (NDCG@10 mean) ===")
    logger.info("  %-45s  NDCG@10   P@10    R@10    Coverage", "Algorithm")
    logger.info("  " + "-" * 85)
    for name, result in all_results:
        agg = result["aggregate"]
        logger.info(
            "  %-45s  %.4f    %.4f  %.4f  %.4f",
            name,
            agg.get("ndcg_at_10_mean", 0.0),
            agg.get("precision_at_10_mean", 0.0),
            agg.get("recall_at_10_mean", 0.0),
            result["catalog_coverage"],
        )

    logger.info("")
    logger.info("Results written to: %s", _RESULTS_DIR)


if __name__ == "__main__":
    run_all_baselines()
