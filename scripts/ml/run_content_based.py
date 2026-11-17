"""Run Phase 4A — Content-Based Filtering evaluation.

Fits ContentBasedRecommender on the training split and evaluates it
on the test split using the same Evaluator and metrics as Phase 3.

Usage (from project root):
    backend\\.venv\\Scripts\\python.exe scripts/ml/run_content_based.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.content_based.content_recommender import ContentBasedRecommender
from ml.data.loader import load_caption_category, load_item_categories
from ml.data.splits import load_splits
from ml.evaluation.evaluator import Evaluator
from ml.experiments.recorder import record_experiment
from ml.features.item_features import load_item_category_map

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    print("=" * 60)
    print("Phase 4A — Content-Based Filtering")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\nLoading splits...")
    train_df, val_df, test_df = load_splits()
    print(f"  train: {train_df.height:,} rows")
    print(f"  val:   {val_df.height:,} rows")
    print(f"  test:  {test_df.height:,} rows")

    print("\nLoading item metadata...")
    item_categories_df = load_item_categories()
    caption_category_df = load_caption_category()
    item_to_category = load_item_category_map()
    print(f"  item_categories: {item_categories_df.height:,} items")
    print(f"  captions:        {caption_category_df.height:,} items")

    # ------------------------------------------------------------------
    # Hyperparameter configurations to evaluate on VALIDATION split
    # ------------------------------------------------------------------
    val_configs = [
        {"tfidf_weight": 0.0, "category_weight": 1.0},
        {"tfidf_weight": 0.25, "category_weight": 1.0},
        {"tfidf_weight": 0.5, "category_weight": 1.0},
        {"tfidf_weight": 0.75, "category_weight": 1.0},
    ]

    evaluator = Evaluator(k_values=[5, 10, 20], relevance_threshold=0.5)

    print("\n--- Hyperparameter tuning on VALIDATION split ---")
    best_cfg = val_configs[0]
    best_val_ndcg = -1.0

    for cfg in val_configs:
        rec = ContentBasedRecommender(
            item_categories_df=item_categories_df,
            caption_category_df=caption_category_df,
            tfidf_weight=cfg["tfidf_weight"],
            category_weight=cfg["category_weight"],
            tfidf_max_features=500,
        )
        rec.fit(train_df)
        val_results = evaluator.evaluate(rec, train_df, val_df, item_to_category)
        val_ndcg = val_results["aggregate"].get("ndcg_at_10_mean", 0.0)
        print(
            f"  tfidf_weight={cfg['tfidf_weight']:.2f}  "
            f"val NDCG@10={val_ndcg:.4f}  "
            f"({rec.fit_seconds:.1f}s fit)"
        )
        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_cfg = cfg

    print(f"\nBest config: tfidf_weight={best_cfg['tfidf_weight']}")
    print(f"Best val NDCG@10: {best_val_ndcg:.4f}")

    # ------------------------------------------------------------------
    # Final evaluation on TEST split with best config
    # ------------------------------------------------------------------
    print("\n--- Final evaluation on TEST split ---")
    best_rec = ContentBasedRecommender(
        item_categories_df=item_categories_df,
        caption_category_df=caption_category_df,
        tfidf_weight=best_cfg["tfidf_weight"],
        category_weight=best_cfg["category_weight"],
        tfidf_max_features=500,
    )
    best_rec.fit(train_df)
    test_results = evaluator.evaluate(best_rec, train_df, test_df, item_to_category)

    agg = test_results["aggregate"]
    print(f"\n{best_rec.name}")
    print(f"  NDCG@10  = {agg.get('ndcg_at_10_mean', 0):.4f}")
    print(f"  P@10     = {agg.get('precision_at_10_mean', 0):.4f}")
    print(f"  R@10     = {agg.get('recall_at_10_mean', 0):.4f}")
    print(f"  MAP@10   = {agg.get('map_at_10_mean', 0):.4f}")
    print(f"  Coverage = {test_results.get('catalog_coverage', 0):.4f}")
    print(f"  Runtime  = {test_results.get('runtime_seconds', 0):.1f}s")

    # Phase 3 reference
    print("\nPhase 3 reference (PopularityRecommender):")
    print("  NDCG@10  = 0.4461")

    if agg.get("ndcg_at_10_mean", 0) > 0.4461:
        print("  ✓ Beats Phase 3 best baseline on NDCG@10")
    else:
        print("  ✗ Does NOT beat Phase 3 best baseline (expected on dense matrix)")

    # ------------------------------------------------------------------
    # Record experiment
    # ------------------------------------------------------------------
    dataset_info = {
        "name": "KuaiRec 2.0 small_matrix",
        "split": "temporal",
        "train_rows": train_df.height,
        "val_rows": val_df.height,
        "test_rows": test_df.height,
    }
    config = {
        "tfidf_weight": best_cfg["tfidf_weight"],
        "category_weight": best_cfg["category_weight"],
        "tfidf_max_features": 500,
        "weight_by_watch_ratio": True,
        "best_val_ndcg10": round(best_val_ndcg, 6),
        "k_values": [5, 10, 20],
        "relevance_threshold": 0.5,
        "fit_seconds": round(best_rec.fit_seconds, 3),
    }
    out_path = record_experiment(
        algorithm_name=best_rec.name,
        config=config,
        dataset_info=dataset_info,
        evaluation_results=test_results,
    )
    print(f"\nSaved → {out_path}")
    print("\nPhase 4A content-based evaluation complete.")


if __name__ == "__main__":
    main()
