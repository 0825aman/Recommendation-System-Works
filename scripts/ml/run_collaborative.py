"""Run Phase 4B — Collaborative Filtering evaluation.

Fits UserBasedCF, ItemBasedCF, and SVDRecommender on the training split,
tunes hyperparameters on validation, and evaluates on the test split.

Usage (from project root):
    backend\\.venv\\Scripts\\python.exe scripts/ml/run_collaborative.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.collaborative.item_cf import ItemBasedCF
from ml.collaborative.matrix_factorization import SVDRecommender
from ml.collaborative.user_cf import UserBasedCF
from ml.data.splits import load_splits
from ml.evaluation.evaluator import Evaluator
from ml.experiments.recorder import record_experiment
from ml.features.item_features import load_item_category_map

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def evaluate_algorithm(
    rec_class: type,
    configs: list[dict],  # type: ignore[type-arg]
    train_df,  # type: ignore[no-untyped-def]
    val_df,  # type: ignore[no-untyped-def]
    test_df,  # type: ignore[no-untyped-def]
    item_to_category: dict[int, str],
    evaluator: Evaluator,
) -> tuple[dict, dict, object]:  # type: ignore[type-ignore]
    """Tune on val, evaluate on test; return (best_config, test_results, fitted_rec)."""
    best_cfg = configs[0]
    best_val_ndcg = -1.0

    for cfg in configs:
        rec = rec_class(**cfg)
        rec.fit(train_df)
        val_results = evaluator.evaluate(rec, train_df, val_df, item_to_category)
        val_ndcg = val_results["aggregate"].get("ndcg_at_10_mean", 0.0)
        fit_s = getattr(rec, "fit_seconds", 0.0)
        print(f"    {cfg}  val NDCG@10={val_ndcg:.4f}  ({fit_s:.1f}s fit)")
        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_cfg = cfg

    # Final eval on test with best config
    best_rec = rec_class(**best_cfg)
    best_rec.fit(train_df)
    test_results = evaluator.evaluate(best_rec, train_df, test_df, item_to_category)
    best_cfg["best_val_ndcg10"] = round(best_val_ndcg, 6)
    return best_cfg, test_results, best_rec


def main() -> None:
    print("=" * 60)
    print("Phase 4B — Collaborative Filtering")
    print("=" * 60)

    print("\nLoading splits...")
    train_df, val_df, test_df = load_splits()
    item_to_category = load_item_category_map()
    print(f"  train: {train_df.height:,}  val: {val_df.height:,}  test: {test_df.height:,}")

    evaluator = Evaluator(k_values=[5, 10, 20], relevance_threshold=0.5)

    # ------------------------------------------------------------------
    # SVD — fast, tune n_components
    # ------------------------------------------------------------------
    print("\n--- SVDRecommender ---")
    svd_configs = [
        {"n_components": 20, "random_state": 42},
        {"n_components": 50, "random_state": 42},
        {"n_components": 100, "random_state": 42},
    ]
    svd_cfg, svd_results, svd_rec = evaluate_algorithm(
        SVDRecommender, svd_configs, train_df, val_df, test_df, item_to_category, evaluator
    )
    agg = svd_results["aggregate"]
    print(f"\nSVDRecommender best: {svd_cfg}")
    print(f"  NDCG@10  = {agg.get('ndcg_at_10_mean', 0):.4f}")
    print(f"  P@10     = {agg.get('precision_at_10_mean', 0):.4f}")
    print(f"  Coverage = {svd_results.get('catalog_coverage', 0):.4f}")

    out = record_experiment(
        algorithm_name=svd_rec.name,
        config={**svd_cfg, "k_values": [5, 10, 20], "relevance_threshold": 0.5,
                "fit_seconds": round(getattr(svd_rec, "fit_seconds", 0), 3)},
        dataset_info={"name": "KuaiRec 2.0 small_matrix", "split": "temporal",
                      "train_rows": train_df.height, "val_rows": val_df.height,
                      "test_rows": test_df.height},
        evaluation_results=svd_results,
    )
    print(f"  Saved -> {out}")

    # ------------------------------------------------------------------
    # UserBasedCF — tune n_neighbors
    # ------------------------------------------------------------------
    print("\n--- UserBasedCF ---")
    ucf_configs = [
        {"n_neighbors": 10},
        {"n_neighbors": 50},
        {"n_neighbors": 100},
    ]
    ucf_cfg, ucf_results, ucf_rec = evaluate_algorithm(
        UserBasedCF, ucf_configs, train_df, val_df, test_df, item_to_category, evaluator
    )
    agg = ucf_results["aggregate"]
    print(f"\nUserBasedCF best: {ucf_cfg}")
    print(f"  NDCG@10  = {agg.get('ndcg_at_10_mean', 0):.4f}")
    print(f"  P@10     = {agg.get('precision_at_10_mean', 0):.4f}")
    print(f"  Coverage = {ucf_results.get('catalog_coverage', 0):.4f}")

    out = record_experiment(
        algorithm_name=ucf_rec.name,
        config={**ucf_cfg, "k_values": [5, 10, 20], "relevance_threshold": 0.5,
                "fit_seconds": round(getattr(ucf_rec, "fit_seconds", 0), 3)},
        dataset_info={"name": "KuaiRec 2.0 small_matrix", "split": "temporal",
                      "train_rows": train_df.height, "val_rows": val_df.height,
                      "test_rows": test_df.height},
        evaluation_results=ucf_results,
    )
    print(f"  Saved -> {out}")

    # ------------------------------------------------------------------
    # ItemBasedCF — tune n_neighbors
    # ------------------------------------------------------------------
    print("\n--- ItemBasedCF ---")
    icf_configs = [
        {"n_neighbors": 10},
        {"n_neighbors": 50},
    ]
    icf_cfg, icf_results, icf_rec = evaluate_algorithm(
        ItemBasedCF, icf_configs, train_df, val_df, test_df, item_to_category, evaluator
    )
    agg = icf_results["aggregate"]
    print(f"\nItemBasedCF best: {icf_cfg}")
    print(f"  NDCG@10  = {agg.get('ndcg_at_10_mean', 0):.4f}")
    print(f"  P@10     = {agg.get('precision_at_10_mean', 0):.4f}")
    print(f"  Coverage = {icf_results.get('catalog_coverage', 0):.4f}")

    out = record_experiment(
        algorithm_name=icf_rec.name,
        config={**icf_cfg, "k_values": [5, 10, 20], "relevance_threshold": 0.5,
                "fit_seconds": round(getattr(icf_rec, "fit_seconds", 0), 3)},
        dataset_info={"name": "KuaiRec 2.0 small_matrix", "split": "temporal",
                      "train_rows": train_df.height, "val_rows": val_df.height,
                      "test_rows": test_df.height},
        evaluation_results=icf_results,
    )
    print(f"  Saved -> {out}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Phase 4B Summary (NDCG@10 on test split)")
    print("=" * 60)
    print(f"  {'Algorithm':<40} {'NDCG@10':>8}")
    print(f"  {'-'*40} {'-'*8}")
    print(f"  {'PopularityRecommender (P3 best)':<40} {'0.4461':>8}")
    print(f"  {'ContentBasedRecommender (P4A)':<40} {'0.0247':>8}")
    ndcg_svd = svd_results['aggregate'].get('ndcg_at_10_mean', 0)
    ndcg_ucf = ucf_results['aggregate'].get('ndcg_at_10_mean', 0)
    ndcg_icf = icf_results['aggregate'].get('ndcg_at_10_mean', 0)
    print(f"  {svd_rec.name:<40} {ndcg_svd:>8.4f}")
    print(f"  {ucf_rec.name:<40} {ndcg_ucf:>8.4f}")
    print(f"  {icf_rec.name:<40} {ndcg_icf:>8.4f}")
    print("\nPhase 4B collaborative filtering evaluation complete.")


if __name__ == "__main__":
    main()
