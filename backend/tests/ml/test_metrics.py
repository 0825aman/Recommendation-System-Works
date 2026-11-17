"""Tests for evaluation metrics (ml/evaluation/metrics.py).

Uses known inputs with pre-computed expected outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

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


class TestPrecisionAtK:
    def test_perfect_precision(self) -> None:
        assert precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0

    def test_zero_precision(self) -> None:
        assert precision_at_k([4, 5, 6], {1, 2, 3}, k=3) == 0.0

    def test_partial_precision(self) -> None:
        assert precision_at_k([1, 4, 2], {1, 2, 3}, k=3) == pytest.approx(2 / 3)

    def test_cutoff_respected(self) -> None:
        # Only top-2 considered; item 3 at rank 3 is ignored
        assert precision_at_k([4, 5, 1], {1}, k=2) == 0.0

    def test_empty_recommended(self) -> None:
        assert precision_at_k([], {1, 2}, k=5) == 0.0

    def test_k_zero_returns_zero(self) -> None:
        assert precision_at_k([1, 2, 3], {1}, k=0) == 0.0


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        assert recall_at_k([1, 2, 3], {1, 2, 3}, k=3) == 1.0

    def test_zero_recall(self) -> None:
        assert recall_at_k([4, 5, 6], {1, 2, 3}, k=3) == 0.0

    def test_partial_recall(self) -> None:
        # 2 of 3 relevant items retrieved
        assert recall_at_k([1, 2, 4], {1, 2, 3}, k=3) == pytest.approx(2 / 3)

    def test_empty_relevant_returns_zero(self) -> None:
        assert recall_at_k([1, 2, 3], set(), k=3) == 0.0


class TestNdcgAtK:
    def test_perfect_ndcg(self) -> None:
        # All relevant items in correct order → NDCG = 1.0
        assert ndcg_at_k([1, 2, 3], {1, 2, 3}, k=3) == pytest.approx(1.0)

    def test_zero_ndcg(self) -> None:
        assert ndcg_at_k([4, 5, 6], {1, 2, 3}, k=3) == 0.0

    def test_higher_rank_better(self) -> None:
        # Item 1 at rank 1 is better than item 1 at rank 3
        ndcg_rank1 = ndcg_at_k([1, 4, 5], {1}, k=3)
        ndcg_rank3 = ndcg_at_k([4, 5, 1], {1}, k=3)
        assert ndcg_rank1 > ndcg_rank3

    def test_empty_relevant_returns_zero(self) -> None:
        assert ndcg_at_k([1, 2, 3], set(), k=3) == 0.0

    def test_known_value(self) -> None:
        # DCG = 1/log2(2) + 1/log2(3) = 1 + 0.631
        # IDCG = 1/log2(2) + 1/log2(3) = same (all 2 relevant found)
        assert ndcg_at_k([1, 2, 3], {1, 2}, k=3) == pytest.approx(1.0)


class TestAveragePrecisionAtK:
    def test_perfect_ap(self) -> None:
        assert average_precision_at_k([1, 2, 3], {1, 2, 3}, k=3) == pytest.approx(1.0)

    def test_zero_ap(self) -> None:
        assert average_precision_at_k([4, 5, 6], {1, 2, 3}, k=3) == 0.0

    def test_known_value(self) -> None:
        # Hits at rank 1 and 3: AP = (1/1 + 2/3) / 2 = 0.833
        assert average_precision_at_k([1, 4, 2], {1, 2}, k=3) == pytest.approx((1.0 + 2 / 3) / 2)


class TestCatalogCoverage:
    def test_full_coverage(self) -> None:
        recs = [[1, 2], [3, 4]]
        assert catalog_coverage(recs, catalog_size=4) == 1.0

    def test_partial_coverage(self) -> None:
        recs = [[1, 2], [1, 2]]
        assert catalog_coverage(recs, catalog_size=4) == pytest.approx(0.5)

    def test_zero_catalog_size(self) -> None:
        assert catalog_coverage([[1, 2]], catalog_size=0) == 0.0


class TestIntraListDiversity:
    def test_all_same_category_zero_diversity(self, category_map: dict[int, str]) -> None:
        # Items 100, 105, 110 → all "dance" (100%5=0 → dance)
        same_cat_items = [i for i, cat in category_map.items() if cat == "dance"][:3]
        result = intra_list_diversity(same_cat_items, category_map)
        assert result == 0.0

    def test_all_different_categories(self) -> None:
        cat_map = {1: "a", 2: "b", 3: "c"}
        result = intra_list_diversity([1, 2, 3], cat_map)
        assert result == pytest.approx(1.0)

    def test_too_few_items_returns_zero(self) -> None:
        assert intra_list_diversity([1], {1: "dance"}) == 0.0


class TestNovelty:
    def test_rare_items_have_higher_novelty(self) -> None:
        pop = {1: 1000, 2: 10}  # item 1 is popular, item 2 is rare
        nov_popular = novelty([1], pop, n_users=1000)
        nov_rare = novelty([2], pop, n_users=1000)
        assert nov_rare > nov_popular

    def test_zero_users_returns_zero(self) -> None:
        assert novelty([1, 2], {1: 10, 2: 5}, n_users=0) == 0.0

    def test_empty_recommendations_returns_zero(self) -> None:
        assert novelty([], {1: 10}, n_users=100) == 0.0


class TestAggregateMetrics:
    def test_known_values(self) -> None:
        per_user = [
            {"precision_at_10": 0.3},
            {"precision_at_10": 0.5},
            {"precision_at_10": 0.7},
        ]
        agg = aggregate_metrics(per_user)
        assert agg["precision_at_10_mean"] == pytest.approx(0.5)
        assert agg["precision_at_10_p10"] == pytest.approx(0.3)
        assert agg["precision_at_10_p90"] == pytest.approx(0.7)

    def test_empty_input_returns_empty(self) -> None:
        assert aggregate_metrics([]) == {}
