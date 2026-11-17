"""Tests for RandomRecommender and PopularityRecommender."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.baselines.popularity_recommender import PopularityRecommender
from ml.baselines.random_recommender import RandomRecommender


class TestRandomRecommender:
    def test_returns_n_items(self, tiny_interactions: pl.DataFrame) -> None:
        rec = RandomRecommender(random_seed=42)
        rec.fit(tiny_interactions)
        result = rec.recommend(0, n=5, exclude_seen=set())
        assert len(result) == 5

    def test_no_duplicates(self, tiny_interactions: pl.DataFrame) -> None:
        rec = RandomRecommender(random_seed=42)
        rec.fit(tiny_interactions)
        result = rec.recommend(0, n=8, exclude_seen=set())
        assert len(result) == len(set(result))

    def test_excludes_seen_items(self, tiny_interactions: pl.DataFrame) -> None:
        rec = RandomRecommender(random_seed=42)
        rec.fit(tiny_interactions)
        seen = {100, 101, 102}
        result = rec.recommend(0, n=5, exclude_seen=seen)
        assert not any(item in seen for item in result)

    def test_deterministic_with_same_seed(self, tiny_interactions: pl.DataFrame) -> None:
        rec1 = RandomRecommender(random_seed=99)
        rec1.fit(tiny_interactions)
        rec2 = RandomRecommender(random_seed=99)
        rec2.fit(tiny_interactions)
        assert rec1.recommend(0, n=5, exclude_seen=set()) == rec2.recommend(0, n=5, exclude_seen=set())

    def test_different_seeds_different_results(self, tiny_interactions: pl.DataFrame) -> None:
        rec1 = RandomRecommender(random_seed=1)
        rec1.fit(tiny_interactions)
        rec2 = RandomRecommender(random_seed=2)
        rec2.fit(tiny_interactions)
        # Very high probability they differ with 10 items
        assert rec1.recommend(0, n=5, exclude_seen=set()) != rec2.recommend(0, n=5, exclude_seen=set())

    def test_recommend_before_fit_raises(self) -> None:
        rec = RandomRecommender()
        with pytest.raises(RuntimeError):
            rec.recommend(0, n=5, exclude_seen=set())

    def test_handles_all_items_seen(self, tiny_interactions: pl.DataFrame) -> None:
        rec = RandomRecommender(random_seed=42)
        rec.fit(tiny_interactions)
        all_items = set(tiny_interactions["video_id"].unique().to_list())
        result = rec.recommend(0, n=5, exclude_seen=all_items)
        assert result == []

    def test_name_property(self) -> None:
        assert RandomRecommender().name == "RandomRecommender"


class TestPopularityRecommender:
    def test_returns_n_items(self, tiny_interactions: pl.DataFrame) -> None:
        rec = PopularityRecommender()
        rec.fit(tiny_interactions)
        result = rec.recommend(0, n=5, exclude_seen=set())
        assert len(result) == 5

    def test_no_duplicates(self, tiny_interactions: pl.DataFrame) -> None:
        rec = PopularityRecommender()
        rec.fit(tiny_interactions)
        result = rec.recommend(0, n=8, exclude_seen=set())
        assert len(result) == len(set(result))

    def test_excludes_seen_items(self, tiny_interactions: pl.DataFrame) -> None:
        rec = PopularityRecommender()
        rec.fit(tiny_interactions)
        seen = {100, 101}
        result = rec.recommend(0, n=5, exclude_seen=seen)
        assert 100 not in result
        assert 101 not in result

    def test_same_ranking_for_all_users(self, tiny_interactions: pl.DataFrame) -> None:
        rec = PopularityRecommender()
        rec.fit(tiny_interactions)
        r0 = rec.recommend(0, n=5, exclude_seen=set())
        r1 = rec.recommend(1, n=5, exclude_seen=set())
        assert r0 == r1  # popularity is global — same for all users

    def test_invalid_metric_raises(self) -> None:
        with pytest.raises(ValueError):
            PopularityRecommender(popularity_metric="invalid")

    def test_interaction_count_mode(self, tiny_interactions: pl.DataFrame) -> None:
        rec = PopularityRecommender(popularity_metric="interaction_count")
        rec.fit(tiny_interactions)
        result = rec.recommend(0, n=5, exclude_seen=set())
        assert len(result) == 5

    def test_recommend_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            PopularityRecommender().recommend(0, n=5, exclude_seen=set())
