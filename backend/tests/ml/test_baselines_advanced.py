"""Tests for CategoryPopularityRecommender, TrendingRecommender, UserPreferenceRecommender."""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.baselines.category_popularity_recommender import CategoryPopularityRecommender
from ml.baselines.trending_recommender import TrendingRecommender
from ml.baselines.user_preference_recommender import UserPreferenceRecommender


class TestCategoryPopularityRecommender:
    def test_returns_n_items(
        self, tiny_interactions: pl.DataFrame, category_map: dict[int, str]
    ) -> None:
        rec = CategoryPopularityRecommender(item_to_category=category_map)
        rec.fit(tiny_interactions)
        result = rec.recommend(0, n=5, exclude_seen=set())
        assert len(result) == 5

    def test_no_duplicates(
        self, tiny_interactions: pl.DataFrame, category_map: dict[int, str]
    ) -> None:
        rec = CategoryPopularityRecommender(item_to_category=category_map)
        rec.fit(tiny_interactions)
        result = rec.recommend(0, n=8, exclude_seen=set())
        assert len(result) == len(set(result))

    def test_excludes_seen_items(
        self, tiny_interactions: pl.DataFrame, category_map: dict[int, str]
    ) -> None:
        rec = CategoryPopularityRecommender(item_to_category=category_map)
        rec.fit(tiny_interactions)
        seen = {100, 101, 102}
        result = rec.recommend(0, n=5, exclude_seen=seen)
        assert not any(item in seen for item in result)

    def test_cold_user_falls_back_to_global(
        self, tiny_interactions: pl.DataFrame, category_map: dict[int, str]
    ) -> None:
        rec = CategoryPopularityRecommender(item_to_category=category_map)
        rec.fit(tiny_interactions)
        # user_id 999 not in training
        result = rec.recommend(999, n=5, exclude_seen=set())
        assert len(result) == 5

    def test_no_category_map_uses_global_fallback(
        self, tiny_interactions: pl.DataFrame
    ) -> None:
        rec = CategoryPopularityRecommender(item_to_category={})
        rec.fit(tiny_interactions)
        result = rec.recommend(0, n=5, exclude_seen=set())
        assert len(result) == 5

    def test_recommend_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            CategoryPopularityRecommender().recommend(0, n=5, exclude_seen=set())


class TestTrendingRecommender:
    def test_returns_n_items(self, tiny_interactions: pl.DataFrame) -> None:
        rec = TrendingRecommender(recency_window_days=30)
        rec.fit(tiny_interactions)
        result = rec.recommend(0, n=5, exclude_seen=set())
        assert len(result) == 5

    def test_no_duplicates(self, tiny_interactions: pl.DataFrame) -> None:
        rec = TrendingRecommender(recency_window_days=30)
        rec.fit(tiny_interactions)
        result = rec.recommend(0, n=8, exclude_seen=set())
        assert len(result) == len(set(result))

    def test_excludes_seen_items(self, tiny_interactions: pl.DataFrame) -> None:
        rec = TrendingRecommender(recency_window_days=30)
        rec.fit(tiny_interactions)
        seen = {100, 101}
        result = rec.recommend(0, n=5, exclude_seen=seen)
        assert 100 not in result and 101 not in result

    def test_very_narrow_window_falls_back(self, tiny_interactions: pl.DataFrame) -> None:
        # Window of 0 days — empty recent set, should fall back gracefully
        # (returns at least 1 item from fallback, may be < n due to narrow window)
        rec = TrendingRecommender(recency_window_days=0)
        rec.fit(tiny_interactions)
        result = rec.recommend(0, n=5, exclude_seen=set())
        assert len(result) >= 1
        assert len(set(result)) == len(result)  # no duplicates

    def test_no_timestamp_column_handled(self) -> None:
        # DataFrame without valid timestamps (all null)
        df = pl.DataFrame({
            "user_id": [0, 1, 2],
            "video_id": [100, 101, 102],
            "watch_ratio": [0.5, 0.8, 0.3],
            "timestamp": [None, None, None],
        }).with_columns([
            pl.col("user_id").cast(pl.Int64),
            pl.col("video_id").cast(pl.Int64),
            pl.col("watch_ratio").cast(pl.Float64),
            pl.col("timestamp").cast(pl.Float64),
        ])
        rec = TrendingRecommender()
        rec.fit(df)
        result = rec.recommend(0, n=3, exclude_seen=set())
        assert len(result) >= 1

    def test_recommend_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            TrendingRecommender().recommend(0, n=5, exclude_seen=set())


class TestUserPreferenceRecommender:
    def test_returns_n_items(
        self, tiny_interactions: pl.DataFrame, category_map: dict[int, str]
    ) -> None:
        rec = UserPreferenceRecommender(item_to_category=category_map)
        rec.fit(tiny_interactions)
        result = rec.recommend(0, n=5, exclude_seen=set())
        assert len(result) == 5

    def test_no_duplicates(
        self, tiny_interactions: pl.DataFrame, category_map: dict[int, str]
    ) -> None:
        rec = UserPreferenceRecommender(item_to_category=category_map)
        rec.fit(tiny_interactions)
        result = rec.recommend(0, n=8, exclude_seen=set())
        assert len(result) == len(set(result))

    def test_excludes_seen_items(
        self, tiny_interactions: pl.DataFrame, category_map: dict[int, str]
    ) -> None:
        rec = UserPreferenceRecommender(item_to_category=category_map)
        rec.fit(tiny_interactions)
        seen = {100, 101}
        result = rec.recommend(0, n=5, exclude_seen=seen)
        assert 100 not in result and 101 not in result

    def test_cold_user_falls_back_to_global(
        self, tiny_interactions: pl.DataFrame, category_map: dict[int, str]
    ) -> None:
        rec = UserPreferenceRecommender(item_to_category=category_map)
        rec.fit(tiny_interactions)
        result = rec.recommend(9999, n=5, exclude_seen=set())
        assert len(result) == 5

    def test_recommend_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            UserPreferenceRecommender().recommend(0, n=5, exclude_seen=set())
