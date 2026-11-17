"""Tests for the data cleaning pipeline.

Uses synthetic Polars DataFrames directly — no CSV or filesystem I/O required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.data.clean_dataset import (
    clean_caption_category,
    clean_item_categories,
    clean_small_matrix,
    clean_user_features,
    clean_user_features_raw,
)

# ---------------------------------------------------------------------------
# clean_small_matrix
# ---------------------------------------------------------------------------

class TestCleanSmallMatrix:
    """Tests for small_matrix cleaning rules."""

    def _make_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "user_id": [0, 1, 2],
            "video_id": [100, 101, 102],
            "play_duration": [5000, 10000, 3000],
            "video_duration": [10000, 10000, 10000],
            "time": ["2020-07-05 00:08:23.438", "2020-07-06 10:00:00.000", None],
            "date": [20200705.0, 20200706.0, None],
            "timestamp": [1593878903.0, 1593979200.0, None],
            "watch_ratio": [0.5, 1.0, 0.3],
        })

    def test_row_count_preserved(self) -> None:
        df = self._make_df()
        result = clean_small_matrix(df, [])
        assert result.height == df.height

    def test_time_parsed_to_datetime(self) -> None:
        df = self._make_df()
        result = clean_small_matrix(df, [])
        assert result["time"].dtype == pl.Datetime

    def test_time_nulls_preserved(self) -> None:
        df = self._make_df()
        result = clean_small_matrix(df, [])
        # Row 2 had null time — must remain null
        assert result["time"].null_count() == 1

    def test_date_cast_to_int64(self) -> None:
        df = self._make_df()
        result = clean_small_matrix(df, [])
        assert result["date"].dtype == pl.Int64

    def test_play_duration_is_int64(self) -> None:
        df = self._make_df()
        result = clean_small_matrix(df, [])
        assert result["play_duration"].dtype == pl.Int64

    def test_video_duration_is_int64(self) -> None:
        df = self._make_df()
        result = clean_small_matrix(df, [])
        assert result["video_duration"].dtype == pl.Int64

    def test_log_entries_recorded(self) -> None:
        df = self._make_df()
        log: list = []
        clean_small_matrix(df, log)
        assert len(log) >= 3  # parse_time, cast_date, cast_durations


# ---------------------------------------------------------------------------
# clean_user_features
# ---------------------------------------------------------------------------

class TestCleanUserFeatures:
    """Tests for user_features cleaning rules."""

    def _make_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "user_id": [0, 1, 2],
            "user_active_degree": ["high_active", "full_active", "UNKNOWN"],
            "is_lowactive_period": [0, 1, 0],
            "is_live_streamer": [1, 0, 0],
            "is_video_author": [0, 1, 0],
            "onehot_feat0": [0, 1, 0],
            "onehot_feat12": [1, 0, 0],
        })

    def test_row_count_preserved(self) -> None:
        df = self._make_df()
        result = clean_user_features(df, [])
        assert result.height == df.height

    def test_binary_flags_cast_to_boolean(self) -> None:
        df = self._make_df()
        result = clean_user_features(df, [])
        for col in ["is_lowactive_period", "is_live_streamer", "is_video_author"]:
            assert result[col].dtype == pl.Boolean, f"{col} should be Boolean"

    def test_non_binary_col_unchanged(self) -> None:
        df = self._make_df()
        result = clean_user_features(df, [])
        assert result["user_id"].dtype == pl.Int64


# ---------------------------------------------------------------------------
# clean_user_features_raw
# ---------------------------------------------------------------------------

class TestCleanUserFeaturesRaw:
    """Tests for user_features_raw cleaning rules."""

    def _make_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "user_id": [0, 1],
            "is_lowactive_period": [0, 1],
            "is_live_author": [1, 0],
            "os_version": ["android10", "ios14"],
        })

    def test_row_count_preserved(self) -> None:
        df = self._make_df()
        result = clean_user_features_raw(df, [])
        assert result.height == df.height

    def test_binary_flags_cast_to_boolean(self) -> None:
        df = self._make_df()
        result = clean_user_features_raw(df, [])
        assert result["is_lowactive_period"].dtype == pl.Boolean
        assert result["is_live_author"].dtype == pl.Boolean

    def test_os_version_is_string(self) -> None:
        df = self._make_df()
        result = clean_user_features_raw(df, [])
        assert result["os_version"].dtype == pl.String


# ---------------------------------------------------------------------------
# clean_item_categories
# ---------------------------------------------------------------------------

class TestCleanItemCategories:
    """Tests for item_categories cleaning rules."""

    def _make_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "video_id": [100, 101, 102],
            "feat": ["[27, 9]", "[5]", "[12, 3, 7]"],
        })

    def test_row_count_preserved(self) -> None:
        df = self._make_df()
        result = clean_item_categories(df, [])
        assert result.height == df.height

    def test_feat_parsed_column_added(self) -> None:
        df = self._make_df()
        result = clean_item_categories(df, [])
        assert "feat_parsed" in result.columns

    def test_feat_parsed_is_list_int64(self) -> None:
        df = self._make_df()
        result = clean_item_categories(df, [])
        assert result["feat_parsed"].dtype == pl.List(pl.Int64)

    def test_feat_parsed_values_correct(self) -> None:
        df = self._make_df()
        result = clean_item_categories(df, [])
        assert result["feat_parsed"][0].to_list() == [27, 9]
        assert result["feat_parsed"][1].to_list() == [5]
        assert result["feat_parsed"][2].to_list() == [12, 3, 7]

    def test_original_feat_column_preserved(self) -> None:
        df = self._make_df()
        result = clean_item_categories(df, [])
        assert "feat" in result.columns


# ---------------------------------------------------------------------------
# clean_caption_category
# ---------------------------------------------------------------------------

class TestCleanCaptionCategory:
    """Tests for kuairec_caption_category cleaning rules (no-op)."""

    def _make_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "video_id": [100, 101, 102],
            "caption": ["text1", None, "text3"],
            "first_level_category_id": [17, 5, 10],
        })

    def test_row_count_preserved(self) -> None:
        df = self._make_df()
        result = clean_caption_category(df, [])
        assert result.height == df.height

    def test_nulls_preserved(self) -> None:
        df = self._make_df()
        result = clean_caption_category(df, [])
        assert result["caption"].null_count() == 1

    def test_log_entry_recorded(self) -> None:
        df = self._make_df()
        log: list = []
        clean_caption_category(df, log)
        assert len(log) == 1
        assert log[0]["rule"] == "no_changes"
