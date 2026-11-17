"""Tests for temporal split logic (ml/data/splits.py).

Uses synthetic DataFrames — no real dataset required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ml.data.splits import create_temporal_split


class TestCreateTemporalSplit:
    """Tests for create_temporal_split."""

    def test_row_counts_sum_to_total(self, tiny_interactions: pl.DataFrame) -> None:
        result = create_temporal_split(tiny_interactions)
        total = tiny_interactions.height
        assert result.train.height + result.val.height + result.test.height == total

    def test_approximate_proportions(self, tiny_interactions: pl.DataFrame) -> None:
        result = create_temporal_split(tiny_interactions, train_ratio=0.70, val_ratio=0.15)
        total = tiny_interactions.height
        # Allow 5% slack for rounding
        assert abs(result.train.height / total - 0.70) < 0.07
        assert abs(result.val.height / total - 0.15) < 0.05

    def test_train_max_ts_less_than_val_min_ts(self, tiny_interactions: pl.DataFrame) -> None:
        result = create_temporal_split(tiny_interactions)
        train_max = result.train.filter(pl.col("timestamp").is_not_null())["timestamp"].max()
        val_min = result.val["timestamp"].min()
        assert train_max is not None
        assert val_min is not None
        assert train_max <= val_min

    def test_val_max_ts_less_than_test_min_ts(self, tiny_interactions: pl.DataFrame) -> None:
        result = create_temporal_split(tiny_interactions)
        val_max = result.val["timestamp"].max()
        test_min = result.test["timestamp"].min()
        assert val_max is not None
        assert test_min is not None
        assert val_max <= test_min

    def test_null_timestamp_rows_go_to_train(self, tiny_with_nulls: pl.DataFrame) -> None:
        result = create_temporal_split(tiny_with_nulls)
        # Val and test should have no null timestamps
        assert result.val.filter(pl.col("timestamp").is_null()).height == 0
        assert result.test.filter(pl.col("timestamp").is_null()).height == 0
        # Train should have exactly 3 null-ts rows
        assert result.train.filter(pl.col("timestamp").is_null()).height == 3

    def test_manifest_has_required_keys(self, tiny_interactions: pl.DataFrame) -> None:
        result = create_temporal_split(tiny_interactions)
        required = {
            "strategy", "train_rows", "val_rows", "test_rows",
            "null_timestamp_rows", "train_ratio", "val_ratio",
            "relevance_threshold",
        }
        assert required.issubset(result.manifest.keys())

    def test_manifest_row_counts_match_dataframes(self, tiny_interactions: pl.DataFrame) -> None:
        result = create_temporal_split(tiny_interactions)
        assert result.manifest["train_rows"] == result.train.height
        assert result.manifest["val_rows"] == result.val.height
        assert result.manifest["test_rows"] == result.test.height

    def test_invalid_ratios_raise(self, tiny_interactions: pl.DataFrame) -> None:
        with pytest.raises(ValueError):
            create_temporal_split(tiny_interactions, train_ratio=0.90, val_ratio=0.15)

    def test_negative_ratio_raises(self, tiny_interactions: pl.DataFrame) -> None:
        with pytest.raises(ValueError):
            create_temporal_split(tiny_interactions, train_ratio=-0.1, val_ratio=0.15)

    def test_no_data_leakage_train_to_test(self, tiny_interactions: pl.DataFrame) -> None:
        """Test: no timestamp in test is earlier than the train split cutoff."""
        result = create_temporal_split(tiny_interactions)
        train_timed_max = (
            result.train
            .filter(pl.col("timestamp").is_not_null())["timestamp"]
            .max()
        )
        test_min = result.test["timestamp"].min()
        assert test_min is not None
        assert train_timed_max is not None
        assert test_min > train_timed_max

    def test_manifest_relevant_rows_match_threshold(self, tiny_interactions: pl.DataFrame) -> None:
        result = create_temporal_split(tiny_interactions)
        expected_train_relevant = result.train.filter(pl.col("watch_ratio") >= 0.5).height
        assert result.manifest["train_relevant_rows"] == expected_train_relevant
