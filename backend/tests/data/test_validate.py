"""Tests for the data validation pipeline.

Uses synthetic fixtures — no real dataset files required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.data.validate_dataset import (
    check_duplicates,
    check_non_negative,
    check_null_rate,
    check_referential_integrity,
    check_removed_video_id,
    check_watch_ratio_range,
)


class TestCheckDuplicates:
    """Tests for duplicate detection."""

    def test_no_duplicates_passes(self) -> None:
        df = pl.DataFrame({"user_id": [1, 2, 3], "video_id": [10, 20, 30]})
        result = check_duplicates(df, ["user_id", "video_id"], "test")
        assert result["status"] == "pass"
        assert result["duplicate_rows"] == 0

    def test_duplicates_detected(self) -> None:
        df = pl.DataFrame({"user_id": [1, 1, 2], "video_id": [10, 10, 20]})
        result = check_duplicates(df, ["user_id", "video_id"], "test")
        assert result["status"] == "warning"
        assert result["duplicate_rows"] == 1


class TestCheckNonNegative:
    """Tests for non-negative value check."""

    def test_all_positive_passes(self) -> None:
        df = pl.DataFrame({"play_duration": [100, 200, 0, 500]})
        result = check_non_negative(df, "play_duration", "test")
        assert result["status"] == "pass"
        assert result["negative_count"] == 0

    def test_negative_detected(self) -> None:
        df = pl.DataFrame({"play_duration": [100, -50, 200]})
        result = check_non_negative(df, "play_duration", "test")
        assert result["status"] == "fail"
        assert result["negative_count"] == 1

    def test_missing_column_skipped(self) -> None:
        df = pl.DataFrame({"other_col": [1, 2, 3]})
        result = check_non_negative(df, "play_duration", "test")
        assert result["status"] == "skip"


class TestCheckNullRate:
    """Tests for null rate check."""

    def test_no_nulls_passes(self) -> None:
        df = pl.DataFrame({"time": ["2020-01-01", "2020-01-02"]})
        result = check_null_rate(df, "time", "test")
        assert result["status"] == "pass"
        assert result["null_pct"] == 0.0

    def test_null_rate_warning(self) -> None:
        # 1 null in 4 rows = 25% — below the 50% threshold → "warning"
        df = pl.DataFrame({"time": ["2020-01-01", "2020-01-02", "2020-01-03", None]})
        result = check_null_rate(df, "time", "test", threshold_pct=50.0)
        assert result["status"] == "warning"
        assert result["null_pct"] == 25.0

    def test_null_rate_above_threshold_fails(self) -> None:
        df = pl.DataFrame({"time": [None, None, None, None]})
        result = check_null_rate(df, "time", "test", threshold_pct=50.0)
        assert result["status"] == "fail"


class TestCheckReferentialIntegrity:
    """Tests for referential integrity checks."""

    def test_all_ids_present_passes(self) -> None:
        child = pl.DataFrame({"user_id": [1, 2, 3]})
        parent = pl.DataFrame({"user_id": [1, 2, 3, 4, 5]})
        result = check_referential_integrity(child, "user_id", parent, "user_id", "child", "parent")
        assert result["status"] == "pass"
        assert result["orphan_ids_count"] == 0

    def test_orphan_ids_detected(self) -> None:
        child = pl.DataFrame({"user_id": [1, 2, 99]})
        parent = pl.DataFrame({"user_id": [1, 2, 3]})
        result = check_referential_integrity(child, "user_id", parent, "user_id", "child", "parent")
        assert result["status"] == "warning"
        assert result["orphan_ids_count"] == 1
        assert 99 in result["orphan_ids_sample"]


class TestCheckWatchRatioRange:
    """Tests for watch_ratio range analysis."""

    def test_valid_ratios_pass(self) -> None:
        df = pl.DataFrame({"watch_ratio": [0.5, 1.0, 1.5, 2.0]})
        result = check_watch_ratio_range(df, "test")
        assert result["status"] == "pass"
        assert result["n_negative"] == 0
        assert result["n_above_one"] == 2

    def test_negative_ratio_fails(self) -> None:
        df = pl.DataFrame({"watch_ratio": [0.5, -0.1, 1.0]})
        result = check_watch_ratio_range(df, "test")
        assert result["status"] == "fail"
        assert result["n_negative"] == 1

    def test_missing_column_skipped(self) -> None:
        df = pl.DataFrame({"other": [1.0, 2.0]})
        result = check_watch_ratio_range(df, "test")
        assert result["status"] == "skip"


class TestCheckRemovedVideo1225:
    """Tests for verification that video_id=1225 was removed in v2.0."""

    def test_absent_passes(self) -> None:
        df = pl.DataFrame({"video_id": [100, 200, 300]})
        result = check_removed_video_id(df, "test")
        assert result["status"] == "pass"

    def test_present_fails(self) -> None:
        df = pl.DataFrame({"video_id": [100, 1225, 300]})
        result = check_removed_video_id(df, "test")
        assert result["status"] == "fail"
        assert result["count_video_1225"] == 1
