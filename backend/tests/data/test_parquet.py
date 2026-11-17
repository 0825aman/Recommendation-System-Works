"""Tests for Parquet round-trip conversion.

Uses synthetic CSV fixtures from conftest.py to test that the
write_parquet and verify_parquet functions produce correct outputs
without touching the real dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.data.clean_dataset import (
    clean_item_categories,
    clean_small_matrix,
    clean_user_features,
)
from scripts.data.common import PARQUET_FILES
from scripts.data.convert_to_parquet import verify_parquet, write_parquet

# ---------------------------------------------------------------------------
# write_parquet
# ---------------------------------------------------------------------------

class TestWriteParquet:
    """Tests for write_parquet — write to temp dir, verify output dict."""

    def _make_simple_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "user_id": [0, 1, 2, 3, 4],
            "video_id": [100, 101, 102, 103, 104],
            "watch_ratio": [0.5, 1.0, 0.3, 0.8, 1.2],
        })

    def test_write_returns_dict_with_expected_keys(self, tmp_path: Path) -> None:
        """write_parquet returns metadata dict with required keys."""
        # Temporarily redirect output to tmp_path
        name = "user_features"
        df = self._make_simple_df()
        out_path = tmp_path / "user_features.parquet"

        # Monkey-patch PARQUET_FILES for this test
        original = PARQUET_FILES.get(name)
        try:
            PARQUET_FILES[name] = out_path  # type: ignore[index]
            result = write_parquet(name, df)
        finally:
            if original is not None:
                PARQUET_FILES[name] = original  # type: ignore[index]

        assert "rows" in result
        assert "columns" in result
        assert "file_size_bytes" in result
        assert "dtypes" in result
        assert result["rows"] == 5
        assert result["columns"] == 3

    def test_parquet_file_is_readable(self, tmp_path: Path) -> None:
        """Written Parquet file can be read back by Polars."""
        name = "user_features"
        df = self._make_simple_df()
        out_path = tmp_path / "user_features.parquet"

        original = PARQUET_FILES.get(name)
        try:
            PARQUET_FILES[name] = out_path  # type: ignore[index]
            write_parquet(name, df)
        finally:
            if original is not None:
                PARQUET_FILES[name] = original  # type: ignore[index]

        read_back = pl.read_parquet(out_path)
        assert read_back.height == 5
        assert set(read_back.columns) == {"user_id", "video_id", "watch_ratio"}

    def test_row_count_preserved_in_round_trip(self, tmp_path: Path) -> None:
        """Row count is identical before and after write/read."""
        name = "item_categories"
        df = pl.DataFrame({
            "video_id": list(range(50)),
            "feat": [f"[{i}]" for i in range(50)],
        })
        out_path = tmp_path / "item_categories.parquet"

        original = PARQUET_FILES.get(name)
        try:
            PARQUET_FILES[name] = out_path  # type: ignore[index]
            write_parquet(name, df)
            read_back = pl.read_parquet(out_path)
        finally:
            if original is not None:
                PARQUET_FILES[name] = original  # type: ignore[index]

        assert read_back.height == 50


# ---------------------------------------------------------------------------
# verify_parquet
# ---------------------------------------------------------------------------

class TestVerifyParquet:
    """Tests for verify_parquet row-count verification."""

    def test_verify_passes_on_correct_row_count(self, tmp_path: Path) -> None:
        """verify_parquet returns 'pass' when row count matches."""
        name = "user_features"
        df = pl.DataFrame({"user_id": [0, 1, 2]})
        out_path = tmp_path / "user_features.parquet"
        df.write_parquet(out_path)

        original = PARQUET_FILES.get(name)
        try:
            PARQUET_FILES[name] = out_path  # type: ignore[index]
            result = verify_parquet(name, expected_rows=3)
        finally:
            if original is not None:
                PARQUET_FILES[name] = original  # type: ignore[index]

        assert result["status"] == "pass"
        assert result["actual_rows"] == 3

    def test_verify_fails_on_wrong_row_count(self, tmp_path: Path) -> None:
        """verify_parquet returns 'fail' when row count does not match."""
        name = "user_features"
        df = pl.DataFrame({"user_id": [0, 1, 2]})
        out_path = tmp_path / "user_features.parquet"
        df.write_parquet(out_path)

        original = PARQUET_FILES.get(name)
        try:
            PARQUET_FILES[name] = out_path  # type: ignore[index]
            result = verify_parquet(name, expected_rows=99)
        finally:
            if original is not None:
                PARQUET_FILES[name] = original  # type: ignore[index]

        assert result["status"] == "fail"


# ---------------------------------------------------------------------------
# Cleaning pipeline → Parquet schema integration
# ---------------------------------------------------------------------------

class TestCleanThenParquetSchema:
    """Test that cleaning produces schemas expected in final Parquet files."""

    def test_small_matrix_datetime_survives_parquet(self, tmp_path: Path) -> None:
        """Datetime column survives Parquet write/read round-trip."""
        df = pl.DataFrame({
            "user_id": [0, 1],
            "video_id": [100, 101],
            "play_duration": [5000, 10000],
            "video_duration": [10000, 10000],
            "time": ["2020-07-05 00:08:23.438", "2020-07-06 10:00:00.000"],
            "date": [20200705.0, 20200706.0],
            "timestamp": [1593878903.0, 1593979200.0],
            "watch_ratio": [0.5, 1.0],
        })
        cleaned = clean_small_matrix(df, [])

        out = tmp_path / "small_matrix.parquet"
        cleaned.write_parquet(out)
        read_back = pl.read_parquet(out)

        assert read_back["time"].dtype == pl.Datetime
        assert read_back["date"].dtype == pl.Int64

    def test_item_categories_list_col_survives_parquet(self, tmp_path: Path) -> None:
        """List[Int64] feat_parsed column survives Parquet round-trip."""
        df = pl.DataFrame({
            "video_id": [100, 101],
            "feat": ["[27, 9]", "[5]"],
        })
        cleaned = clean_item_categories(df, [])

        out = tmp_path / "item_categories.parquet"
        cleaned.write_parquet(out)
        read_back = pl.read_parquet(out)

        assert read_back["feat_parsed"].dtype == pl.List(pl.Int64)
        assert read_back["feat_parsed"][0].to_list() == [27, 9]

    def test_user_features_boolean_survives_parquet(self, tmp_path: Path) -> None:
        """Boolean columns survive Parquet write/read round-trip."""
        df = pl.DataFrame({
            "user_id": [0, 1],
            "user_active_degree": ["high_active", "UNKNOWN"],
            "is_lowactive_period": [0, 1],
            "is_live_streamer": [1, 0],
            "is_video_author": [0, 0],
        })
        cleaned = clean_user_features(df, [])

        out = tmp_path / "user_features.parquet"
        cleaned.write_parquet(out)
        read_back = pl.read_parquet(out)

        assert read_back["is_lowactive_period"].dtype == pl.Boolean
        assert read_back["is_live_streamer"].dtype == pl.Boolean
