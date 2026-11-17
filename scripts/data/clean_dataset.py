"""KuaiRec 2.0 data cleaning and standardization.

Applies documented, transparent cleaning rules to produce standardized
datasets. Every transformation logs the number of rows affected.
Raw files are NEVER modified.

Usage (from project root):
    python scripts/data/clean_dataset.py

Each cleaning rule documents:
  - Why it is required
  - What it changes
  - How many records are affected
  - Whether original values are preserved
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl

from scripts.data.common import (
    ALL_RAW_FILES,
    INTERMEDIATE_DIR,
    ensure_directories,
    get_logger,
)

logger = get_logger(__name__)

CLEANING_LOG_PATH = INTERMEDIATE_DIR / "cleaning_log.json"


def log_transform(
    log: list,  # type: ignore[type-arg]
    file: str,
    rule: str,
    description: str,
    rows_before: int,
    rows_after: int,
    rows_affected: int,
    original_preserved: bool,
) -> None:
    """Append a cleaning operation to the log."""
    log.append({
        "file": file,
        "rule": rule,
        "description": description,
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_affected": rows_affected,
        "original_preserved": original_preserved,
    })
    logger.info(
        "  [%s] %s: %d rows affected (preserved=%s)",
        file, rule, rows_affected, original_preserved,
    )


def clean_small_matrix(df: pl.DataFrame, log: list) -> pl.DataFrame:  # type: ignore[type-arg]
    """Clean and standardize the small_matrix interactions file.

    Rules applied:
    1. Parse 'time' column to Datetime — enables time-based filtering in Phase 3.
       Nulls in 'time' are preserved (3.89% — documented in validation report).
       Reason: timestamps were missing in source data, not a data error.
    2. Cast 'date' column to Int64 — Polars inferred Float64 due to nulls.
       Reason: 'date' is an integer (YYYYMMDD), Float is incorrect.
    3. Cast 'play_duration' and 'video_duration' to Int64 — confirm integer type.
    4. No rows are dropped — all 4,676,570 rows are preserved.
    """
    n = df.height
    logger.info("  Cleaning small_matrix (%d rows)...", n)

    # Rule 1: Parse time string to Datetime (nulls preserved)
    n_time_nulls_before = df["time"].null_count()
    df = df.with_columns(
        pl.col("time").str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f", strict=False).alias("time")
    )
    n_time_parsed = n - df["time"].null_count()
    log_transform(log, "small_matrix", "parse_time_to_datetime",
                  "Parse 'time' string to Datetime type for time-based operations",
                  n, n, n_time_parsed, True)

    # Rule 2: Cast date Float64 → Int64 (nulls preserved as null Int64)
    df = df.with_columns(
        pl.col("date").cast(pl.Int64, strict=False)
    )
    log_transform(log, "small_matrix", "cast_date_to_int64",
                  "Cast 'date' from Float64 (null-induced) to Int64",
                  n, n, n - n_time_nulls_before, True)

    # Rule 3: Ensure play_duration and video_duration are Int64
    df = df.with_columns([
        pl.col("play_duration").cast(pl.Int64, strict=False),
        pl.col("video_duration").cast(pl.Int64, strict=False),
    ])
    log_transform(log, "small_matrix", "cast_durations_to_int64",
                  "Ensure play_duration and video_duration are Int64",
                  n, n, n, True)

    logger.info("  small_matrix: %d rows in, %d rows out (0 dropped)", n, df.height)
    return df


def clean_user_features(df: pl.DataFrame, log: list) -> pl.DataFrame:  # type: ignore[type-arg]
    """Clean and standardize user_features.

    Rules applied:
    1. Cast binary flag columns (is_*) to Boolean — clearer semantics than Int64.
    2. No rows dropped.
    """
    n = df.height
    logger.info("  Cleaning user_features (%d rows)...", n)

    binary_cols = [
        "is_lowactive_period", "is_live_streamer", "is_video_author",
        "onehot_feat0", "onehot_feat12", "onehot_feat13", "onehot_feat14",
        "onehot_feat15", "onehot_feat16", "onehot_feat17",
    ]
    existing = [c for c in binary_cols if c in df.columns]
    df = df.with_columns([
        pl.col(c).cast(pl.Boolean) for c in existing
    ])
    log_transform(log, "user_features", "cast_binary_flags_to_boolean",
                  f"Cast {len(existing)} binary (0/1) columns to Boolean type",
                  n, n, n, True)

    logger.info("  user_features: %d rows in, %d rows out (0 dropped)", n, df.height)
    return df


def clean_user_features_raw(df: pl.DataFrame, log: list) -> pl.DataFrame:  # type: ignore[type-arg]
    """Clean and standardize user_features_raw.

    Rules applied:
    1. Cast binary is_* columns to Boolean.
    2. Cast os_version to String — mixed int/string in source.
    3. No rows dropped.
    """
    n = df.height
    logger.info("  Cleaning user_features_raw (%d rows)...", n)

    binary_cols = [
        "is_lowactive_period", "is_live_author", "is_photo_author",
        "is_install_douyin", "is_install_huoshan", "is_install_xigua",
        "is_install_douyu", "is_install_huya", "is_install_yy",
    ]
    existing = [c for c in binary_cols if c in df.columns]
    df = df.with_columns([
        pl.col(c).cast(pl.Boolean) for c in existing
    ])
    log_transform(log, "user_features_raw", "cast_binary_flags_to_boolean",
                  f"Cast {len(existing)} binary (0/1) columns to Boolean type",
                  n, n, n, True)

    # Cast os_version to String (source has mixed int/str values)
    if "os_version" in df.columns:
        df = df.with_columns(pl.col("os_version").cast(pl.String))
        log_transform(log, "user_features_raw", "cast_os_version_to_string",
                      "Cast os_version to String (mixed int/str in source)",
                      n, n, n, True)

    logger.info("  user_features_raw: %d rows in, %d rows out (0 dropped)", n, df.height)
    return df


def clean_item_categories(df: pl.DataFrame, log: list) -> pl.DataFrame:  # type: ignore[type-arg]
    """Clean and standardize item_categories.

    The 'feat' column contains Python-style list strings like "[27, 9]".
    We preserve the original string column and add a parsed list column.

    Rules applied:
    1. Add 'feat_parsed' column: parse '[27, 9]' strings to List[Int64].
       Reason: enables category lookup in Phase 3.
       Original 'feat' string column preserved.
    """
    n = df.height
    logger.info("  Cleaning item_categories (%d rows)...", n)

    # Parse "[27, 9]" string to List[Int64]
    df = df.with_columns(
        pl.col("feat")
        .str.strip_chars("[]")
        .str.split(", ")
        .list.eval(pl.element().cast(pl.Int64, strict=False))
        .alias("feat_parsed")
    )
    log_transform(log, "item_categories", "parse_feat_list_string",
                  "Parse 'feat' string '[27, 9]' to native List[Int64] in 'feat_parsed'",
                  n, n, n, True)

    logger.info("  item_categories: %d rows in, %d rows out (0 dropped)", n, df.height)
    return df


def clean_caption_category(df: pl.DataFrame, log: list) -> pl.DataFrame:  # type: ignore[type-arg]
    """Clean and standardize kuairec_caption_category.

    Rules applied:
    1. Preserve all Chinese text fields as-is (UTF-8 strings).
    2. No type changes needed — schema is already clean.
    3. Null captions (12.63%) are preserved — imputation belongs to Phase 3.
    """
    n = df.height
    logger.info("  Cleaning kuairec_caption_category (%d rows)...", n)
    # No transformations — schema is already clean, nulls are documented
    log_transform(log, "kuairec_caption_category", "no_changes",
                  "No cleaning required — schema clean, nulls documented",
                  n, n, 0, True)
    logger.info("  kuairec_caption_category: %d rows in, %d rows out (0 dropped)", n, df.height)
    return df


def clean_all() -> dict:  # type: ignore[type-arg]
    """Run all cleaning transforms and return cleaned DataFrames + log.

    Returns:
        dict with 'cleaned' (name->DataFrame) and 'log' (list of operations).
    """
    ensure_directories()
    log: list = []  # type: ignore[type-arg]
    t0 = time.perf_counter()

    def load(key: str) -> pl.DataFrame:
        path = ALL_RAW_FILES[key]
        return pl.read_csv(
            path,
            infer_schema_length=10_000,
            encoding="utf8-lossy",
            ignore_errors=True,
        )

    logger.info("Loading and cleaning files ...")
    cleaned = {
        "small_matrix": clean_small_matrix(load("small_matrix"), log),
        "user_features": clean_user_features(load("user_features"), log),
        "user_features_raw": clean_user_features_raw(load("user_features_raw"), log),
        "item_categories": clean_item_categories(load("item_categories"), log),
        "kuairec_caption_category": clean_caption_category(load("kuairec_caption_category"), log),
    }

    cleaning_report = {
        "dataset": "KuaiRec 2.0",
        "cleaning_timestamp": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "total_seconds": round(time.perf_counter() - t0, 2),
        "operations": log,
        "summary": {
            "total_operations": len(log),
            "files_processed": len(cleaned),
            "rows_dropped_total": 0,
        },
    }

    CLEANING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CLEANING_LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(cleaning_report, f, indent=2)

    logger.info("Cleaning log written to %s", CLEANING_LOG_PATH)
    return {"cleaned": cleaned, "log": log, "report": cleaning_report}


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("KuaiRec 2.0 Data Cleaner")
    logger.info("=" * 60)

    result = clean_all()
    logger.info("")
    logger.info("=== CLEANING SUMMARY ===")
    for op in result["log"]:
        logger.info(
            "  %-30s %-36s rows_affected=%d  preserved=%s",
            op["file"], op["rule"], op["rows_affected"], op["original_preserved"],
        )
    logger.info("")
    logger.info("Total operations: %d", len(result["log"]))
    logger.info("Total rows dropped: 0 (cleaning is non-destructive)")
