"""KuaiRec 2.0 data validation.

Validates referential integrity, duplicate detection, numerical range
checks, and cross-file consistency. Every check is explicit and produces
a pass/fail/warning result with counts — nothing is silent.

Usage (from project root):
    python scripts/data/validate_dataset.py

Output:
    data/interim/kuairec/validated/validation_report.json
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
    VALIDATED_DIR,
    ensure_directories,
    get_logger,
)

logger = get_logger(__name__)

VALIDATION_REPORT_PATH = VALIDATED_DIR / "validation_report.json"


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def check_duplicates(df: pl.DataFrame, key_cols: list[str], file_name: str) -> dict:  # type: ignore[type-arg]
    """Check for duplicate rows based on key columns."""
    total = df.height
    n_unique = df.select(key_cols).n_unique()
    n_dupes = total - n_unique
    status = "pass" if n_dupes == 0 else "warning"
    return {
        "check": "duplicate_keys",
        "file": file_name,
        "key_columns": key_cols,
        "total_rows": total,
        "unique_rows": n_unique,
        "duplicate_rows": n_dupes,
        "status": status,
        "note": f"{n_dupes} duplicate ({', '.join(key_cols)}) pairs found." if n_dupes else "No duplicates.",
    }


def check_non_negative(df: pl.DataFrame, col: str, file_name: str) -> dict:  # type: ignore[type-arg]
    """Check that a numeric column has no negative values."""
    if col not in df.columns:
        return {"check": f"non_negative_{col}", "file": file_name, "status": "skip", "note": "Column not present."}
    n_neg = df.filter(pl.col(col) < 0).height
    status = "pass" if n_neg == 0 else "fail"
    return {
        "check": f"non_negative_{col}",
        "file": file_name,
        "column": col,
        "negative_count": n_neg,
        "status": status,
        "note": f"{n_neg} negative values found in {col}." if n_neg else f"All {col} values are non-negative.",
    }


def check_null_rate(df: pl.DataFrame, col: str, file_name: str, threshold_pct: float = 50.0) -> dict:  # type: ignore[type-arg]
    """Flag if null rate in a column exceeds threshold."""
    if col not in df.columns:
        return {"check": f"null_rate_{col}", "file": file_name, "status": "skip", "note": "Column not present."}
    n_null = df[col].null_count()
    pct = round(n_null / max(df.height, 1) * 100, 2)
    status = "fail" if pct > threshold_pct else ("warning" if pct > 0 else "pass")
    return {
        "check": f"null_rate_{col}",
        "file": file_name,
        "column": col,
        "null_count": n_null,
        "null_pct": pct,
        "threshold_pct": threshold_pct,
        "status": status,
        "note": f"{pct}% nulls in {col}.",
    }


def check_referential_integrity(
    child_df: pl.DataFrame,
    child_col: str,
    parent_df: pl.DataFrame,
    parent_col: str,
    child_name: str,
    parent_name: str,
) -> dict:  # type: ignore[type-arg]
    """Check that all values in child_col exist in parent_col."""
    child_ids = child_df[child_col].unique()
    parent_ids = parent_df[parent_col].unique()
    parent_set = set(parent_ids.to_list())
    orphans = child_ids.filter(~child_ids.is_in(list(parent_set)))
    n_orphans = len(orphans)
    status = "pass" if n_orphans == 0 else "warning"
    return {
        "check": "referential_integrity",
        "child_file": child_name,
        "child_column": child_col,
        "parent_file": parent_name,
        "parent_column": parent_col,
        "child_unique_ids": len(child_ids),
        "parent_unique_ids": len(parent_ids),
        "orphan_ids_count": n_orphans,
        "orphan_ids_sample": orphans.head(10).to_list() if n_orphans > 0 else [],
        "status": status,
        "note": (
            f"{n_orphans} {child_col} values in {child_name} not found in {parent_name}."
            if n_orphans else
            f"All {child_col} values in {child_name} exist in {parent_name}."
        ),
    }


def check_watch_ratio_range(df: pl.DataFrame, file_name: str) -> dict:  # type: ignore[type-arg]
    """Analyze watch_ratio distribution — values > 1.0 are valid (replays)."""
    col = "watch_ratio"
    if col not in df.columns:
        return {"check": "watch_ratio_range", "file": file_name, "status": "skip"}

    valid = df.filter(pl.col(col).is_not_null())
    n_zero = valid.filter(pl.col(col) == 0.0).height
    n_above_one = valid.filter(pl.col(col) > 1.0).height
    n_above_five = valid.filter(pl.col(col) > 5.0).height
    n_negative = valid.filter(pl.col(col) < 0.0).height
    min_val = float(valid[col].min())  # type: ignore[arg-type]
    max_val = float(valid[col].max())  # type: ignore[arg-type]
    mean_val = round(float(valid[col].mean()), 4)  # type: ignore[arg-type]

    return {
        "check": "watch_ratio_range",
        "file": file_name,
        "total_non_null": valid.height,
        "n_zero": n_zero,
        "n_above_one": n_above_one,
        "n_above_five": n_above_five,
        "n_negative": n_negative,
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "status": "fail" if n_negative > 0 else "pass",
        "note": (
            f"watch_ratio range [{min_val:.4f}, {max_val:.4f}], mean={mean_val:.4f}. "
            f"{n_above_one} values > 1.0 (valid replays). "
            f"{n_above_five} values > 5.0 (possible anomalies, preserved). "
            f"{n_negative} negative values (invalid)."
        ),
    }


def check_removed_video_id(df: pl.DataFrame, file_name: str) -> dict:  # type: ignore[type-arg]
    """Verify video_id=1225 was removed in KuaiRec v2.0."""
    if "video_id" not in df.columns:
        return {"check": "removed_video_1225", "file": file_name, "status": "skip"}
    n_found = df.filter(pl.col("video_id") == 1225).height
    status = "fail" if n_found > 0 else "pass"
    return {
        "check": "removed_video_1225",
        "file": file_name,
        "count_video_1225": n_found,
        "status": status,
        "note": (
            f"video_id=1225 found {n_found} times — dataset may not be v2.0."
            if n_found else
            "video_id=1225 correctly absent (v2.0 confirmed)."
        ),
    }


# ---------------------------------------------------------------------------
# Main validation routine
# ---------------------------------------------------------------------------

def validate_all() -> dict:  # type: ignore[type-arg]
    """Run all validation checks on the Phase 2 scope files.

    Returns:
        Validation report dictionary.
    """
    ensure_directories()

    report: dict = {  # type: ignore[type-arg]
        "dataset": "KuaiRec 2.0",
        "validation_timestamp": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "checks": [],
        "summary": {},
    }

    t0 = time.perf_counter()

    # -----------------------------------------------------------------------
    # Load Phase 2 scope files (fully into memory — all small enough)
    # -----------------------------------------------------------------------
    logger.info("Loading files for validation...")

    def load(key: str) -> pl.DataFrame:
        path = ALL_RAW_FILES[key]
        logger.info("  Loading %s ...", key)
        return pl.read_csv(
            path,
            infer_schema_length=10_000,
            encoding="utf8-lossy",
            ignore_errors=True,
        )

    small_matrix = load("small_matrix")
    user_features = load("user_features")
    item_categories = load("item_categories")
    caption_category = load("kuairec_caption_category")
    load("video_raw_categories_multi")  # loaded but not validated in Phase 2 scope
    user_features_raw = load("user_features_raw")

    checks = report["checks"]

    # -----------------------------------------------------------------------
    # small_matrix checks
    # -----------------------------------------------------------------------
    logger.info("Validating small_matrix ...")
    checks.append(check_duplicates(small_matrix, ["user_id", "video_id"], "small_matrix"))
    checks.append(check_non_negative(small_matrix, "play_duration", "small_matrix"))
    checks.append(check_non_negative(small_matrix, "video_duration", "small_matrix"))
    checks.append(check_non_negative(small_matrix, "watch_ratio", "small_matrix"))
    checks.append(check_null_rate(small_matrix, "time", "small_matrix", threshold_pct=10.0))
    checks.append(check_null_rate(small_matrix, "timestamp", "small_matrix", threshold_pct=10.0))
    checks.append(check_watch_ratio_range(small_matrix, "small_matrix"))
    checks.append(check_removed_video_id(small_matrix, "small_matrix"))

    # -----------------------------------------------------------------------
    # user_features checks
    # -----------------------------------------------------------------------
    logger.info("Validating user_features ...")
    checks.append(check_duplicates(user_features, ["user_id"], "user_features"))
    checks.append(check_null_rate(user_features, "user_active_degree", "user_features"))

    # -----------------------------------------------------------------------
    # item_categories checks
    # -----------------------------------------------------------------------
    logger.info("Validating item_categories ...")
    checks.append(check_duplicates(item_categories, ["video_id"], "item_categories"))
    checks.append(check_removed_video_id(item_categories, "item_categories"))

    # -----------------------------------------------------------------------
    # kuairec_caption_category checks
    # -----------------------------------------------------------------------
    logger.info("Validating kuairec_caption_category ...")
    checks.append(check_duplicates(caption_category, ["video_id"], "kuairec_caption_category"))
    checks.append(check_null_rate(caption_category, "caption", "kuairec_caption_category"))
    checks.append(check_null_rate(caption_category, "manual_cover_text", "kuairec_caption_category"))

    # -----------------------------------------------------------------------
    # user_features_raw checks
    # -----------------------------------------------------------------------
    logger.info("Validating user_features_raw ...")
    checks.append(check_duplicates(user_features_raw, ["user_id"], "user_features_raw"))

    # -----------------------------------------------------------------------
    # Referential integrity: small_matrix → user_features
    # -----------------------------------------------------------------------
    logger.info("Checking referential integrity ...")
    checks.append(check_referential_integrity(
        small_matrix, "user_id",
        user_features, "user_id",
        "small_matrix", "user_features",
    ))

    # -----------------------------------------------------------------------
    # Referential integrity: small_matrix → item_categories
    # -----------------------------------------------------------------------
    checks.append(check_referential_integrity(
        small_matrix, "video_id",
        item_categories, "video_id",
        "small_matrix", "item_categories",
    ))

    # -----------------------------------------------------------------------
    # Referential integrity: caption_category → item_categories
    # -----------------------------------------------------------------------
    checks.append(check_referential_integrity(
        caption_category, "video_id",
        item_categories, "video_id",
        "kuairec_caption_category", "item_categories",
    ))

    # -----------------------------------------------------------------------
    # Cross-file user count consistency
    # -----------------------------------------------------------------------
    users_in_matrix = small_matrix["user_id"].n_unique()
    users_in_features = user_features["user_id"].n_unique()
    users_in_raw = user_features_raw["user_id"].n_unique()
    checks.append({
        "check": "user_count_consistency",
        "small_matrix_unique_users": users_in_matrix,
        "user_features_rows": users_in_features,
        "user_features_raw_rows": users_in_raw,
        "status": "pass" if users_in_matrix <= users_in_features else "warning",
        "note": (
            f"small_matrix has {users_in_matrix} unique users, "
            f"user_features has {users_in_features} rows, "
            f"user_features_raw has {users_in_raw} rows."
        ),
    })

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    status_counts = {"pass": 0, "warning": 0, "fail": 0, "skip": 0}
    for c in checks:
        status_counts[c.get("status", "skip")] += 1

    report["summary"] = {
        "total_checks": len(checks),
        **status_counts,
        "total_seconds": round(time.perf_counter() - t0, 2),
    }

    # Write report
    VALIDATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with VALIDATION_REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Validation report written to %s", VALIDATION_REPORT_PATH)
    return report


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("KuaiRec 2.0 Data Validator")
    logger.info("=" * 60)

    report = validate_all()

    logger.info("")
    logger.info("=== VALIDATION SUMMARY ===")
    s = report["summary"]
    logger.info(
        "  Total: %d  |  Pass: %d  |  Warning: %d  |  Fail: %d  |  Skip: %d",
        s["total_checks"], s["pass"], s["warning"], s["fail"], s["skip"],
    )
    logger.info("")
    for check in report["checks"]:
        icon = {"pass": "OK ", "warning": "WRN", "fail": "ERR", "skip": "SKP"}.get(check.get("status", "skip"), "???")
        logger.info("  [%s] %s", icon, check.get("note", check.get("check")))

    logger.info("")
    logger.info("Report: %s", VALIDATION_REPORT_PATH)
