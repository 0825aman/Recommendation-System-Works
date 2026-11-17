"""KuaiRec 2.0 dataset inspector.

Uses Polars lazy evaluation (scan_csv) to inspect all raw CSV files
without loading entire files into memory. Produces a structured
inspection report with schema, row counts, null rates, and samples.

Usage (from project root):
    python scripts/data/inspect_dataset.py

Output:
    data/interim/kuairec/inspected/inspection_report.json
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
    INSPECTED_DIR,
    ensure_directories,
    format_bytes,
    get_logger,
)

logger = get_logger(__name__)

INSPECTION_REPORT_PATH = INSPECTED_DIR / "inspection_report.json"

# Files to skip full inspection (too large for Phase 2 focus)
# They are still size-checked but not fully scanned.
LARGE_FILES_SKIP_SCAN = {"big_matrix", "item_daily_features", "social_network"}


def inspect_file(name: str, path: Path) -> dict:  # type: ignore[type-arg]
    """Inspect a single CSV file using Polars lazy evaluation.

    For large files in LARGE_FILES_SKIP_SCAN, only file size and
    header are read without scanning all rows.

    Args:
        name: Logical name of the file (key in ALL_RAW_FILES).
        path: Path to the CSV file.

    Returns:
        Dictionary with inspection results.
    """
    result: dict = {  # type: ignore[type-arg]
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "file_size_bytes": 0,
        "file_size_human": "0 B",
    }

    if not path.exists():
        logger.warning("File not found: %s", path)
        result["error"] = "file_not_found"
        return result

    result["file_size_bytes"] = path.stat().st_size
    result["file_size_human"] = format_bytes(path.stat().st_size)

    # For very large files, only scan headers and first few rows
    if name in LARGE_FILES_SKIP_SCAN:
        logger.info("  [partial scan] %s (%s)", name, result["file_size_human"])
        try:
            # Read just the first 1000 rows to get schema
            sample = pl.read_csv(
                path,
                n_rows=1000,
                infer_schema_length=1000,
                encoding="utf8-lossy",
                ignore_errors=True,
            )
            result["scan_mode"] = "partial_1000_rows"
            result["columns"] = sample.width
            result["column_names"] = sample.columns
            result["dtypes"] = {col: str(dtype) for col, dtype in zip(sample.columns, sample.dtypes)}
            result["sample_rows"] = sample.head(3).to_dicts()
            result["note"] = "Large file — partial scan only. Full row count not computed in Phase 2."
        except Exception as exc:  # noqa: BLE001
            logger.error("  Error scanning %s: %s", name, exc)
            result["error"] = str(exc)
        return result

    # Full lazy scan for Phase 2 scope files
    logger.info("  [full scan] %s (%s)", name, result["file_size_human"])
    try:
        lf = pl.scan_csv(
            path,
            infer_schema_length=10_000,
            encoding="utf8-lossy",
            ignore_errors=True,
        )

        # Collect schema without loading data
        schema = lf.collect_schema()
        result["columns"] = len(schema)
        result["column_names"] = list(schema.names())
        result["dtypes"] = {name_: str(dtype) for name_, dtype in schema.items()}

        # Collect full frame for stats (lazy → eager)
        t0 = time.perf_counter()
        df = lf.collect()
        elapsed = time.perf_counter() - t0

        result["rows"] = df.height
        result["load_seconds"] = round(elapsed, 2)

        # Null counts per column
        null_counts = df.null_count()
        result["null_counts"] = {
            col: int(null_counts[col][0])
            for col in null_counts.columns
        }
        result["null_pct"] = {
            col: round(int(null_counts[col][0]) / max(df.height, 1) * 100, 2)
            for col in null_counts.columns
        }

        # Numeric column statistics
        numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes)
                        if dtype in (pl.Int32, pl.Int64, pl.Float32, pl.Float64,
                                     pl.UInt32, pl.UInt64)]
        if numeric_cols:
            stats = df.select(numeric_cols).describe()
            result["numeric_stats"] = stats.to_dicts()

        # Sample rows (first 3 as dicts)
        result["sample_rows"] = df.head(3).to_dicts()

        # String column cardinalities (for small files)
        str_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.String]
        if str_cols and df.height <= 500_000:
            result["string_cardinalities"] = {
                col: df[col].n_unique() for col in str_cols
            }

        logger.info(
            "    rows=%d, cols=%d, nulls=%d, %.1fs",
            df.height,
            df.width,
            sum(result["null_counts"].values()),
            elapsed,
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("  Error scanning %s: %s", name, exc)
        result["error"] = str(exc)

    return result


def inspect_all() -> dict:  # type: ignore[type-arg]
    """Inspect all KuaiRec 2.0 raw CSV files.

    Returns:
        Full inspection report dictionary.
    """
    ensure_directories()

    report: dict = {  # type: ignore[type-arg]
        "dataset": "KuaiRec 2.0",
        "inspection_timestamp": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "polars_version": pl.__version__,
        "files": {},
    }

    total_start = time.perf_counter()

    for name, path in ALL_RAW_FILES.items():
        logger.info("Inspecting: %s", name)
        file_result = inspect_file(name, path)
        report["files"][name] = file_result

    report["total_seconds"] = round(time.perf_counter() - total_start, 2)

    # Summary
    present = sum(1 for f in report["files"].values() if f.get("exists", False))
    report["summary"] = {
        "total_files": len(ALL_RAW_FILES),
        "files_present": present,
        "files_missing": len(ALL_RAW_FILES) - present,
    }

    # Write report
    INSPECTION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INSPECTION_REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info("Inspection report written to %s", INSPECTION_REPORT_PATH)
    return report


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("KuaiRec 2.0 Dataset Inspector")
    logger.info("=" * 60)

    report = inspect_all()

    logger.info("")
    logger.info("=== INSPECTION SUMMARY ===")
    for name, info in report["files"].items():
        if not info.get("exists"):
            logger.warning("  MISSING: %s", name)
            continue
        rows = info.get("rows", "N/A (partial)")
        cols = info.get("columns", "?")
        size = info.get("file_size_human", "?")
        total_nulls = sum(info.get("null_counts", {}).values()) if "null_counts" in info else "N/A"
        logger.info(
            "  %-38s  rows=%-12s  cols=%-3s  size=%-10s  nulls=%s",
            name, rows, cols, size, total_nulls
        )

    logger.info("")
    logger.info("Total time: %.1fs", report["total_seconds"])
    logger.info("Report: %s", INSPECTION_REPORT_PATH)
