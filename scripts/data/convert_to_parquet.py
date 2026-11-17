"""KuaiRec 2.0 Parquet conversion.

Converts the Phase 2 scope files from CSV to Parquet using Polars,
after applying cleaning and standardization from clean_dataset.py.

Usage (from project root):
    python scripts/data/convert_to_parquet.py

Output:
    data/processed/kuairec/parquet/small_matrix.parquet
    data/processed/kuairec/parquet/user_features.parquet
    data/processed/kuairec/parquet/user_features_raw.parquet
    data/processed/kuairec/parquet/item_categories.parquet
    data/processed/kuairec/parquet/kuairec_caption_category.parquet

Compression: zstd (better ratio than snappy, fast decompression)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl

from scripts.data.clean_dataset import clean_all
from scripts.data.common import (
    PARQUET_DIR,
    PARQUET_FILES,
    ensure_directories,
    format_bytes,
    get_logger,
)

logger = get_logger(__name__)

PARQUET_MANIFEST_PATH = PARQUET_DIR / "parquet_manifest.json"


def write_parquet(name: str, df: pl.DataFrame) -> dict:  # type: ignore[type-arg]
    """Write a Polars DataFrame to Parquet with zstd compression.

    Args:
        name: Logical name (key in PARQUET_FILES).
        df: Cleaned DataFrame to write.

    Returns:
        dict with output path, row count, file size, and schema.
    """
    output_path = PARQUET_FILES[name]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("  Writing %s.parquet (%d rows, %d cols)...", name, df.height, df.width)
    t0 = time.perf_counter()

    df.write_parquet(output_path, compression="zstd")

    elapsed = time.perf_counter() - t0
    file_size = output_path.stat().st_size

    logger.info(
        "  -> %s: %s, %.2fs",
        output_path.name, format_bytes(file_size), elapsed,
    )

    return {
        "name": name,
        "path": str(output_path),
        "rows": df.height,
        "columns": df.width,
        "column_names": df.columns,
        "dtypes": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
        "file_size_bytes": file_size,
        "file_size_human": format_bytes(file_size),
        "compression": "zstd",
        "write_seconds": round(elapsed, 3),
    }


def verify_parquet(name: str, expected_rows: int) -> dict:  # type: ignore[type-arg]
    """Read back a Parquet file and verify row count.

    Args:
        name: Logical name of the file.
        expected_rows: Expected number of rows.

    Returns:
        dict with verification status.
    """
    path = PARQUET_FILES[name]
    df = pl.read_parquet(path)
    actual_rows = df.height
    status = "pass" if actual_rows == expected_rows else "fail"
    result = {
        "name": name,
        "expected_rows": expected_rows,
        "actual_rows": actual_rows,
        "status": status,
        "note": (
            f"Row count matches: {actual_rows}"
            if status == "pass"
            else f"Row count mismatch: expected {expected_rows}, got {actual_rows}"
        ),
    }
    icon = "OK " if status == "pass" else "ERR"
    logger.info("  [%s] %s: %d rows", icon, name, actual_rows)
    return result


def convert_all() -> dict:  # type: ignore[type-arg]
    """Clean all Phase 2 files and write to Parquet.

    Returns:
        Conversion manifest dictionary.
    """
    ensure_directories()
    t0 = time.perf_counter()

    # Run cleaning pipeline
    logger.info("Running cleaning pipeline ...")
    cleaning_result = clean_all()
    cleaned = cleaning_result["cleaned"]

    # Write Parquet files
    logger.info("Writing Parquet files ...")
    outputs = {}
    for name, df in cleaned.items():
        outputs[name] = write_parquet(name, df)

    # Verify round-trips
    logger.info("Verifying Parquet files ...")
    verifications = {}
    for name, info in outputs.items():
        verifications[name] = verify_parquet(name, info["rows"])

    manifest = {
        "dataset": "KuaiRec 2.0",
        "conversion_timestamp": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "compression": "zstd",
        "files": outputs,
        "verification": verifications,
        "total_seconds": round(time.perf_counter() - t0, 2),
        "summary": {
            "files_written": len(outputs),
            "all_verified": all(v["status"] == "pass" for v in verifications.values()),
        },
    }

    PARQUET_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PARQUET_MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info("Parquet manifest written to %s", PARQUET_MANIFEST_PATH)
    return manifest


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("KuaiRec 2.0 Parquet Converter")
    logger.info("=" * 60)

    manifest = convert_all()

    logger.info("")
    logger.info("=== PARQUET CONVERSION SUMMARY ===")
    for name, info in manifest["files"].items():
        logger.info(
            "  %-38s  rows=%-10d  size=%-10s  time=%.2fs",
            name + ".parquet",
            info["rows"],
            info["file_size_human"],
            info["write_seconds"],
        )
    logger.info("")
    all_ok = manifest["summary"]["all_verified"]
    logger.info("Verification: %s", "ALL PASS" if all_ok else "FAILURES DETECTED")
    logger.info("Total time: %.1fs", manifest["total_seconds"])

    if not all_ok:
        sys.exit(1)
