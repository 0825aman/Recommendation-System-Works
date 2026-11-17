"""Temporal train/validation/test split for KuaiRec 2.0 small_matrix.

Strategy (ADR-P3-04):
- Sort interactions by timestamp (Float64 unix seconds).
- Null-timestamp rows are assigned to the training split.
- Split proportions: 70% train / 15% validation / 15% test by time.
- Output: three Parquet files + a JSON manifest in data/processed/kuairec/splits/.

Usage (from project root):
    backend\\.venv\\Scripts\\python.exe scripts/ml/prepare_splits.py

Or import directly:
    from ml.data.splits import create_temporal_split
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import polars as pl

# Project root resolution
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SPLITS_DIR = _PROJECT_ROOT / "data" / "processed" / "kuairec" / "splits"


class SplitResult(NamedTuple):
    """Container for the three split DataFrames and their manifest."""

    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame
    manifest: dict  # type: ignore[type-arg]


def create_temporal_split(
    df: pl.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> SplitResult:
    """Split an interaction DataFrame into train/val/test using temporal ordering.

    Null-timestamp rows are separated first and appended to the training split.
    The remaining (non-null timestamp) rows are sorted by timestamp and split
    at the computed percentile cutoffs.

    Args:
        df:          Interaction DataFrame. Must contain 'timestamp' (Float64)
                     and 'user_id', 'video_id', 'watch_ratio' columns.
        train_ratio: Fraction of non-null rows assigned to train (default 0.70).
        val_ratio:   Fraction of non-null rows assigned to val (default 0.15).
                     test_ratio = 1 - train_ratio - val_ratio.

    Returns:
        SplitResult(train, val, test, manifest).

    Raises:
        ValueError: If ratios sum to > 1.0 or are negative.
    """
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must be positive.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0 to leave room for test.")

    test_ratio = 1.0 - train_ratio - val_ratio

    # Separate null-timestamp rows (assigned to train per ADR-P3-04)
    null_mask = df["timestamp"].is_null()
    null_rows = df.filter(null_mask)
    timed_rows = df.filter(~null_mask)

    null_count = null_rows.height
    timed_count = timed_rows.height
    total = df.height

    # Sort timed rows by timestamp ascending
    timed_sorted = timed_rows.sort("timestamp", descending=False)

    # Compute cutoff indices
    train_end = math.floor(timed_count * train_ratio)
    val_end = train_end + math.floor(timed_count * val_ratio)

    timed_train = timed_sorted.slice(0, train_end)
    timed_val = timed_sorted.slice(train_end, val_end - train_end)
    timed_test = timed_sorted.slice(val_end)

    # Append null rows to training
    train = pl.concat([timed_train, null_rows], rechunk=True)
    val = timed_val
    test = timed_test

    # Cutoff timestamps for reproducibility
    train_ts_max = float(timed_train["timestamp"].max())  # type: ignore[arg-type]
    val_ts_max = float(timed_val["timestamp"].max()) if timed_val.height > 0 else float("nan")
    test_ts_max = float(timed_test["timestamp"].max()) if timed_test.height > 0 else float("nan")

    manifest: dict = {  # type: ignore[type-arg]
        "created_at": datetime.now(UTC).isoformat(),
        "strategy": "temporal",
        "adr": "ADR-P3-04",
        "total_rows": total,
        "null_timestamp_rows": null_count,
        "null_rows_assigned_to": "train",
        "timed_rows": timed_count,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": round(test_ratio, 4),
        "train_rows": train.height,
        "val_rows": val.height,
        "test_rows": test.height,
        "train_timed_rows": timed_train.height,
        "train_max_timestamp": train_ts_max,
        "val_max_timestamp": val_ts_max,
        "test_max_timestamp": test_ts_max,
        "relevance_threshold": 0.5,
        "train_relevant_rows": int(train.filter(pl.col("watch_ratio") >= 0.5).height),
        "val_relevant_rows": int(val.filter(pl.col("watch_ratio") >= 0.5).height),
        "test_relevant_rows": int(test.filter(pl.col("watch_ratio") >= 0.5).height),
    }

    return SplitResult(train=train, val=val, test=test, manifest=manifest)


def save_splits(result: SplitResult, splits_dir: Path | None = None) -> Path:
    """Write train/val/test Parquet files and split manifest to disk.

    Args:
        result:     SplitResult from create_temporal_split().
        splits_dir: Output directory. Defaults to data/processed/kuairec/splits/.

    Returns:
        Path to the splits directory.
    """
    out_dir = splits_dir or _SPLITS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    result.train.write_parquet(out_dir / "train.parquet", compression="zstd")
    result.val.write_parquet(out_dir / "val.parquet", compression="zstd")
    result.test.write_parquet(out_dir / "test.parquet", compression="zstd")

    manifest_path = out_dir / "split_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(result.manifest, f, indent=2, default=str)

    return out_dir


def load_splits(splits_dir: Path | None = None) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load saved train/val/test Parquet files from disk.

    Args:
        splits_dir: Directory containing split files. Defaults to
                    data/processed/kuairec/splits/.

    Returns:
        Tuple of (train, val, test) DataFrames.

    Raises:
        FileNotFoundError: If split files have not been created yet.
    """
    out_dir = splits_dir or _SPLITS_DIR
    for name in ("train.parquet", "val.parquet", "test.parquet"):
        p = out_dir / name
        if not p.exists():
            raise FileNotFoundError(
                f"Split file not found: {p}. "
                "Run scripts/ml/prepare_splits.py first."
            )
    return (
        pl.read_parquet(out_dir / "train.parquet"),
        pl.read_parquet(out_dir / "val.parquet"),
        pl.read_parquet(out_dir / "test.parquet"),
    )
