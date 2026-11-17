"""Create and save temporal train/validation/test splits for small_matrix.

Usage (from project root):
    backend\\.venv\\Scripts\\python.exe scripts/ml/prepare_splits.py

Output:
    data/processed/kuairec/splits/train.parquet
    data/processed/kuairec/splits/val.parquet
    data/processed/kuairec/splits/test.parquet
    data/processed/kuairec/splits/split_manifest.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.data.loader import load_small_matrix
from ml.data.splits import create_temporal_split, save_splits
from scripts.data.common import get_logger

logger = get_logger(__name__)


def main() -> None:
    """Run temporal split and save output files."""
    logger.info("=" * 60)
    logger.info("KuaiRec 2.0 Temporal Split")
    logger.info("=" * 60)

    logger.info("Loading small_matrix ...")
    df = load_small_matrix()
    logger.info("  Loaded %d rows", df.height)

    logger.info("Creating temporal split (70/15/15) ...")
    result = create_temporal_split(df, train_ratio=0.70, val_ratio=0.15)

    m = result.manifest
    logger.info("  Total rows:          %d", m["total_rows"])
    logger.info("  Null-ts rows (to train): %d", m["null_timestamp_rows"])
    logger.info("  Train rows:          %d (relevant: %d)", m["train_rows"], m["train_relevant_rows"])
    logger.info("  Val rows:            %d (relevant: %d)", m["val_rows"], m["val_relevant_rows"])
    logger.info("  Test rows:           %d (relevant: %d)", m["test_rows"], m["test_relevant_rows"])
    logger.info("  Train max timestamp: %.0f", m["train_max_timestamp"])
    logger.info("  Val max timestamp:   %.0f", m["val_max_timestamp"])
    logger.info("  Test max timestamp:  %.0f", m["test_max_timestamp"])

    logger.info("Saving split files ...")
    out_dir = save_splits(result)
    logger.info("  Written to: %s", out_dir)
    logger.info("Split complete.")


if __name__ == "__main__":
    main()
