"""Shared constants, paths, and utilities for data engineering scripts.

Import this module in all data scripts to ensure consistent path resolution
regardless of which directory the script is run from.

Usage:
    from scripts.data.common import RAW_DIR, PROCESSED_DIR, get_logger

Or run directly to verify paths:
    python scripts/data/common.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
# This file is at: <project_root>/scripts/data/common.py
# Project root is two levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Data directory structure
# ---------------------------------------------------------------------------
DATA_ROOT = PROJECT_ROOT / "data"

# Raw — immutable original downloads, never modified after download
RAW_DIR = DATA_ROOT / "raw" / "kuairec" / "original_downloads"

# Interim — inspection and validation outputs
INTERIM_DIR = DATA_ROOT / "interim" / "kuairec"
INSPECTED_DIR = INTERIM_DIR / "inspected"
VALIDATED_DIR = INTERIM_DIR / "validated"
INTERMEDIATE_DIR = INTERIM_DIR / "intermediate"

# Processed — final Parquet files and reports
PROCESSED_DIR = DATA_ROOT / "processed" / "kuairec"
PARQUET_DIR = PROCESSED_DIR / "parquet"
REPORTS_DIR = PROCESSED_DIR / "reports"

# Manifests and metadata
MANIFESTS_DIR = DATA_ROOT / "manifests"
METADATA_DIR = DATA_ROOT / "metadata" / "schema"
MANIFEST_PATH = MANIFESTS_DIR / "dataset_manifest.json"

# ---------------------------------------------------------------------------
# KuaiRec file paths — raw downloads
# ---------------------------------------------------------------------------
KUAIREC_ZIP = RAW_DIR / "KuaiRec.zip"

# Files extracted from the zip
RAW_FILES = {
    "big_matrix": RAW_DIR / "big_matrix.csv",
    "small_matrix": RAW_DIR / "small_matrix.csv",
    "social_network": RAW_DIR / "social_network.csv",
    "user_features": RAW_DIR / "user_features.csv",
    "item_categories": RAW_DIR / "item_categories.csv",
    "item_daily_features": RAW_DIR / "item_daily_features.csv",
    # Also included in the new zip
    "kuairec_caption_category": RAW_DIR / "kuairec_caption_category.csv",
}

# Files downloaded separately (not in older zip versions)
EXTRA_FILES = {
    "video_raw_categories_multi": RAW_DIR / "video_raw_categories_multi.csv",
    "user_features_raw": RAW_DIR / "user_features_raw.csv",
}

ALL_RAW_FILES = {**RAW_FILES, **EXTRA_FILES}

# ---------------------------------------------------------------------------
# Processed Parquet paths — Phase 2 scope (5 files)
# ---------------------------------------------------------------------------
PARQUET_FILES = {
    "small_matrix": PARQUET_DIR / "small_matrix.parquet",
    "user_features": PARQUET_DIR / "user_features.parquet",
    "user_features_raw": PARQUET_DIR / "user_features_raw.parquet",
    "item_categories": PARQUET_DIR / "item_categories.parquet",
    "kuairec_caption_category": PARQUET_DIR / "kuairec_caption_category.parquet",
}

# ---------------------------------------------------------------------------
# Zenodo download URLs and expected checksums (verified 2026-11-17)
# ---------------------------------------------------------------------------
ZENODO_RECORD = "https://zenodo.org/records/18164998"
ZENODO_DOI = "10.5281/zenodo.18164998"
DATASET_LICENSE = "CC BY 4.0"

DOWNLOAD_MANIFEST = [
    {
        "key": "KuaiRec.zip",
        "url": "https://zenodo.org/records/18164998/files/KuaiRec.zip",
        "size_bytes": 431_964_858,
        "md5": "261550d472c48eff4990fb13c0e5bcf7",
        "destination": str(KUAIREC_ZIP),
        "extract": True,
    },
    {
        "key": "kuairec_caption_category.csv",
        "url": "https://zenodo.org/records/18164998/files/kuairec_caption_category.csv",
        "size_bytes": 1_964_578,
        "md5": "31bc38cdccdf75a71df137779035f8cb",
        "destination": str(RAW_FILES["kuairec_caption_category"]),
        "extract": False,
    },
    {
        "key": "video_raw_categories_multi.csv",
        "url": "https://zenodo.org/records/18164998/files/video_raw_categories_multi.csv",
        "size_bytes": 1_724_451,
        "md5": "d05eea147135d2cdf7759fba5c0d70d4",
        "destination": str(EXTRA_FILES["video_raw_categories_multi"]),
        "extract": False,
    },
    {
        "key": "user_features_raw.csv",
        "url": "https://zenodo.org/records/18164998/files/user_features_raw.csv",
        "size_bytes": 1_541_558,
        "md5": "3969b8120035e7ced36d56926a7cbd24",
        "destination": str(EXTRA_FILES["user_features_raw"]),
        "extract": False,
    },
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a named logger with consistent formatting for data scripts.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def ensure_directories() -> None:
    """Create all required data directories if they do not already exist."""
    dirs = [
        RAW_DIR,
        INSPECTED_DIR,
        VALIDATED_DIR,
        INTERMEDIATE_DIR,
        PARQUET_DIR,
        REPORTS_DIR,
        MANIFESTS_DIR,
        METADATA_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def format_bytes(size: int) -> str:
    """Format a byte count as a human-readable string.

    Args:
        size: Size in bytes.

    Returns:
        Human-readable string like '411.8 MB'.
    """
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size //= 1024
    return f"{size:.1f} TB"


if __name__ == "__main__":
    logger = get_logger(__name__)
    ensure_directories()
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Data root:    %s", DATA_ROOT)
    logger.info("Raw dir:      %s", RAW_DIR)
    logger.info("Parquet dir:  %s", PARQUET_DIR)
    logger.info("Manifest:     %s", MANIFEST_PATH)
    logger.info("All directories verified.")
