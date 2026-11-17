"""Parquet data loader for the ML module.

Provides lazy-loading access to the KuaiRec 2.0 Parquet files.
All paths resolve relative to the project root automatically.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

# Project root (ml/data/loader.py → ml/ → project root)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PARQUET_DIR = _PROJECT_ROOT / "data" / "processed" / "kuairec" / "parquet"


def load_small_matrix() -> pl.DataFrame:
    """Load the full small_matrix interaction file.

    Returns:
        DataFrame with columns: user_id, video_id, play_duration,
        video_duration, time, date, timestamp, watch_ratio.
    """
    return pl.read_parquet(_PARQUET_DIR / "small_matrix.parquet")


def load_user_features() -> pl.DataFrame:
    """Load user demographic and engagement features.

    Returns:
        DataFrame with 7,176 rows and 31 columns (user_id as Int64 PK).
    """
    return pl.read_parquet(_PARQUET_DIR / "user_features.parquet")


def load_item_categories() -> pl.DataFrame:
    """Load item category tags.

    Returns:
        DataFrame with columns: video_id, feat (String), feat_parsed (List[Int64]).
    """
    return pl.read_parquet(_PARQUET_DIR / "item_categories.parquet")


def load_caption_category() -> pl.DataFrame:
    """Load video captions and three-level category hierarchy.

    Returns:
        DataFrame with 10 columns including first/second/third level category names.
    """
    return pl.read_parquet(_PARQUET_DIR / "kuairec_caption_category.parquet")
