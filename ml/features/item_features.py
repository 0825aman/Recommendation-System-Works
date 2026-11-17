"""Item feature extraction for category-based recommenders.

Loads item category information from the Phase 2 Parquet files
and provides lookup functions for category-aware algorithms.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PARQUET_DIR = _PROJECT_ROOT / "data" / "processed" / "kuairec" / "parquet"


def load_item_category_map(
    caption_path: Path | None = None,
) -> dict[int, str]:
    """Build a video_id → first_level_category_name mapping.

    Args:
        caption_path: Path to kuairec_caption_category.parquet.
                      Defaults to the standard Phase 2 Parquet location.

    Returns:
        Dict mapping video_id (int) to first-level category name (str).
        Items without a category entry are absent from the dict.
    """
    path = caption_path or (_PARQUET_DIR / "kuairec_caption_category.parquet")
    df = pl.read_parquet(path, columns=["video_id", "first_level_category_name"])
    return dict(zip(df["video_id"].to_list(), df["first_level_category_name"].to_list()))


def get_item_categories(
    video_ids: list[int],
    item_to_category: dict[int, str],
) -> dict[int, str]:
    """Filter an item_to_category mapping to a specific set of video_ids.

    Args:
        video_ids:        List of video IDs to look up.
        item_to_category: Full mapping.

    Returns:
        Filtered mapping containing only the requested video_ids.
    """
    return {v: item_to_category[v] for v in video_ids if v in item_to_category}
