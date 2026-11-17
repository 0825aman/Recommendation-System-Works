"""User feature extraction for user-preference heuristic recommender."""

from __future__ import annotations

import polars as pl


def compute_user_category_affinity(
    train_df: pl.DataFrame,
    item_to_category: dict[int, str],
) -> dict[int, dict[str, float]]:
    """Compute per-user category affinity scores from training interactions.

    Affinity = mean watch_ratio for items watched in that category.
    Users who engage deeply (high watch_ratio) with a category score higher
    than users who merely click through.

    Args:
        train_df:         Training interaction DataFrame.
        item_to_category: Mapping from video_id to category name.

    Returns:
        Dict: user_id → {category_name: mean_watch_ratio}.
    """
    df = train_df.with_columns(
        pl.col("video_id").replace_strict(item_to_category, default=None).alias("category")
    ).filter(pl.col("category").is_not_null())

    if df.height == 0:
        return {}

    agg = (
        df.group_by(["user_id", "category"])
        .agg(pl.col("watch_ratio").mean().alias("affinity"))
    )

    result: dict[int, dict[str, float]] = {}
    for row in agg.iter_rows(named=True):
        uid = row["user_id"]
        if uid not in result:
            result[uid] = {}
        result[uid][row["category"]] = float(row["affinity"])

    return result
