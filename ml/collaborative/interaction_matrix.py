"""User-item interaction matrix builder for Collaborative Filtering.

Converts a Polars training DataFrame into a dense numpy array of shape
(n_users, n_items) where each cell contains the user's watch_ratio
(clipped to max_watch_ratio to remove outliers like wr=571.5).

Index maps (user_id ↔ row_idx, video_id ↔ col_idx) are returned for
efficient O(1) lookup in both directions.

Usage:
    from ml.collaborative.interaction_matrix import build_interaction_matrix

    matrix, user_to_row, item_to_col, row_to_user, col_to_item = (
        build_interaction_matrix(train_df)
    )
"""

from __future__ import annotations

import numpy as np
import polars as pl


def build_interaction_matrix(
    train_df: pl.DataFrame,
    max_watch_ratio: float = 5.0,
) -> tuple[
    np.ndarray,
    dict[int, int],
    dict[int, int],
    dict[int, int],
    dict[int, int],
]:
    """Build a dense user-item interaction matrix from training interactions.

    Args:
        train_df:        Training interactions. Must contain columns:
                         user_id (Int64), video_id (Int64), watch_ratio (Float64).
        max_watch_ratio: Clip watch_ratio at this value to remove outliers.
                         The raw maximum in KuaiRec 2.0 is 571.5; clipping at 5.0
                         removes extreme replays while preserving signal.

    Returns:
        Tuple of:
          matrix       (np.ndarray, float32, shape n_users × n_items):
                       matrix[u, i] = clipped watch_ratio for user u, item i.
                       0.0 for unobserved pairs.
          user_to_row  (dict[int, int]): user_id → row index
          item_to_col  (dict[int, int]): video_id → column index
          row_to_user  (dict[int, int]): row index → user_id
          col_to_item  (dict[int, int]): column index → video_id
    """
    # Clip watch_ratio before building the matrix
    df = train_df.with_columns(
        pl.col("watch_ratio").clip(upper_bound=max_watch_ratio).alias("watch_ratio")
    )

    # Build sorted, stable index maps
    user_ids: list[int] = sorted(df["user_id"].unique().to_list())
    item_ids: list[int] = sorted(df["video_id"].unique().to_list())

    user_to_row: dict[int, int] = {uid: i for i, uid in enumerate(user_ids)}
    item_to_col: dict[int, int] = {iid: j for j, iid in enumerate(item_ids)}
    row_to_user: dict[int, int] = {i: uid for uid, i in user_to_row.items()}
    col_to_item: dict[int, int] = {j: iid for iid, j in item_to_col.items()}

    n_users = len(user_ids)
    n_items = len(item_ids)

    matrix = np.zeros((n_users, n_items), dtype=np.float32)

    # Fill using vectorised Polars operations for speed
    users_col = df["user_id"].to_list()
    items_col = df["video_id"].to_list()
    ratios_col = df["watch_ratio"].to_list()

    for uid, iid, wr in zip(users_col, items_col, ratios_col):
        if uid in user_to_row and iid in item_to_col:
            matrix[user_to_row[uid], item_to_col[iid]] = float(wr)

    return matrix, user_to_row, item_to_col, row_to_user, col_to_item
