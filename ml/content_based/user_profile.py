"""User preference profile builder for Content-Based Filtering.

Constructs a feature-space representation of each user based on the items
they have interacted with in the training split.

The user profile is a weighted average of item feature vectors,
where the weight is the interaction's watch_ratio (optionally).
L2-normalisation is applied so the resulting vector can be compared
directly with normalised item vectors via dot product.

Data leakage prevention:
    Only train_df is ever accessed. Validation and test splits are NEVER
    used here. The Evaluator builds the seen-item sets from train_df too,
    so there is no leakage pathway.

Usage:
    from ml.content_based.user_profile import build_user_profiles
"""

from __future__ import annotations

import math

import polars as pl


def _weighted_average(
    vectors: list[list[float]],
    weights: list[float],
) -> list[float]:
    """Return the weighted average of a list of vectors.

    Args:
        vectors: List of feature vectors (all same length).
        weights: Per-vector weights (must be non-negative).

    Returns:
        Weighted average vector. Returns zero vector if all weights are 0.
    """
    if not vectors:
        return []

    dim = len(vectors[0])
    total_weight = sum(weights)

    if total_weight == 0.0:
        return [0.0] * dim

    result = [0.0] * dim
    for vec, w in zip(vectors, weights):
        for j in range(dim):
            result[j] += w * vec[j]

    return [x / total_weight for x in result]


def _l2_normalize(vec: list[float]) -> list[float]:
    """Return L2-normalised copy of a vector."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec[:]
    return [x / norm for x in vec]


def build_user_profiles(
    train_df: pl.DataFrame,
    item_feature_matrix: list[list[float]],
    id_to_idx: dict[int, int],
    weight_by_watch_ratio: bool = True,
    normalize: bool = True,
) -> dict[int, list[float]]:
    """Build a feature-space user profile for each user in train_df.

    For each user:
      1. Collect all interactions from train_df.
      2. Look up each item's feature vector.
      3. Compute weighted average (optionally weighted by watch_ratio).
      4. Optionally L2-normalise the resulting profile.

    Users whose interacted items are all missing from the feature matrix
    receive a zero vector (cold-start indicator).

    Args:
        train_df:             Training interactions DataFrame. Must contain
                              columns: user_id, video_id, watch_ratio.
        item_feature_matrix:  Item feature matrix from build_item_feature_matrix.
        id_to_idx:            video_id → row index mapping.
        weight_by_watch_ratio: If True, weight each item by watch_ratio.
                              If False, uniform weight (1.0 per interaction).
        normalize:            If True, L2-normalise each profile vector.

    Returns:
        dict[user_id, profile_vector] — one vector per user in train_df.
    """
    if item_feature_matrix:
        dim = len(item_feature_matrix[0])
    else:
        dim = 0

    # Group interactions by user
    user_items = (
        train_df.select(["user_id", "video_id", "watch_ratio"])
        .group_by("user_id")
        .agg([
            pl.col("video_id").alias("video_ids"),
            pl.col("watch_ratio").alias("watch_ratios"),
        ])
    )

    profiles: dict[int, list[float]] = {}

    for row in user_items.iter_rows(named=True):
        uid: int = row["user_id"]
        video_ids: list[int] = row["video_ids"]
        watch_ratios: list[float] = row["watch_ratios"]

        vectors = []
        weights = []
        for vid, wr in zip(video_ids, watch_ratios):
            if vid in id_to_idx:
                vectors.append(item_feature_matrix[id_to_idx[vid]])
                weights.append(float(wr) if weight_by_watch_ratio else 1.0)

        if not vectors:
            # Cold-start: user has no items in the feature matrix
            profiles[uid] = [0.0] * dim
            continue

        profile = _weighted_average(vectors, weights)

        if normalize:
            profile = _l2_normalize(profile)

        profiles[uid] = profile

    return profiles
