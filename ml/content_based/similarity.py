"""Cosine similarity computation for Content-Based Filtering.

Provides:
  - cosine_similarity_pair    : similarity between two vectors (pure Python, for unit tests)
  - score_items_for_vector    : score all items vs a query vector (numpy-accelerated)
  - top_k_similar_items       : ranked list of most similar items for a query vector
  - build_score_matrix        : pre-compute all user × item scores at once (batch, fast)

Performance note:
    Pure-Python loops over 1,411 users × 3,327 items × ~760 feature dims would be
    ~3.6 billion float ops — impractically slow in CPython.
    We use numpy (available via scikit-learn) for matrix operations:
      - build_score_matrix converts the item matrix and all user profiles to numpy arrays
        and runs a single dot-product: scores = profiles @ item_matrix.T  (shape: U × N)
      - Per-user top-K extraction then uses numpy argpartition for O(N) partial sort.
    This reduces evaluation time from ~hours to ~seconds.

Usage:
    from ml.content_based.similarity import build_score_matrix, top_k_similar_items
"""

from __future__ import annotations

import math

import numpy as np


def cosine_similarity_pair(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors (pure Python).

    Used in unit tests. For bulk scoring use build_score_matrix.

    Args:
        a: First vector.
        b: Second vector (same length as a).

    Returns:
        Cosine similarity in [-1, 1]. Returns 0.0 if either vector is zero.
    """
    if len(a) != len(b):
        msg = f"Vector length mismatch: {len(a)} vs {len(b)}"
        raise ValueError(msg)

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def score_items_for_vector(
    query_vector: list[float],
    item_matrix: list[list[float]],
    idx_to_id: dict[int, int],
) -> list[tuple[int, float]]:
    """Score all items in the matrix against a single query vector.

    Uses numpy for fast dot-product computation.
    Assumes item_matrix rows are already L2-normalised (from build_item_feature_matrix).

    Args:
        query_vector:  Feature vector for the query. Does NOT need to be normalised.
        item_matrix:   List of item feature vectors (n_items × n_features).
        idx_to_id:     Row index → video_id mapping.

    Returns:
        List of (video_id, score) tuples, unsorted.
    """
    q = np.array(query_vector, dtype=np.float32)
    norm_q = float(np.linalg.norm(q))

    n_items = len(item_matrix)
    if norm_q == 0.0:
        return [(idx_to_id[i], 0.0) for i in range(n_items)]

    q_norm = q / norm_q
    item_arr = np.array(item_matrix, dtype=np.float32)  # shape (N, D)
    dot_scores = item_arr @ q_norm  # shape (N,)

    return [(idx_to_id[i], float(dot_scores[i])) for i in range(n_items)]


def build_score_matrix(
    user_profiles: dict[int, list[float]],
    item_matrix: list[list[float]],
    user_id_order: list[int],
) -> np.ndarray:
    """Pre-compute scores for all users against all items in one matrix multiply.

    Both user profiles and item vectors should already be L2-normalised so that
    dot product equals cosine similarity.

    Args:
        user_profiles:  dict of user_id → profile vector.
        item_matrix:    List of n_items feature vectors (n_items × n_features).
        user_id_order:  Ordered list of user_ids whose rows to include.

    Returns:
        numpy array of shape (n_users, n_items) with scores[u, i] = similarity of
        user u with item i. Users with zero profiles get all-zero rows.
    """
    if not item_matrix or not user_id_order:
        return np.zeros((len(user_id_order), len(item_matrix)), dtype=np.float32)

    item_arr = np.array(item_matrix, dtype=np.float32)  # (N, D)
    n_users = len(user_id_order)
    dim = item_arr.shape[1]

    profile_arr = np.zeros((n_users, dim), dtype=np.float32)
    for u_idx, uid in enumerate(user_id_order):
        profile = user_profiles.get(uid)
        if profile is not None:
            profile_arr[u_idx] = profile

    # (n_users, D) @ (D, n_items) → (n_users, n_items)
    scores: np.ndarray = profile_arr @ item_arr.T
    return scores


def top_k_similar_items(
    query_vector: list[float],
    item_matrix: list[list[float]],
    idx_to_id: dict[int, int],
    k: int,
    exclude_ids: set[int] | None = None,
) -> list[int]:
    """Return the top-K most similar items to a query vector.

    Args:
        query_vector: Feature vector for the query (user profile or item vector).
        item_matrix:  List of item feature vectors (n_items × n_features).
        idx_to_id:    Row index → video_id mapping.
        k:            Number of items to return.
        exclude_ids:  Set of video_ids to exclude from results.

    Returns:
        Sorted list of up to k video_ids (best match first).
    """
    if k <= 0:
        return []

    exclude = exclude_ids or set()
    scored = score_items_for_vector(query_vector, item_matrix, idx_to_id)

    # Sort descending by score
    scored.sort(key=lambda x: x[1], reverse=True)

    result = []
    for vid, _score in scored:
        if vid not in exclude:
            result.append(vid)
            if len(result) >= k:
                break
    return result
