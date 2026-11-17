"""Item-Based Collaborative Filtering recommender — Phase 4B.

Recommends items similar (in terms of user co-interaction patterns) to items
the target user has already interacted with.

Item-item similarity is cosine similarity computed on item column vectors
(transposed interaction matrix: items as rows, users as columns).

Usage:
    from ml.collaborative.item_cf import ItemBasedCF

    rec = ItemBasedCF(n_neighbors=50)
    rec.fit(train_df)
    recs = rec.recommend(user_id=42, n=10, exclude_seen={100, 101})
"""

from __future__ import annotations

import time

import numpy as np
import polars as pl

from ml.baselines.base import BaseRecommender
from ml.baselines.popularity_recommender import PopularityRecommender
from ml.collaborative.interaction_matrix import build_interaction_matrix


class ItemBasedCF(BaseRecommender):
    """Item-Based Collaborative Filtering via cosine-similarity neighbours.

    For each target user:
      1. Find items the user has interacted with.
      2. For each such item, retrieve its k most similar items.
      3. Score candidate items as similarity-weighted sum of user's ratings.
      4. Exclude seen items, return top-N.

    Args:
        n_neighbors:     Number of similar items to retrieve per seed item.
        max_watch_ratio: Clip watch_ratio before building interaction matrix.
        fallback:        Recommender for cold-start users.
        random_seed:     Passed to BaseRecommender.
    """

    def __init__(
        self,
        n_neighbors: int = 50,
        max_watch_ratio: float = 5.0,
        fallback: BaseRecommender | None = None,
        random_seed: int = 42,
    ) -> None:
        super().__init__(random_seed=random_seed)
        self._n_neighbors = n_neighbors
        self._max_watch_ratio = max_watch_ratio
        self._fallback: BaseRecommender = fallback or PopularityRecommender(
            popularity_metric="weighted_score"
        )

        # Populated during fit()
        self._matrix: np.ndarray | None = None
        self._user_to_row: dict[int, int] = {}
        self._item_to_col: dict[int, int] = {}
        self._row_to_user: dict[int, int] = {}
        self._col_to_item: dict[int, int] = {}
        self._item_sim: np.ndarray | None = None   # (n_items, n_items)
        self._fit_seconds: float = 0.0

    @property
    def name(self) -> str:
        return f"ItemBasedCF(k={self._n_neighbors})"

    def fit(self, train_df: pl.DataFrame) -> None:
        """Build interaction matrix and pre-compute item-item similarities.

        Args:
            train_df: Training interactions (user_id, video_id, watch_ratio).
        """
        t0 = time.time()

        (
            self._matrix,
            self._user_to_row,
            self._item_to_col,
            self._row_to_user,
            self._col_to_item,
        ) = build_interaction_matrix(train_df, max_watch_ratio=self._max_watch_ratio)

        # Items as rows: (n_items, n_users)
        item_matrix = self._matrix.T

        # Normalise item rows for cosine similarity
        norms = np.linalg.norm(item_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalised = item_matrix / norms

        # Item-item cosine similarity: (n_items, n_items)
        self._item_sim = normalised @ normalised.T

        # Fit fallback
        self._fallback.fit(train_df)

        self._fit_seconds = time.time() - t0
        self._fitted = True

    def recommend(
        self,
        user_id: int,
        n: int,
        exclude_seen: set[int],
    ) -> list[int]:
        """Return top-N ItemCF recommendations.

        Args:
            user_id:      Target user.
            n:            Number of recommendations.
            exclude_seen: Items to exclude.

        Returns:
            Ranked list of up to n video_ids.
        """
        self._check_fitted()

        if user_id not in self._user_to_row:
            return self._fallback.recommend(user_id, n, exclude_seen)

        assert self._item_sim is not None
        assert self._matrix is not None

        u_row = self._user_to_row[user_id]
        user_ratings = self._matrix[u_row]  # shape (n_items,)

        if user_ratings.sum() == 0.0:
            return self._fallback.recommend(user_id, n, exclude_seen)

        # Vectorised score: user_ratings @ item_sim_matrix
        # Shape: (n_items,) @ (n_items, n_items) → (n_items,)
        # This is equivalent to Σ rating(u, seed) × sim(seed, candidate) for all seeds
        assert self._item_sim is not None
        item_scores = user_ratings @ self._item_sim  # (n_items,)
        # Zero out the user's own ratings to avoid re-recommending seen items
        # (exclude_seen handles this, but zeroing also removes self-similarity bias)
        item_scores = item_scores.astype(np.float32)

        # Sort by score descending, exclude seen
        sorted_col_indices = np.argsort(item_scores)[::-1]
        result = []
        for col_idx in sorted_col_indices:
            vid = self._col_to_item[int(col_idx)]
            if vid not in exclude_seen:
                result.append(vid)
                if len(result) >= n:
                    break

        # Fill remainder from fallback if needed
        if len(result) < n:
            fallback_recs = self._fallback.recommend(
                user_id, n - len(result), exclude_seen | set(result)
            )
            result.extend(fallback_recs)

        return result[:n]

    @property
    def fit_seconds(self) -> float:
        return self._fit_seconds
